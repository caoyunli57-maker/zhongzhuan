#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
商品识别模型训练脚本 —— RK3576 电子秤本地识别
================================================
功能: 迁移学习微调 MobileNetV3-Large / EfficientNet-B0, 支持混合精度、
      类别不平衡采样、数据增强、最优权重保存、易混淆类别分析、导出 ONNX。

用法示例:
  python train.py --data D:\\scale_dataset --epochs 30
  python train.py --data D:\\scale_dataset --model efficientnet_b0 --batch-size 48
  python train.py --data D:\\scale_dataset --no-hflip     # 包装文字多时建议关闭水平翻转

数据集目录结构 (每个子文件夹 = 一个类别, 文件夹名即类别名):
  D:\\scale_dataset\\
    ├── apple\\        000001.jpg ...
    ├── banana\\       ...
    └── pork_belly\\   ...

产出 (全部写入 --out 目录):
  best_model.pth     验证集准确率最高的权重
  last_model.pth     最后一轮权重
  model.onnx         用于转 .rknn 的中间格式
  labels.txt         类别顺序表 (部署时必须一起用, 顺序错则全错)
  calib_list.txt     INT8 量化校准图清单 (转换时用)
  train_report.json  训练结果报告
"""
import argparse
import collections
import json
import os
import random
import sys
import time

# ---- 防御性修复 ----
# 某些运行环境会通过 PYTHONPATH 注入一个外部资源包目录, 其中的 PIL 副本
# 已损坏(缺少 _imaging 二进制模块), 且优先级高于本项目虚拟环境, 导致导入
# torchvision 时报 "cannot import name '_imaging' from 'PIL'"。
# 此处在加载任何第三方库之前, 把这类注入路径从模块搜索路径中剔除,
# 确保使用虚拟环境自带的完整依赖。仅移除同时含
# resources/packages/pip/sets 特征的注入目录, 不会误删正常路径。
_INJECT_MARKERS = {"resources", "packages", "pip", "sets"}
sys.path[:] = [
    p for p in sys.path
    if not (_INJECT_MARKERS <= set(
        seg for seg in p.replace("\\", "/").split("/") if seg))
]

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from torchvision import datasets, models, transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
# RKNN 转换时使用的 0-255 尺度归一化参数, 必须与此处等价
RKNN_MEAN = [123.675, 116.28, 103.53]
RKNN_STD = [58.395, 57.12, 57.375]


def parse_args():
    p = argparse.ArgumentParser(description="RK3576 电子秤商品识别训练")
    p.add_argument("--data", required=True, help="数据集根目录 (子文件夹为类别)")
    p.add_argument("--out", default=None, help="产出目录, 默认 <data>/../train_output")
    p.add_argument("--model", default="mobilenet_v3_large",
                   choices=["mobilenet_v3_large", "efficientnet_b0", "mobilenet_v2"],
                   help="骨干网络, 默认 mobilenet_v3_large (NPU 支持最好)")
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=32, help="8G 显存建议 32, 224 分辨率")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--label-smoothing", type=float, default=0.1)
    p.add_argument("--val-ratio", type=float, default=0.1, help="验证集比例")
    p.add_argument("--freeze-epochs", type=int, default=1,
                   help="先冻结骨干网络训练的轮数, 稳定小数据集, 设 0 关闭")
    p.add_argument("--no-hflip", action="store_true",
                   help="关闭水平翻转增强 (包装上文字多时建议开启此项)")
    p.add_argument("--no-balanced", action="store_true", help="关闭类别平衡采样")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--calib-count", type=int, default=200, help="量化校准图数量")
    p.add_argument("--resume", default=None, help="从权重文件继续训练")
    p.add_argument("--export-only", action="store_true",
                   help="只做 ONNX 导出, 不训练 (配合 --resume 使用)")
    return p.parse_args()


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_model(name, num_classes):
    if name == "mobilenet_v3_large":
        m = models.mobilenet_v3_large(
            weights=models.MobileNet_V3_Large_Weights.IMAGENET1K_V1)
        m.classifier[3] = nn.Linear(m.classifier[3].in_features, num_classes)
    elif name == "mobilenet_v2":
        m = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, num_classes)
    elif name == "efficientnet_b0":
        m = models.efficientnet_b0(
            weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, num_classes)
    else:
        raise ValueError(name)
    return m


def build_transforms(img_size, no_hflip):
    flips = [] if no_hflip else [transforms.RandomHorizontalFlip()]
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.65, 1.0), ratio=(0.85, 1.18)),
        *flips,
        transforms.RandomApply(
            [transforms.ColorJitter(0.30, 0.30, 0.30, 0.04)], p=0.7),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        transforms.RandomErasing(p=0.25, scale=(0.02, 0.18)),
    ])
    val_tf = transforms.Compose([
        transforms.Resize(int(img_size * 1.14)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    return train_tf, val_tf


def stratified_split(full_ds, val_ratio, seed):
    """按类别分层切分, 保证每个类别都同时出现在训练集和验证集"""
    by_class = collections.defaultdict(list)
    for idx, (_, label) in enumerate(full_ds.samples):
        by_class[label].append(idx)
    rng = random.Random(seed)
    train_idx, val_idx, tiny = [], [], []
    for label in sorted(by_class):
        idxs = by_class[label][:]
        rng.shuffle(idxs)
        if len(idxs) < 5:
            # 样本太少, 全部进训练集, 不参与验证 (避免验证集污染)
            train_idx.extend(idxs)
            tiny.append((full_ds.classes[label], len(idxs)))
            continue
        n_val = max(1, int(round(len(idxs) * val_ratio)))
        val_idx.extend(idxs[:n_val])
        train_idx.extend(idxs[n_val:])
    return train_idx, val_idx, tiny


def set_requires_grad(model, name, flag):
    if name == "mobilenet_v3_large":
        blocks = model.features
    elif name == "mobilenet_v2":
        blocks = model.features
    else:
        blocks = model.features
    for p in blocks.parameters():
        p.requires_grad = flag


@torch.no_grad()
def evaluate(model, loader, device, num_classes):
    model.eval()
    correct = total = 0
    per_cls_hit = np.zeros(num_classes, dtype=np.int64)
    per_cls_cnt = np.zeros(num_classes, dtype=np.int64)
    conf = np.zeros((num_classes, num_classes), dtype=np.int64)
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            out = model(x)
        pred = out.argmax(1)
        correct += (pred == y).sum().item()
        total += y.size(0)
        np.add.at(per_cls_hit, y.cpu().numpy()[
            (pred == y).cpu().numpy()], 1)
        np.add.at(per_cls_cnt, y.cpu().numpy(), 1)
        np.add.at(conf, (y.cpu().numpy(), pred.cpu().numpy()), 1)
    acc = correct / max(total, 1)
    with np.errstate(invalid="ignore", divide="ignore"):
        per_cls_acc = np.where(per_cls_cnt > 0, per_cls_hit /
                               np.maximum(per_cls_cnt, 1), np.nan)
    return acc, per_cls_acc, per_cls_cnt, conf


def top_confusions(conf, classes, k=15):
    """输出最容易混淆的类别对, 指导数据补采"""
    pairs = []
    n = conf.shape[0]
    for i in range(n):
        for j in range(n):
            if i != j and conf[i, j] > 0:
                pairs.append((int(conf[i, j]), classes[i], classes[j]))
    pairs.sort(reverse=True)
    return pairs[:k]


def export_onnx(model, path, img_size, device):
    model.eval()
    model.cpu()
    dummy = torch.randn(1, 3, img_size, img_size)
    torch.onnx.export(
        model, dummy, path,
        input_names=["input"], output_names=["output"],
        opset_version=12, do_constant_folding=True,
    )
    print("[ONNX] 已导出: %s" % path)


def main():
    args = parse_args()
    seed_all(args.seed)

    if not os.path.isdir(args.data):
        raise SystemExit("[错误] 数据集目录不存在: %s" % args.data)
    out_dir = args.out or os.path.join(
        os.path.dirname(os.path.abspath(args.data.rstrip("\\/"))), "train_output")
    os.makedirs(out_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 62)
    print("设备      : %s" % device)
    if device.type == "cuda":
        print("显卡      : %s" % torch.cuda.get_device_name(0))
        print("显存      : %.1f GB" %
              (torch.cuda.get_device_properties(0).total_memory / 1024**3))
    print("数据集    : %s" % args.data)
    print("产出目录  : %s" % out_dir)
    print("骨干网络  : %s" % args.model)
    print("=" * 62)

    train_tf, val_tf = build_transforms(args.img_size, args.no_hflip)
    full = datasets.ImageFolder(args.data)
    classes = full.classes
    num_classes = len(classes)
    if num_classes < 2:
        raise SystemExit("[错误] 至少要 2 个类别, 当前 %d 个。请在数据集目录下建立类别子文件夹。"
                         % num_classes)

    train_idx, val_idx, tiny = stratified_split(full, args.val_ratio, args.seed)
    if tiny:
        print("[警告] 以下类别样本过少 (<5 张), 已全部归入训练集且不参与验证:")
        for name, cnt in tiny:
            print("        %s: %d 张" % (name, cnt))

    train_ds = Subset(datasets.ImageFolder(args.data, transform=train_tf), train_idx)
    val_ds = Subset(datasets.ImageFolder(args.data, transform=val_tf), val_idx)
    print("类别数    : %d" % num_classes)
    print("训练样本  : %d   验证样本: %d" % (len(train_ds), len(val_ds)))

    counts = collections.Counter(l for _, l in
                                 [full.samples[i] for i in train_idx])
    with open(os.path.join(out_dir, "labels.txt"), "w", encoding="utf-8") as f:
        for c in classes:
            f.write(c + "\n")
    print("[已保存] labels.txt (类别顺序表, 部署时必须配套使用)")

    sampler = None
    if not args.no_balanced and len(set(counts.values())) > 1:
        weights = [1.0 / counts[l] for _, l in
                   [full.samples[i] for i in train_idx]]
        sampler = WeightedRandomSampler(
            weights, num_samples=len(train_ds), replacement=True)
        print("类别平衡  : 已启用 (自动补偿样本少的类别)")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              sampler=sampler, shuffle=(sampler is None),
                              num_workers=args.num_workers, pin_memory=True,
                              drop_last=True if len(train_ds) > args.batch_size else False)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)

    model = build_model(args.model, num_classes).to(device)
    if args.resume and os.path.isfile(args.resume):
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt)
        print("[已加载] 权重: %s" % args.resume)

    if args.export_only:
        export_onnx(model, os.path.join(out_dir, "model.onnx"),
                    args.img_size, device)
        return

    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    head_params, base_params = [], []
    for n, p in model.named_parameters():
        (head_params if "classifier" in n or "fc" in n else base_params).append(p)
    optimizer = torch.optim.AdamW([
        {"params": base_params, "lr": args.lr * 0.1},
        {"params": head_params, "lr": args.lr},
    ], weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)
    scaler = torch.amp.GradScaler(enabled=device.type == "cuda")

    best_acc, best_epoch = 0.0, -1
    history = []
    t_start = time.time()

    for epoch in range(1, args.epochs + 1):
        freeze = args.freeze_epochs > 0 and epoch <= args.freeze_epochs
        set_requires_grad(model, args.model, not freeze)
        model.train()
        run_loss, run_hit, run_cnt, ep_t = 0.0, 0, 0, time.time()

        for x, y in train_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type,
                                enabled=device.type == "cuda"):
                out = model(x)
                loss = criterion(out, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            run_loss += loss.item() * y.size(0)
            run_hit += (out.argmax(1) == y).sum().item()
            run_cnt += y.size(0)

        scheduler.step()
        train_acc = run_hit / max(run_cnt, 1)
        val_acc, per_cls_acc, per_cls_cnt, conf = evaluate(
            model, val_loader, device, num_classes)
        cur_lr = optimizer.param_groups[-1]["lr"]
        history.append({"epoch": epoch, "train_acc": round(train_acc, 4),
                        "val_acc": round(val_acc, 4), "loss": round(run_loss / max(run_cnt, 1), 4)})

        flag = ""
        if val_acc > best_acc:
            best_acc, best_epoch = val_acc, epoch
            torch.save({"model": model.state_dict(), "classes": classes,
                        "img_size": args.img_size, "arch": args.model,
                        "epoch": epoch, "val_acc": val_acc},
                       os.path.join(out_dir, "best_model.pth"))
            flag = "  <- 最优, 已保存"
        print("轮次 %2d/%d | 冻结骨干:%s | 训练准确率 %.4f | 验证准确率 %.4f | "
              "损失 %.4f | 学习率 %.2e | %.1fs%s"
              % (epoch, args.epochs, "是" if freeze else "否", train_acc,
                 val_acc, run_loss / max(run_cnt, 1), cur_lr,
                 time.time() - ep_t, flag))

    elapsed = time.time() - t_start
    torch.save({"model": model.state_dict(), "classes": classes,
                "img_size": args.img_size, "arch": args.model},
               os.path.join(out_dir, "last_model.pth"))

    # 用最优权重做最终评估与导出
    best_path = os.path.join(out_dir, "best_model.pth")
    if os.path.isfile(best_path):
        model.load_state_dict(torch.load(best_path, map_location=device)["model"])
    val_acc, per_cls_acc, per_cls_cnt, conf = evaluate(
        model, val_loader, device, num_classes)

    print("\n" + "=" * 62)
    print("训练完成 | 耗时 %.1f 分钟 | 最优验证准确率 %.4f (第 %d 轮)"
          % (elapsed / 60, best_acc, best_epoch))
    print("=" * 62)
    print("\n各类别验证准确率:")
    order = sorted(range(num_classes),
                   key=lambda i: (np.nan_to_num(per_cls_acc[i], nan=1.0)))
    for i in order:
        if per_cls_cnt[i] > 0:
            mark = "  <- 需补数据" if per_cls_acc[i] < 0.85 else ""
            print("  %-24s %.4f  (%d 张)%s"
                  % (classes[i], per_cls_acc[i], per_cls_cnt[i], mark))

    confusions = top_confusions(conf, classes)
    if confusions:
        print("\n最易混淆的类别对 (优先补采这两类的对比样本):")
        for cnt, t, p in confusions:
            print("  %-18s 被误判为 %-18s %d 次" % (t, p, cnt))

    export_onnx(model, os.path.join(out_dir, "model.onnx"), args.img_size, device)

    # 生成 INT8 量化校准清单 (从验证集图片中取)
    calib_paths = []
    for i in val_idx:
        calib_paths.append(full.samples[i][0])
    random.Random(args.seed).shuffle(calib_paths)
    calib_paths = calib_paths[:args.calib_count]
    calib_file = os.path.join(out_dir, "calib_list.txt")
    with open(calib_file, "w", encoding="utf-8") as f:
        for p in calib_paths:
            f.write(p + "\n")
    print("[已保存] calib_list.txt (%d 张量化校准图)" % len(calib_paths))

    report = {
        "arch": args.model, "img_size": args.img_size,
        "num_classes": num_classes, "classes": classes,
        "train_samples": len(train_ds), "val_samples": len(val_ds),
        "epochs": args.epochs, "best_val_acc": round(best_acc, 4),
        "best_epoch": best_epoch, "final_val_acc": round(val_acc, 4),
        "elapsed_minutes": round(elapsed / 60, 1),
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "rknn_mean_values": RKNN_MEAN, "rknn_std_values": RKNN_STD,
        "history": history,
        "per_class_acc": {classes[i]: (None if np.isnan(per_cls_acc[i])
                                       else round(float(per_cls_acc[i]), 4))
                          for i in range(num_classes)},
        "top_confusions": [{"true": t, "pred": p, "count": c}
                           for c, t, p in confusions],
    }
    rp = os.path.join(out_dir, "train_report.json")
    with open(rp, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print("[已保存] train_report.json")
    print("\n下一步: 在 WSL2 中运行 convert_rknn.py 把 model.onnx 转为 .rknn")


if __name__ == "__main__":
    main()
