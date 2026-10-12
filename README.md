# 电子秤商品图像分类：训练与验证

用于在另一台电脑上训练 MobileNetV3-Large / EfficientNet-B0，查看初步验证效果。此仓库没有训练完成的专属模型；训练结束后会生成权重和 ONNX。

## 数据内容

- 从称端验收照片中整理出 **731 张图片、142 个商品目录**。
- 按商品类别合并原“合格”和“不合格”目录；这里的合格与不合格是验收状态，不是模型分类标签。保留其来源状态供人工复核；图片可解码并不保证商品标签一定正确。
- 排除 27 张无商品归属照片，排除 15 张同类重复或跨类标签冲突的图片。具体记录见 `metadata/dataset_summary.json`。
- `metadata/photo_manifest.csv` 保存每张图片的来源、类别、尺寸和像素哈希；`metadata/class_counts.csv` 保存实际图片数量。
- 未包含本机虚拟环境、两类假数据、假数据训练产物及板端部署文件。

## 克隆与安装

```bash
git clone https://github.com/caoyunli57-maker/zhongzhuan.git
cd zhongzhuan
python -m venv .venv
```

Windows PowerShell 使用下面的 Python 路径；Linux 使用 `.venv/bin/python` 替换即可。

先按训练电脑的显卡与驱动，在 https://pytorch.org/get-started/locally/ 选择 PyTorch 安装命令，确保安装到 `.venv` 中。只用 CPU 做流程测试时：

```powershell
.\.venv\Scripts\python.exe -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -c "import torch; print('CUDA:', torch.cuda.is_available())"
```

原项目使用 PyTorch 2.6.0 / torchvision 0.21.0。建议先保持这一组合；使用 NVIDIA GPU 时安装对应兼容的 CUDA 版本。

## 开始训练

先运行 1 轮确认环境与数据读取正常：

```powershell
.\.venv\Scripts\python.exe train.py --data dataset --out runs/check --epochs 1 --batch-size 16 --num-workers 0
```

再运行正式的初步训练验证：

```powershell
.\.venv\Scripts\python.exe train.py --data dataset --out runs/mobilenet --epochs 30 --batch-size 16 --num-workers 0 --seed 42
```

对比 EfficientNet-B0：

```powershell
.\.venv\Scripts\python.exe train.py --data dataset --out runs/efficientnet --model efficientnet_b0 --epochs 30 --batch-size 16 --num-workers 0 --seed 42
```

首次运行需要联网下载 ImageNet 预训练权重。显存不足时减小 `--batch-size`。此处目录名虽来源于 `D:\yolo`，项目实际做的是整图分类，无需 YOLO 检测框标注。

## 查看结果和解释限制

输出在所选 `runs/` 子目录：`best_model.pth`、`last_model.pth`、`model.onnx`、`labels.txt`、`train_report.json`。报告包含验证准确率、每类验证准确率和易混淆类别。

当前脚本随机按类划分训练和验证，默认验证比例 10%；**52 类达到至少 5 张，另外 90 类只进入训练、不参与验证**。所以总验证准确率不代表全部 142 类的识别能力。没有独立测试集；相近时间/同批商品照片也可能造成结果偏乐观。正式验收前需要补采并按批次/日期划分独立测试集。

`metadata/` 在数据目录外，避免被 ImageFolder 当作商品类别。不要把 `合格/不合格` 当作训练标签，也不要把生成的模型或虚拟环境提交到普通 Git。
