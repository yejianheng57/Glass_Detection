# 模型脚本说明

本目录只放训练和推理相关脚本：

```text
scripts/model/
  train_yolo_seg.py
  predict_yolo_seg.py
  infer_realsense_yolo_seg.py
```

模型阶段负责：

```text
YOLO Seg 数据集
  -> 训练 YOLO26 Seg
  -> 离线预测验证
  -> D435i 实时推理
```

训练前需要先完成数据整理和检查，详见 `scripts/data/README.md`。

## 模型训练

默认训练 `yolo26n-seg.pt`：

```bash
python scripts/model/train_yolo_seg.py
```

默认参数：

- `--data datasets/glass_rect/data.yaml`
- `--model yolo26n-seg.pt`
- `--epochs 100`
- `--imgsz 640`
- `--batch 8`
- `--workers 0`

建议先用 `n` 模型跑通流程。如果 `n` 的漏检较多或 mask 不够贴边，再尝试 `s`：

```bash
python scripts/model/train_yolo_seg.py \
  --model yolo26s-seg.pt \
  --name glass_hole_yolo26s \
  --epochs 100 \
  --imgsz 640 \
  --batch 8
```

指定显卡：

```bash
python scripts/model/train_yolo_seg.py --device 0
```

显存不够时降低 batch：

```bash
python scripts/model/train_yolo_seg.py --batch 4
python scripts/model/train_yolo_seg.py --batch 2
```

训练完成后，默认权重路径类似：

```text
runs/segment/glass_hole_yolo26n/weights/best.pt
```

## 离线预测验证

用验证集或新采集图片检查分割效果：

```bash
python scripts/model/predict_yolo_seg.py \
  --weights runs/segment/glass_hole_yolo26n/weights/best.pt \
  --source datasets/glass_rect/images/val
```

输出内容：

```text
runs/predict/glass_hole/overlays/
runs/predict/glass_hole/predictions.csv
```

其中：

- `overlays/` 保存画好检测框和分割掩膜的图片。
- `predictions.csv` 保存每个空洞实例的置信度、检测框、中心点、面积、角度和多边形点。

常用参数：

```bash
python scripts/model/predict_yolo_seg.py --conf 0.35 --iou 0.7
python scripts/model/predict_yolo_seg.py --source datasets/glass_rect/raw_images
```

## 实时推理

连接 D435i 后运行：

```bash
python scripts/model/infer_realsense_yolo_seg.py \
  --weights runs/segment/glass_hole_yolo26n/weights/best.pt
```

窗口快捷键：

- `q` 或 `Esc`：退出。
- `s`：保存当前原图和叠加结果到 `runs/realsense/glass_hole/`。

常用参数：

```bash
python scripts/model/infer_realsense_yolo_seg.py --conf 0.35
python scripts/model/infer_realsense_yolo_seg.py --width 1280 --height 720 --fps 30
python scripts/model/infer_realsense_yolo_seg.py --device 0
```

实时推理输出：

```text
D435i RGB 图像
  -> YOLO26 Seg best.pt
  -> 每个空洞的 box、confidence、mask
  -> 轮廓中心点、面积、方向等几何结果
```

## 算力建议

- 第一阶段优先使用 `yolo26n-seg.pt`。
- 如果 `n` 效果不足，再训练 `yolo26s-seg.pt`。
- 训练建议使用 NVIDIA GPU；只有 CPU 时可以跑推理验证，但训练会很慢。
- 实时部署优先看稳定帧率和漏检率，如果 `n` 已满足要求，不必上更大的模型。

## 后续机械臂坐标

当前模型只输出图像中的分割结果。如果要服务机械臂，还需要在推理结果之后增加：

- 相机内参标定。
- 镜头畸变校正。
- 固定相机到桌面平面的单应性变换，或机械臂末端相机的手眼标定。
- 图像像素坐标到桌面坐标系或机械臂坐标系的转换。
