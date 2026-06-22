# 模型脚本

本目录包含 YOLO Seg 训练、离线预测和 D435i 实时推理脚本。

```text
scripts/model/
  train_yolo_seg.py              # 训练 YOLO26 Seg
  predict_yolo_seg.py            # 离线预测并导出结果
  infer_realsense_yolo_seg.py    # D435i 实时推理
```

训练前先完成数据整理和检查，见 `scripts/data/README.md`。

## 推荐训练命令

D435i RGB 为 `1280x720`，推荐使用接近原始比例且对齐 YOLO 步长的矩形输入 `736 1280`，顺序是 `height width`，或者直接--imgsz 960 （代表压缩到960*960）。

```bash
python scripts/model/train_yolo_seg.py \
  --data datasets/glass_rect/data.yaml \
  --device 0,1,2 \
  --model ./yolo26n-seg.pt \
  --name glass_hole_yolo26n_1280x736_e150 \
  --epochs 150 \
  --imgsz 736 1280 \
  --batch 24 \
  --workers 8
```

训练完成后权重通常在：

```text
runs/segment/glass_hole_yolo26n_1280x736_e150/weights/best.pt
```

## 离线预测

```bash
python scripts/model/predict_yolo_seg.py \
  --weights runs/segment/glass_hole_yolo26n_1280x736_e150/weights/best.pt \
  --source datasets/glass_rect/images/val \
  --imgsz 736 1280 \
  --conf 0.25 \
  --device 0
```

输出：

```text
runs/predict/glass_hole/overlays/       # 叠加 mask 和 box 的图片
runs/predict/glass_hole/predictions.csv # 置信度、box、中心点、面积、角度、多边形点
```

## D435i 实时推理

```bash
python scripts/model/infer_realsense_yolo_seg.py \
  --weights runs/segment/glass_hole_yolo26n_1280x736_e150/weights/best.pt \
  --width 1280 \
  --height 720 \
  --fps 30 \
  --imgsz 736 1280 \
  --conf 0.25 \
  --device 0
```

窗口快捷键：

- `s`：保存当前原图和叠加结果到 `runs/realsense/glass_hole/`。
- `q` 或 `Esc`：退出。

## 其他常用参数

```bash
# 方形输入
python scripts/model/train_yolo_seg.py --imgsz 640

# 显存不足时降低 batch
python scripts/model/train_yolo_seg.py --batch 12
python scripts/model/train_yolo_seg.py --batch 8

# 尝试更大的模型
python scripts/model/train_yolo_seg.py \
  --model ./yolo26s-seg.pt \
  --name glass_hole_yolo26s_960_e150 \
  --epochs 150 \
  --imgsz 960 \
  --batch 12
```

## 选择建议

- 优先使用 `yolo26n-seg.pt`，当前 `736x1280` 训练结果综合最好。
- 如果漏检明显或 mask 不够贴边，再尝试 `yolo26s-seg.pt` 或更大输入尺寸。
- 实时部署优先看帧率、漏检率和误检率；如果 `n` 已满足要求，不必上更大的模型。
- 若要用于机械臂，还需要相机标定、畸变校正、手眼标定或平面坐标转换。

