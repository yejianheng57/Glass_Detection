# Glass Detection

玻璃孔洞实例分割项目，使用 YOLO Seg 完成数据采集、标注、训练、离线预测和 D435i 实时推理。

## 目录

- 数据脚本：`scripts/data/README.md`
- 模型脚本：`scripts/model/README.md`
- 需求说明：`require.md`

## 快速流程

```bash
pip install -r requirements.txt

python scripts/data/prepare_yolo_dataset.py --clean
python scripts/data/check_yolo_seg_dataset.py --check-images

python scripts/model/train_yolo_seg.py
python scripts/model/predict_yolo_seg.py --weights runs/segment/glass_hole_yolo26n/weights/best.pt
python scripts/model/infer_realsense_yolo_seg.py --weights runs/segment/glass_hole_yolo26n/weights/best.pt
```

## 数据目录

```text
datasets/glass_rect/
  raw_images/       # 原始图片
  raw_labels/       # 原始标注
  images/train/     # YOLO 训练图片
  images/val/       # YOLO 验证图片
  labels/train/     # YOLO 训练标签
  labels/val/       # YOLO 验证标签
  data.yaml
```