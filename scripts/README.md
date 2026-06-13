# 脚本说明

脚本按用途拆成两组：

```text
scripts/
  data/
    collect_realsense_rgb.py
    annotate_rect_holes.py
    prepare_yolo_dataset.py
    check_yolo_seg_dataset.py
    README.md
  model/
    train_yolo_seg.py
    predict_yolo_seg.py
    infer_realsense_yolo_seg.py
    README.md
```

## 数据相关

数据采集、人工标注、训练集整理和标签检查都在 `scripts/data/`。

详见 `scripts/data/README.md`。

常用流程：

```bash
python scripts/data/collect_realsense_rgb.py
python scripts/data/annotate_rect_holes.py
python scripts/data/prepare_yolo_dataset.py --clean
python scripts/data/check_yolo_seg_dataset.py --check-images
```

## 模型相关

YOLO26 Seg 训练、离线预测和 D435i 实时推理都在 `scripts/model/`。

详见 `scripts/model/README.md`。

常用流程：

```bash
python scripts/model/train_yolo_seg.py
python scripts/model/predict_yolo_seg.py --weights runs/segment/glass_hole_yolo26n/weights/best.pt
python scripts/model/infer_realsense_yolo_seg.py --weights runs/segment/glass_hole_yolo26n/weights/best.pt
```

## 数据流

```text
raw_images + raw_labels
  -> scripts/data/prepare_yolo_dataset.py
  -> images/train, images/val, labels/train, labels/val, data.yaml
  -> scripts/model/train_yolo_seg.py
  -> best.pt
  -> scripts/model/predict_yolo_seg.py 或 scripts/model/infer_realsense_yolo_seg.py
```

