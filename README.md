# Glass_Detection

玻璃空洞实例分割项目。方案见 `require.md`，脚本总入口见 `scripts/README.md`。

脚本已经按用途拆分：

- 数据相关：`scripts/data/README.md`
- 训练和推理相关：`scripts/model/README.md`

采集和标注完成后的主流程：

```bash
pip install -r requirements.txt
python scripts/data/prepare_yolo_dataset.py --clean
python scripts/data/check_yolo_seg_dataset.py --check-images
python scripts/model/train_yolo_seg.py
python scripts/model/predict_yolo_seg.py --weights runs/segment/glass_hole_yolo26n/weights/best.pt
python scripts/model/infer_realsense_yolo_seg.py --weights runs/segment/glass_hole_yolo26n/weights/best.pt
```

默认数据目录：

```text
datasets/glass_rect/
  raw_images/
  raw_labels/
  images/train/
  images/val/
  labels/train/
  labels/val/
  data.yaml
```