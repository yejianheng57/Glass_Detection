# 数据脚本

本目录包含数据采集、标注、整理和检查脚本。

```text
scripts/data/
  collect_realsense_rgb.py       # D435i RGB 采集
  annotate_rect_holes.py         # 多边形标注
  prepare_yolo_dataset.py        # 整理 YOLO Seg 数据集
  check_yolo_seg_dataset.py      # 检查图片和标签
```

## 流程

```bash
# 1. 采集 D435i RGB 图片
python scripts/data/collect_realsense_rgb.py --width 1280 --height 720 --fps 30

# 2. 标注玻璃孔洞多边形
python scripts/data/annotate_rect_holes.py

# 3. 整理为 YOLO Seg 数据集
python scripts/data/prepare_yolo_dataset.py --clean

# 4. 检查数据集
python scripts/data/check_yolo_seg_dataset.py --check-images
```

## 默认目录

```text
datasets/glass_rect/
  raw_images/       # 原始图片
  raw_labels/       # 原始 YOLO Seg 标注
  images/train/
  images/val/
  labels/train/
  labels/val/
  data.yaml
```

`prepare_yolo_dataset.py` 支持两种输入：直接读取 `raw_images/`、`raw_labels/`，或扫描 `datasets/glass_rect/*/raw_images` 和 `raw_labels` 形式的多场景目录。多场景合并时会给文件名加场景前缀，避免重名覆盖。

## 标注说明

YOLO Seg 标签格式：

```text
class_id x1 y1 x2 y2 x3 y3 ... xN yN
```

- `class_id` 固定为 `0`。
- 坐标归一化到 `0-1`。
- 每个孔洞实例至少 3 个点。
- 同一个实例的点应沿边界连续点击，不要交叉跳点。

标注快捷键：

- `f` 或 `Enter`：结束当前实例。
- `s`：保存当前图片标签。
- `u`：撤销上一个点。
- `d`：删除当前未完成实例或最后一个完整实例。
- `n` / `p`：下一张 / 上一张。
- `q` 或 `Esc`：退出，修改会自动保存。

## 常用命令

```bash
# 自定义采集路径
python scripts/data/collect_realsense_rgb.py \
  --output-dir datasets/glass_rect/raw_images \
  --log-file datasets/glass_rect/capture_log.csv \
  --prefix glass

# 自定义标注路径
python scripts/data/annotate_rect_holes.py \
  --images-dir datasets/glass_rect/raw_images \
  --labels-dir datasets/glass_rect/raw_labels

# 合并指定场景
python scripts/data/prepare_yolo_dataset.py \
  --scene-dirs \
    datasets/glass_rect/glass_rect_END_GREEN_clean \
    datasets/glass_rect/glass_rect_UP_RED_dirty_arm \
  --output-dir datasets/glass_rect \
  --val-ratio 0.2 \
  --clean
```

## 检查重点

如果检查脚本报错，通常需要回到标注阶段修正：

- `missing label file`
- `coordinates must be normalized to 0-1`
- `needs class_id and at least 3 x/y pairs`
- `orphan label without image`
