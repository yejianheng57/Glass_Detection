# 数据脚本说明

本目录只放数据相关脚本：

```text
scripts/data/
  collect_realsense_rgb.py
  annotate_rect_holes.py
  prepare_yolo_dataset.py
  check_yolo_seg_dataset.py
```

数据阶段负责：

```text
D435i RGB 采集
  -> 人工多边形标注
  -> 整理 YOLO Seg 数据集
  -> 检查图片和标签
```

## 数据采集

连接 Intel RealSense D435i 后运行：

```bash
python scripts/data/collect_realsense_rgb.py
```

默认行为：

- 图片保存到 `datasets/glass_rect/raw_images/`。
- 采集日志保存到 `datasets/glass_rect/capture_log.csv`。
- 按 `Enter` 保存当前 RGB 图像。
- 按 `q` 或 `Esc` 退出。

常用参数：

```bash
python scripts/data/collect_realsense_rgb.py --width 1280 --height 720 --fps 30
python scripts/data/collect_realsense_rgb.py --mode fixed
python scripts/data/collect_realsense_rgb.py --mode robot --robot-pose 100 200 300 0 0 1.57
```

自定义保存路径：

```bash
python scripts/data/collect_realsense_rgb.py \
  --output-dir datasets/glass_rect/raw_images \
  --log-file datasets/glass_rect/capture_log.csv \
  --prefix glass
```

Windows PowerShell 中可以把多行命令写成一行，或者用反引号换行。

## 数据标注

采集图片后运行：

```bash
python scripts/data/annotate_rect_holes.py
```

如果使用自定义数据目录：

```bash
python scripts/data/annotate_rect_holes.py \
  --images-dir datasets/glass_rect/raw_images \
  --labels-dir datasets/glass_rect/raw_labels \
  --point-radius 2
```

标注规则：

```text
沿空洞边界顺时针或逆时针依次点击多个点
按 f 或 Enter 结束当前空洞实例
继续点击下一个空洞实例
```

快捷键：

- `s`：保存当前图片标签。
- `f` 或 `Enter`：结束当前实例。
- `u`：撤销上一个点。
- `d`：删除当前未完成目标或最后一个完整目标。
- `n` 或右方向键：下一张。
- `p` 或左方向键：上一张。
- `q` 或 `Esc`：退出，若当前标签有修改会自动保存。

## 标签格式

每张图片对应一个同名 `.txt` 标签文件：

```text
glass_000001.jpg
glass_000001.txt
```

YOLO Seg 单行标签格式：

```text
class_id x1 y1 x2 y2 x3 y3 ... xN yN
```

说明：

- `class_id` 固定为 `0`。
- 坐标是归一化到 0-1 的多边形点。
- 每个空洞实例至少 3 个点。

## 整理训练集

采集和标注完成后运行：

```bash
python scripts/data/prepare_yolo_dataset.py --clean
```

输入目录：

```text
datasets/glass_rect/raw_images/
datasets/glass_rect/raw_labels/
```

输出目录：

```text
datasets/glass_rect/images/train/
datasets/glass_rect/images/val/
datasets/glass_rect/labels/train/
datasets/glass_rect/labels/val/
datasets/glass_rect/data.yaml
```

常用参数：

```bash
python scripts/data/prepare_yolo_dataset.py \
  --raw-images-dir datasets/glass_rect/raw_images \
  --raw-labels-dir datasets/glass_rect/raw_labels \
  --output-dir datasets/glass_rect \
  --val-ratio 0.2 \
  --seed 42 \
  --clean
```

## 检查数据集

整理完成后检查标签和图片：

```bash
python scripts/data/check_yolo_seg_dataset.py --check-images
```

如果出现以下错误，需要回到标注工具修正：

- `missing label file`
- `coordinates must be normalized to 0-1`
- `needs class_id and at least 3 x/y pairs`
- `orphan label without image`

## 注意事项

- 数据采集尽量覆盖不同位置、角度、光照和反光情况。
- 固定相机采集时尽量保持相机位置和姿态不变。
- 机械臂末端相机采集时建议同步记录每张图片对应的机械臂位姿。
- 同一个空洞实例的点必须沿边界连续点击，不要交叉跳点。
