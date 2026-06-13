# 玻璃空洞实例分割

本项目用于玻璃上下料场景中的视觉检测。目标是对玻璃圆盘中的每个空洞进行实例分割，而不是只检测矩形框或固定 4 个角点。空洞不一定是标准长方形，因此每个空洞按实际边界标注为一个多边形实例：

```text
点 1 -> 点 2 -> 点 3 -> ... -> 点 N -> 结束当前实例
```

模型方案采用 YOLO Seg。每个空洞作为 1 个分割实例，类别建议命名为 `glass_hole`，每个实例包含不固定数量的多边形轮廓点。

## 环境安装

建议使用 Python 3.10 或更新版本。

```bash
pip install -r requirements.txt
```

依赖包括：

- `opencv-python`：图像预览、采集显示和标注界面
- `numpy`：图像和点数据处理
- `pyrealsense2`：Intel RealSense D435i 相机采集
- `ultralytics`：后续 YOLO Seg 训练和推理

## 项目结构

```text
Object_Detection/
  scripts/
    collect_realsense_rgb.py
    annotate_rect_holes.py
  datasets/
    glass_rect/
      raw_images/
      raw_labels/
      capture_log.csv
  require.md
  requirements.txt
  README.md
```

其中：

- `raw_images/` 保存采集到的原始 RGB 图片。
- `raw_labels/` 保存人工标注得到的 YOLO Seg 标签。
- `capture_log.csv` 保存采集日志，包括图片名、时间、相机序列号、相机参数和可选机械臂位姿。

## 数据采集

连接 Intel RealSense D435i 后运行：

```bash
python scripts/collect_realsense_rgb.py
```

默认行为：

- 打开 D435i RGB 实时预览。
- 按 `Enter` 保存当前 RGB 图像。
- 按 `q` 或 `Esc` 退出。
- 图片默认保存到 `datasets/glass_rect/raw_images/`。
- 图片默认命名为 `glass_000001.jpg`、`glass_000002.jpg`。
- 采集日志默认保存到 `datasets/glass_rect/capture_log.csv`。

常用参数：

```bash
python scripts/collect_realsense_rgb.py --output-dir datasets/glass_rect_clean2/raw_images
python scripts/collect_realsense_rgb.py --width 1280 --height 720 --fps 30
```

多个参数可以写在同一条命令中。命令较长时，推荐用 `\` 换行，便于阅读：

```bash
python scripts/collect_realsense_rgb.py \
  --output-dir datasets/glass_rect/raw_images \
  --log-file datasets/glass_rect/capture_log.csv \
  --prefix glass \
  --width 1280 \
  --height 720 \
  --fps 30 \
  --mode fixed
```

参数说明：

- `--output-dir`：图片保存目录。
- `--log-file`：采集日志保存路径。
- `--prefix`：图片文件名前缀，例如 `glass` 会生成 `glass_000001.jpg`。
- `--width`：RGB 图像宽度。
- `--height`：RGB 图像高度。
- `--fps`：RGB 采集帧率。
- `--mode`：采集模式，可选 `fixed` 或 `robot`。
- `--robot-pose`：机械臂末端位姿，需要连续输入 6 个数：`X Y Z RX RY RZ`。

固定相机模式：

```bash
python scripts/collect_realsense_rgb.py --mode fixed
```

机械臂末端相机模式，并记录当前机械臂末端位姿：

```bash
python scripts/collect_realsense_rgb.py --mode robot --robot-pose 100 200 300 0 0 1.57
```

机械臂模式也可以同时指定图片路径、日志路径、图像尺寸和文件名前缀：

```bash
python scripts/collect_realsense_rgb.py \
  --output-dir datasets/glass_rect/raw_images \
  --log-file datasets/glass_rect/capture_log.csv \
  --prefix glass \
  --width 1280 \
  --height 720 \
  --fps 30 \
  --mode robot \
  --robot-pose 100 200 300 0 0 1.57
```

注意：如果要保存到项目目录下，路径前面不要加 `/`。例如应写 `datasets/glass_rect/raw_images`，不要写 `/datasets/glass_rect/raw_images`，否则会尝试写入 Linux 根目录并可能出现权限错误。

第一阶段建议先采集 200-500 张图像，覆盖玻璃在桌面上的不同位置、角度、光照和反光情况。

## 数据标注

采集图片后运行：

```bash
python scripts/annotate_rect_holes.py
```

多个参数可以写在同一条命令中：

```bash
python scripts/annotate_rect_holes.py \
  --images-dir datasets/TEST/glass_rect3/raw_images \
  --labels-dir datasets/TEST/glass_rect3/raw_labels \
  --point-radius 1
```

参数说明：

- `--images-dir`：要打开并标注的图片目录。
- `--labels-dir`：YOLO Seg `.txt` 标签保存目录。
- `--point-radius`：标注点在窗口中的显示半径，默认 `1`，数值越小越不遮挡边界。

如果采集图片时使用了自定义目录，标注时也要把 `--images-dir` 指向同一个图片目录。例如：

```bash
python scripts/annotate_rect_holes.py \
  --images-dir datasets/glass_rect/raw_images \
  --labels-dir datasets/glass_rect/raw_labels \
  --point-radius 2
```

标注窗口中移动鼠标时，会在鼠标位置显示红色十字准星。点击记录的是原图中的真实像素坐标，标签格式不变。

标注规则：

```text
沿空洞边界顺时针或逆时针依次点击多个点
按 f 或 Enter 结束当前空洞实例
继续点击下一个空洞实例
```

每个空洞实例至少需要 3 个点，点数不固定。边界越复杂，可以点击更多点；按 `f` 或 `Enter` 后，当前点序列才会保存为一个完整实例。

快捷键：

- `s`：保存当前图片标签
- `f` 或 `Enter`：结束当前实例，并开始下一个实例
- `u`：撤销上一个点
- `d`：删除当前未完成目标或最后一个完整目标
- `n` 或右方向键：下一张
- `p` 或左方向键：上一张
- `q` 或 `Esc`：退出，若当前标签有修改会自动保存

已有标签会在重新打开图片时自动加载，便于继续修改。

## 标签格式

标签采用 YOLO Seg 格式。每张图片对应一个同名 `.txt` 文件：

```text
glass_000001.jpg
glass_000001.txt
```

每个空洞实例对应一行：

```text
class_id x1 y1 x2 y2 x3 y3 ... xN yN
```

说明：

- `class_id` 固定为 `0`。
- `x1 y1 ... xN yN` 是实例分割多边形轮廓点坐标，归一化到 0-1。
- 每行至少包含 3 个点，也就是 `class_id` 后至少有 6 个坐标值。

## 整理训练集

训练前建议将数据整理为 YOLO Seg 目录：

```text
datasets/
  glass_rect/
    images/
      train/
      val/
    labels/
      train/
      val/
    data.yaml
```

`data.yaml` 示例：

```yaml
path: datasets/glass_rect
train: images/train
val: images/val

names:
  0: glass_hole
```

## 模型训练

整理好训练集后，可使用 Ultralytics YOLO Seg 训练：

```bash
yolo segment train data=datasets/glass_rect/data.yaml model=yolo26n-seg.pt epochs=100 imgsz=640
```

如果显卡性能较弱，优先使用较小的 Seg 模型；如果数据量和算力充足，可以尝试更大的 Seg 模型。

## 注意事项

- 同一个实例的多边形点应沿边界连续点击，建议统一使用顺时针或逆时针方向。
- 标注每个空洞后必须按 `f` 或 `Enter` 结束当前实例，否则该实例不会作为完整标签保存。
- 采集时尽量减少玻璃反光，保证空洞边界清晰可见。
- 固定相机采集时尽量保持相机位置和姿态不变。
- 机械臂末端相机采集时建议同步记录每张图片对应的机械臂位姿，便于后续坐标转换和误差分析。
- 后续如果需要将图像坐标转换到实际物理坐标，还需要进行相机标定、畸变校正和坐标系转换。

