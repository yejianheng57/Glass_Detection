import {
  BarChart,
  Card,
  CardBody,
  CardHeader,
  Callout,
  Grid,
  H1,
  H2,
  Pill,
  Row,
  Stack,
  Stat,
  Table,
  Text,
  useHostTheme,
} from "cursor/canvas";

const runs = [
  {
    label: "n 1280x736",
    run: "glass_hole_yolo26n_1280x736_e150",
    model: "YOLO26n-seg",
    imgsz: "736x1280",
    epochs: 150,
    batch: 24,
    timeH: 0.69,
    secPerEpoch: 16.5,
    finalBox: 0.97359,
    finalMask: 0.94471,
    bestBox: 0.97395,
    bestBoxEpoch: 135,
    bestMask: 0.94751,
    bestMaskEpoch: 134,
    bestMean: 0.95936,
    bestMeanEpoch: 133,
    zeroMetricEpochs: 0,
    nanValEpochs: 0,
    minValBox: 0.25808,
    minValSeg: 0.38898,
  },
  {
    label: "m 960",
    run: "glass_hole_yolo26m_960_e150",
    model: "YOLO26m-seg",
    imgsz: "960",
    epochs: 150,
    batch: 24,
    timeH: 1.8,
    secPerEpoch: 43.2,
    finalBox: 0.97139,
    finalMask: 0.93684,
    bestBox: 0.9741,
    bestBoxEpoch: 141,
    bestMask: 0.93779,
    bestMaskEpoch: 134,
    bestMean: 0.9549,
    bestMeanEpoch: 141,
    zeroMetricEpochs: 2,
    nanValEpochs: 2,
    minValBox: 0.26172,
    minValSeg: 0.38015,
  },
  {
    label: "n 640",
    run: "glass_hole_yolo26n",
    model: "YOLO26n-seg",
    imgsz: "640",
    epochs: 100,
    batch: 24,
    timeH: 0.42,
    secPerEpoch: 15.3,
    finalBox: 0.96567,
    finalMask: 0.908,
    bestBox: 0.9657,
    bestBoxEpoch: 98,
    bestMask: 0.91903,
    bestMaskEpoch: 30,
    bestMean: 0.93854,
    bestMeanEpoch: 90,
    zeroMetricEpochs: 0,
    nanValEpochs: 0,
    minValBox: 0.26975,
    minValSeg: 0.35887,
  },
  {
    label: "s 640",
    run: "glass_hole_yolo26s_640_e200",
    model: "YOLO26s-seg",
    imgsz: "640",
    epochs: 200,
    batch: 24,
    timeH: 1.78,
    secPerEpoch: 32,
    finalBox: 0.96748,
    finalMask: 0.91424,
    bestBox: 0.96967,
    bestBoxEpoch: 184,
    bestMask: 0.9199,
    bestMaskEpoch: 134,
    bestMean: 0.94308,
    bestMeanEpoch: 178,
    zeroMetricEpochs: 1,
    nanValEpochs: 24,
    minValBox: 0.26681,
    minValSeg: 0.36289,
  },
  {
    label: "s 960",
    run: "glass_hole_yolo26s_960_e150",
    model: "YOLO26s-seg",
    imgsz: "960",
    epochs: 150,
    batch: 12,
    timeH: 1.92,
    secPerEpoch: 46.2,
    finalBox: 0.97034,
    finalMask: 0.93719,
    bestBox: 0.97263,
    bestBoxEpoch: 135,
    bestMask: 0.93942,
    bestMaskEpoch: 144,
    bestMean: 0.95538,
    bestMeanEpoch: 139,
    zeroMetricEpochs: 0,
    nanValEpochs: 0,
    minValBox: 0.26059,
    minValSeg: 0.37753,
  },
];

const fmtPct = (value: number) => `${(value * 100).toFixed(2)}%`;
const fmtNum = (value: number) => value.toFixed(3);

export default function GlassYoloResultsComparison() {
  const theme = useHostTheme();
  const categories = runs.map((run) => run.label);
  const bestRun = runs[0];
  const bestBoxRun = runs[1];
  const runnerUp = runs[4];
  const timeSaved = 1 - bestRun.timeH / runnerUp.timeH;

  return (
    <Stack gap={18} style={{ padding: 20, color: theme.text.primary }}>
      <Stack gap={6}>
        <H1>玻璃孔洞分割训练结果对比</H1>
        <Text tone="secondary">
          Source: `results.csv` 和 `args.yaml` · 比较 5 次 YOLO26 segmentation 训练 ·
          主指标为 `metrics/mAP50-95(M)`，检测框指标作为辅助参考。
        </Text>
      </Stack>

      <Callout tone="success" title="推荐选择">
        <Text>
          `glass_hole_yolo26n_1280x736_e150` 综合最好：最高 Mask mAP50-95 为{" "}
          <Text as="span" weight="semibold">
            {fmtPct(bestRun.bestMask)}
          </Text>
          ，最终 Mask mAP50-95 为{" "}
          <Text as="span" weight="semibold">
            {fmtPct(bestRun.finalMask)}
          </Text>
          ，且训练耗时只需 {bestRun.timeH.toFixed(2)} 小时，比次优 `s 960` 少约{" "}
          {(timeSaved * 100).toFixed(0)}%。
        </Text>
      </Callout>

      <Grid columns={4} gap={12}>
        <Stat value={fmtPct(bestRun.bestMask)} label="最高 Mask mAP50-95" tone="success" />
        <Stat value={fmtPct(bestBoxRun.bestBox)} label="最高 Box mAP50-95 (m 960)" tone="info" />
        <Stat value={`${bestRun.timeH.toFixed(2)}h`} label="推荐 run 训练耗时" tone="info" />
        <Stat value="0" label="推荐 run 异常验证 epoch" />
      </Grid>

      <Grid columns="minmax(0, 1.2fr) minmax(0, 1fr)" gap={16}>
        <Card>
          <CardHeader trailing={<Pill active size="sm">主指标</Pill>}>
            Mask mAP50-95 对比
          </CardHeader>
          <CardBody>
            <BarChart
              categories={categories}
              series={[
                { name: "Final Mask mAP50-95", data: runs.map((run) => run.finalMask) },
                { name: "Best Mask mAP50-95", data: runs.map((run) => run.bestMask) },
              ]}
              beginAtZero={false}
              yMin={0.9}
              yMax={0.955}
              valueSuffix=""
              height={260}
            />
            <Text size="small" tone="tertiary">
              横轴：训练 run；纵轴：Mask mAP50-95。Source: `results.csv`，取最终 epoch 与全程峰值。
            </Text>
          </CardBody>
        </Card>

        <Card>
          <CardHeader>训练耗时</CardHeader>
          <CardBody>
            <BarChart
              categories={categories}
              series={[{ name: "训练耗时", data: runs.map((run) => run.timeH), tone: "info" }]}
              height={260}
              valueSuffix=" h"
              showValues
            />
            <Text size="small" tone="tertiary">
              横轴：训练 run；纵轴：总训练耗时（小时）。Source: `results.csv` 的最终 `time`。
            </Text>
          </CardBody>
        </Card>
      </Grid>

      <H2>完整指标表</H2>
      <Table
        headers={[
          "Run",
          "模型 / 输入",
          "最终 B",
          "最终 M",
          "最佳 B",
          "最佳 M",
          "耗时",
          "异常",
        ]}
        rows={runs.map((run) => [
          run.label,
          `${run.model} / ${run.imgsz} / e${run.epochs} / b${run.batch}`,
          fmtPct(run.finalBox),
          fmtPct(run.finalMask),
          `${fmtPct(run.bestBox)} @${run.bestBoxEpoch}`,
          `${fmtPct(run.bestMask)} @${run.bestMaskEpoch}`,
          `${run.timeH.toFixed(2)}h (${run.secPerEpoch.toFixed(1)}s/ep)`,
          `${run.zeroMetricEpochs} zero, ${run.nanValEpochs} nan-val`,
        ])}
        columnAlign={["left", "left", "right", "right", "right", "right", "right", "right"]}
        rowTone={["success", "neutral", "neutral", "warning", "info"]}
        striped
      />

      <Grid columns={3} gap={14}>
        <Stack gap={6}>
          <H2>精度结论</H2>
          <Text>
            `n 1280x736` 的 Mask mAP50-95 第一；Box 峰值与第一的 `m 960`
            基本持平，只低 {fmtPct(bestBoxRun.bestBox - bestRun.bestBox)}。Mask 峰值比 `s 960`
            高 {fmtPct(bestRun.bestMask - runnerUp.bestMask)}，最终 Mask 高{" "}
            {fmtPct(bestRun.finalMask - runnerUp.finalMask)}。
          </Text>
        </Stack>
        <Stack gap={6}>
          <H2>尺寸收益</H2>
          <Text>
            640 输入明显吃亏：`n 640` 和 `s 640` 的最终 Mask mAP50-95 只有{" "}
            {fmtPct(runs[2].finalMask)} / {fmtPct(runs[3].finalMask)}，比 960/1280
            档低约 2.3 到 3.7 个百分点。
          </Text>
        </Stack>
        <Stack gap={6}>
          <H2>稳定性</H2>
          <Text>
            `s 640 e200` 有 24 个 epoch 的验证 loss 出现 `nan`，且多训到 200 epoch
            也没有追上 `s 960` 或 `n 1280x736`，不建议作为主模型。
          </Text>
        </Stack>
      </Grid>

      <Card>
        <CardHeader>结论排序</CardHeader>
        <CardBody>
          <Table
            headers={["推荐级别", "Run", "原因"]}
            rows={[
              ["1", "n 1280x736 e150", "精度最高、耗时最低之一、无异常记录，综合最佳"],
              ["2", "s 960 e150", "分割精度第二，稳定，但耗时约为推荐方案 2.8 倍"],
              ["3", "m 960 e150", "Box 指标略强，但 Mask 指标低于 s 960，且有早期异常 epoch"],
              ["4", "s 640 e200", "训练长、异常多，分割精度仍低"],
              ["5", "n 640 e100", "最快但分割精度明显落后，只适合极端轻量/快速验证"],
            ]}
            columnAlign={["right", "left", "left"]}
            rowTone={["success", "info", "neutral", "warning", "neutral"]}
            framed={false}
          />
        </CardBody>
      </Card>
    </Stack>
  );
}
