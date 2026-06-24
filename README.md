<div align="center">

# APRNet

**Anatomy-aware Pretraining and Pseudo-label Rectification for Semi-Supervised Maxillary Sinus Segmentation**

Semi-Supervised Maxillary Sinus Segmentation

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white)

<!-- <img src="assets/appr-overview.svg" alt="APPR overview" width="100%" /> -->
<p align="center">
  <img src="figures/framework.png" alt="APRNet Overall Framework" width="900">
</p>
<p align="center"><em>Figure 1: The overall three-stage pipeline of the proposed APRNet framework.</em></p>
</div>

## Overview

APRNet is a three-stage semi-supervised framework designed for anatomically constrained bilateral maxillary sinus segmentation.

- **Stage I – Anatomy-aware pretraining** initializes the segmentation network with robust aerated-region priors extracted from unlabeled scans.
- **Stage II – Semantic rectification** improves labeled supervision with side-aware constraints that reduce left-right semantic confusion.
- **Stage III – Pseudo-label co-refinement** progressively fuses anatomical priors and model predictions to obtain cleaner unlabeled supervision.

This repository contains the core training, evaluation, prior modeling, loss design, and 3D backbone implementation used by APRNet.

## Abstract

Semi-supervised medical image segmentation aims to leverage large volumes of unlabeled data with extremely limited annotation budgets. For anatomically complex bilateral structures, random initialization, foreground semantic misalignment, and noisy pseudo-label accumulation can significantly undermine training stability. APRNet addresses this problem with a unified three-stage framework that combines anatomy-aware pretraining, semantic rectification, and progressive pseudo-label co-refinement for maxillary sinus segmentation.

## Method

APRNet follows a progressive training pipeline:

1. **Anatomy-aware pretraining** extracts stable aerated-region priors from unlabeled scans to warm-start the model.
2. **Semantic rectification** imposes side-aware supervision to reduce left-right confusion and preserve bilateral exclusivity.
3. **Pseudo-label co-refinement** jointly uses prior guidance and segmentation responses to iteratively improve unlabeled supervision.

In practice, the framework is designed to improve three things at once: foreground localization, bilateral semantic consistency, and boundary quality.

<p align="center">
  <img src="figures/paradigm.png" alt="APRNet Module Details" width="850">
</p>
<p align="center"><em>Figure 2: Detailed illustration of the core modules in APRNet.</em></p>

## Highlights

- Unified `train.py` entry for `unsup`, `sup`, and `corefined` stages.
- Explicit bilateral semantic handling for left/right maxillary sinus consistency.
- Anatomy-aware prior extraction implemented in `PriorNet.py`.
- MONAI-based 3D training and sliding-window inference pipeline.
- Lightweight evaluation entry in `eval.py` for checkpoint-based testing.

## Repository Layout

```text
APRNet/
├── PriorNet.py      # anatomy-aware prior extraction and pretraining modules
├── VISTA3D.py       # 3D segmentation backbone builder
├── dataset.py       # data loading, transforms, split helpers, orientation checks
├── loss.py          # side-aware loss definitions
├── train.py         # unified three-stage training entry
├── eval.py          # checkpoint evaluation script
├── requirements.txt # minimal runtime dependencies
└── README.md
```

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

If you are using CUDA, install the PyTorch build that matches your local driver before installing the remaining packages.

## Data Preparation

The code expects a dataset root and a split file:

```text
dataset_root/
├── images/...
├── labels/...
└── splits.json
```

Key notes:

- `dataset.py` contains orientation consistency checks to keep bilateral semantics aligned.
- `--split-json` defaults to `<dataset-root>/splits.json`.
- `--split-ratio` supports settings such as `5`, `10`, `20`, `50`, and `100`.

## Datasets

APRNet is organized around two public craniofacial datasets described in the manuscript:

- **NasalSeg**: 130 craniofacial 3D CT volumes with bilateral maxillary sinus annotations.
- **ToothFairy2**: a multicenter CBCT benchmark, from which 45 maxillary-sinus-labeled cases are used in this project.

The experimental protocol uses an `8:2` train/test split and semi-supervised settings with `5%`, `10%`, `20%`, and `50%` labeled data.

## Training

### Stage I: anatomy-aware pretraining

```bash
python train.py \
  --stage unsup \
  --dataset-root /path/to/dataset_root \
  --split-ratio 20 \
  --gpu 0 \
  --amp
```

### Stage II: supervised semantic rectification

```bash
python train.py \
  --stage sup \
  --dataset-root /path/to/dataset_root \
  --split-ratio 20 \
  --init-checkpoint /path/to/stage1_checkpoint.pt \
  --gpu 0 \
  --amp
```

### Stage III: collaborative pseudo-label refinement

```bash
python train.py \
  --stage corefined \
  --dataset-root /path/to/dataset_root \
  --split-ratio 20 \
  --init-checkpoint /path/to/stage2_checkpoint.pt \
  --gpu 0 \
  --amp
```

## Evaluation

```bash
python eval.py \
  --checkpoint /path/to/checkpoint.pt \
  --dataset-root /path/to/dataset_root \
  --split-ratio 20 \
  --eval-key test \
  --gpu 0 \
  --amp
```

## Results

### Quantitative Comparison with SOTA Methods
<!-- <p align="center">
  <img src="figures/combined.png" alt="SOTA Quantitative Results" width="850">
</p>
<p align="center"><em>Table 1: Quantitative comparison between APRNet and state-of-the-art methods.</em></p> -->
<p><strong>Comparison of maxillary sinus segmentation results on NasalSeg and ToothFairy2 under different labeled-data ratios.</p>

<table border="1" cellpadding="4" cellspacing="0" style="width:100%; font-size:12px; border-collapse:collapse; text-align:center;">
  <thead>
    <tr>
      <th>Dataset</th>
      <th>Ratio</th>
      <th>Model</th>
      <th>Dice↑<br/>(Left / Right Sinus)</th>
      <th>PPV↑<br/>(Left / Right Sinus)</th>
      <th>HD95↓<br/>(Left / Right Sinus)</th>
      <th>ASSD↓<br/>(Left / Right Sinus)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="32" style="writing-mode:vertical-rl; transform:rotate(180deg); font-weight:bold;">NasalSeg</td>
      <td rowspan="8" style="font-weight:bold;">5%</td>
      <td>CGS (TMI 2025)</td>
      <td>0.5313±0.1957 / 0.4751±0.1552</td>
      <td>0.8549±0.2181 / 0.9328±0.1496</td>
      <td>8.91±5.26 / 17.50±9.11</td>
      <td>2.06±2.33 / 2.47±0.65</td>
    </tr>
    <tr>
      <td>DECSeg (PR 2025)</td>
      <td>0.5602±0.2996 / 0.4390±0.2241</td>
      <td>0.7660±0.3349 / 0.7984±0.2988</td>
      <td>20.91±30.41 / 20.25±27.61</td>
      <td>13.07±25.74 / 11.19±23.11</td>
    </tr>
    <tr>
      <td>STAC (BIBM 2024)</td>
      <td>0.7459±0.1141 / 0.6177±0.1653</td>
      <td>0.6288±0.1436 / 0.4760±0.1722</td>
      <td>38.12±16.52 / 82.56±24.36</td>
      <td>9.12±5.15 / 24.12±11.01</td>
    </tr>
    <tr>
      <td>MetaSSL (TMI 2025)</td>
      <td>0.5835±0.0845 / 0.5469±0.1119</td>
      <td>0.4295±0.0855 / 0.3962±0.1016</td>
      <td>83.28±6.84 / 82.59±7.23</td>
      <td>26.18±4.90 / 26.63±5.51</td>
    </tr>
    <tr>
      <td>DiffRect (MICCAI 2024)</td>
      <td>0.7469±0.1946 / 0.7376±0.1649</td>
      <td>0.9306±0.1668 / <span style="color:red">0.9352±0.1460</span></td>
      <td>10.35±11.53 / 12.05±14.41</td>
      <td>3.10±4.56 / 3.16±4.96</td>
    </tr>
    <tr>
      <td>Text-SemiSeg (MICCAI 2025)</td>
      <td>0.5442±0.3280 / 0.5590±0.3075</td>
      <td>0.4675±0.3413 / 0.4713±0.3188</td>
      <td>33.80±22.56 / 36.27±25.05</td>
      <td>15.30±11.74 / 16.40±12.29</td>
    </tr>
    <tr>
      <td>SynFoC (CVPR 2025)</td>
      <td><span style="color:orange">0.8898±0.1670</span> / <span style="color:orange">0.8658±0.1876</span></td>
      <td><span style="color:red">0.9601±0.0681</span> / 0.9046±0.2131</td>
      <td><span style="color:orange">3.95±6.71</span> / <span style="color:orange">6.33±13.08</span></td>
      <td><span style="color:red">0.99±0.99</span> / <span style="color:orange">2.40±4.80</span></td>
    </tr>
    <tr style="background-color:#e0e0e0; font-weight:bold;">
      <td>APPR (Ours)</td>
      <td><span style="color:red">0.9126±0.0814</span> / <span style="color:red">0.8991±0.0983</span></td>
      <td><span style="color:orange">0.9447±0.1166</span> / <span style="color:orange">0.9335±0.1444</span></td>
      <td><span style="color:red">3.35±14.26</span> / <span style="color:red">2.89±14.75</span></td>
      <td><span style="color:orange">1.64±1.80</span> / <span style="color:red">1.65±1.93</span></td>
    </tr>
    <tr>
      <td rowspan="8" style="font-weight:bold;">10%</td>
      <td>CGS (TMI 2025)</td>
      <td>0.8861±0.1894 / <span style="color:orange">0.9265±0.1135</span></td>
      <td>0.7661±0.2820 / 0.7440±0.2398</td>
      <td>4.42±9.32 / <span style="color:orange">2.18±2.92</span></td>
      <td>3.09±10.72 / <span style="color:orange">0.73±0.64</span></td>
    </tr>
    <tr>
      <td>DECSeg (PR 2025)</td>
      <td>0.6547±0.3060 / 0.6560±0.3282</td>
      <td>0.7912±0.3450 / 0.7945±0.3474</td>
      <td>17.67±29.55 / 18.94±31.27</td>
      <td>11.78±24.34 / 12.47±25.79</td>
    </tr>
    <tr>
      <td>STAC (BIBM 2024)</td>
      <td>0.7614±0.1049 / 0.6712±0.1834</td>
      <td>0.6602±0.1499 / 0.5395±0.1959</td>
      <td>31.58±15.53 / 64.34±39.78</td>
      <td>7.32±3.92 / 18.95±15.70</td>
    </tr>
    <tr>
      <td>MetaSSL (TMI 2025)</td>
      <td>0.6168±0.0727 / 0.5780±0.1029</td>
      <td>0.4569±0.0757 / 0.4207±0.0950</td>
      <td>83.53±7.49 / 82.17±8.90</td>
      <td>24.67±5.51 / 25.25±5.78</td>
    </tr>
    <tr>
      <td>DiffRect (MICCAI 2024)</td>
      <td>0.8605±0.1828 / 0.8508±0.1925</td>
      <td>0.9026±0.1839 / 0.8974±0.1827</td>
      <td>14.15±13.78 / 16.40±15.68</td>
      <td>2.39±2.51 / 3.22±5.29</td>
    </tr>
    <tr>
      <td>Text-SemiSeg (MICCAI 2025)</td>
      <td><span style="color:orange">0.8958±0.0990</span> / 0.9019±0.0661</td>
      <td>0.8306±0.1148 / 0.8385±0.0567</td>
      <td>8.93±17.01 / 8.30±16.73</td>
      <td>3.90±6.66 / 2.84±3.80</td>
    </tr>
    <tr>
      <td>SynFoC (CVPR 2025)</td>
      <td>0.8957±0.1830 / 0.8977±0.1222</td>
      <td><span style="color:orange">0.9422±0.0848</span> / <span style="color:orange">0.9150±0.1412</span></td>
      <td><span style="color:orange">3.18±5.06</span> / 5.95±13.32</td>
      <td><span style="color:orange">0.91±1.04</span> / 2.05±4.46</td>
    </tr>
    <tr style="background-color:#e0e0e0; font-weight:bold;">
      <td>APPR (Ours)</td>
      <td><span style="color:red">0.9388±0.0351</span> / <span style="color:red">0.9332±0.0782</span></td>
      <td><span style="color:red">0.9453±0.0421</span> / <span style="color:red">0.9285±0.1147</span></td>
      <td><span style="color:red">2.72±6.95</span> / <span style="color:red">1.78±3.07</span></td>
      <td><span style="color:red">0.40±0.68</span> / <span style="color:red">0.32±0.53</span></td>
    </tr>
    <tr>
      <td rowspan="8" style="font-weight:bold;">20%</td>
      <td>CGS (TMI 2025)</td>
      <td>0.8182±0.1835 / 0.8257±0.1097</td>
      <td><span style="color:red">0.9712±0.1471</span> / 0.8920±0.1973</td>
      <td>5.39±11.20 / 5.74±4.34</td>
      <td>2.27±6.95 / <span style="color:orange">1.52±1.45</span></td>
    </tr>
    <tr>
      <td>DECSeg (PR 2025)</td>
      <td>0.8181±0.2435 / 0.8012±0.2605</td>
      <td>0.8270±0.2487 / 0.8464±0.2541</td>
      <td>10.89±23.77 / 11.00±23.34</td>
      <td>6.58±19.35 / 6.47±18.74</td>
    </tr>
    <tr>
      <td>STAC (BIBM 2024)</td>
      <td>0.7670±0.1172 / 0.6860±0.2145</td>
      <td>0.6448±0.1438 / 0.5669±0.2189</td>
      <td>30.87±14.07 / 71.59±31.92</td>
      <td>7.09±4.55 / 20.51±12.34</td>
    </tr>
    <tr>
      <td>MetaSSL (TMI 2025)</td>
      <td>0.6416±0.0549 / 0.6010±0.0889</td>
      <td>0.4814±0.0598 / 0.4418±0.0845</td>
      <td>82.06±12.62 / 80.47±14.05</td>
      <td>23.25±6.21 / 24.13±6.27</td>
    </tr>
    <tr>
      <td>DiffRect (MICCAI 2024)</td>
      <td>0.8952±0.1826 / 0.8980±0.1817</td>
      <td>0.9077±0.1876 / <span style="color:orange">0.9133±0.1848</span></td>
      <td>11.40±17.47 / 10.59±16.87</td>
      <td>3.46±9.99 / 3.44±10.19</td>
    </tr>
    <tr>
      <td>Text-SemiSeg (MICCAI 2025)</td>
      <td><span style="color:orange">0.9119±0.0751</span> / <span style="color:orange">0.9102±0.0623</span></td>
      <td>0.8498±0.0940 / 0.8486±0.0597</td>
      <td>4.24±13.69 / <span style="color:orange">4.32±11.90</span></td>
      <td>2.64±4.82 / 2.18±3.42</td>
    </tr>
    <tr>
      <td>SynFoC (CVPR 2025)</td>
      <td>0.9091±0.1478 / 0.8838±0.1714</td>
      <td><span style="color:orange">0.9509±0.0701</span> / 0.9065±0.1845</td>
      <td><span style="color:red">2.73±4.06</span> / 6.25±13.73</td>
      <td><span style="color:orange">0.87±1.08</span> / 2.20±4.52</td>
    </tr>
    <tr style="background-color:#e0e0e0; font-weight:bold;">
      <td>APPR (Ours)</td>
      <td><span style="color:red">0.9455±0.0276</span> / <span style="color:red">0.9301±0.0895</span></td>
      <td>0.9431±0.0502 / <span style="color:red">0.9202±0.1255</span></td>
      <td><span style="color:orange">3.78±7.89</span> / <span style="color:red">2.53±5.46</span></td>
      <td><span style="color:red">0.62±0.78</span> / <span style="color:red">0.50±0.93</span></td>
    </tr>
    <tr>
      <td rowspan="8" style="font-weight:bold;">50%</td>
      <td>CGS (TMI 2025)</td>
      <td>0.8293±0.1848 / 0.8020±0.1763</td>
      <td>0.9519±0.2002 / 0.9071±0.2289</td>
      <td>5.89±14.55 / 6.86±8.99</td>
      <td>1.64±4.00 / 3.23±9.55</td>
    </tr>
    <tr>
      <td>DECSeg (PR 2025)</td>
      <td>0.7574±0.3212 / 0.7535±0.3200</td>
      <td>0.8123±0.3011 / 0.8318±0.3076</td>
      <td>13.78±28.38 / 13.91±27.93</td>
      <td>8.98±22.04 / 8.98±21.64</td>
    </tr>
    <tr>
      <td>STAC (BIBM 2024)</td>
      <td>0.7988±0.1056 / 0.6778±0.1903</td>
      <td>0.6885±0.1394 / 0.5518±0.2074</td>
      <td>26.65±15.05 / 61.10±36.10</td>
      <td>6.01±4.13 / 19.44±15.59</td>
    </tr>
    <tr>
      <td>MetaSSL (TMI 2025)</td>
      <td>0.6509±0.0528 / 0.6100±0.0842</td>
      <td>0.4917±0.0582 / 0.4505±0.0816</td>
      <td>80.78±14.77 / 80.18±14.04</td>
      <td>22.45±5.64 / 23.88±6.52</td>
    </tr>
    <tr>
      <td>DiffRect (MICCAI 2024)</td>
      <td><span style="color:orange">0.9087±0.1335</span> / <span style="color:orange">0.9200±0.1110</span></td>
      <td>0.9286±0.1528 / 0.9455±0.1014</td>
      <td>7.34±12.88 / 6.64±11.81</td>
      <td>2.11±3.49 / <span style="color:orange">1.62±2.58</span></td>
    </tr>
    <tr>
      <td>Text-SemiSeg (MICCAI 2025)</td>
      <td>0.9076±0.1088 / 0.9153±0.0469</td>
      <td>0.8470±0.1230 / 0.8528±0.0539</td>
      <td><span style="color:orange">4.04±12.92</span> / <span style="color:orange">4.58±15.17</span></td>
      <td>1.96±5.23 / 1.79±3.44</td>
    </tr>
    <tr>
      <td>SynFoC (CVPR 2025)</td>
      <td>0.8618±0.1753 / 0.8507±0.1598</td>
      <td><span style="color:red">0.9707±0.0623</span> / <span style="color:orange">0.9505±0.1158</span></td>
      <td>4.46±6.47 / 6.53±12.45</td>
      <td><span style="color:orange">1.22±0.97</span> / 2.18±4.25</td>
    </tr>
    <tr style="background-color:#e0e0e0; font-weight:bold;">
      <td>APPR (Ours)</td>
      <td><span style="color:red">0.9594±0.0175</span> / <span style="color:red">0.9601±0.0214</span></td>
      <td><span style="color:orange">0.9641±0.0198</span> / <span style="color:red">0.9594±0.0328</span></td>
      <td><span style="color:red">0.77±0.39</span> / <span style="color:red">0.96±1.33</span></td>
      <td><span style="color:red">0.18±0.11</span> / <span style="color:red">0.24±0.47</span></td>
    </tr>
    <tr>
      <td rowspan="32" style="writing-mode:vertical-rl; transform:rotate(180deg); font-weight:bold;">ToothFairy2</td>
      <td rowspan="8" style="font-weight:bold;">5%</td>
      <td>CGS (TMI 2025)</td>
      <td>0.2828±0.1823 / 0.2401±0.2120</td>
      <td>0.5000±0.2965 / 0.4435±0.2932</td>
      <td>53.55±20.27 / 54.31±33.92</td>
      <td>15.95±11.07 / 17.74±14.22</td>
    </tr>
    <tr>
      <td>DECSeg (PR 2025)</td>
      <td>0.4649±0.2460 / 0.5054±0.1941</td>
      <td>0.7103±0.2242 / 0.6069±0.1861</td>
      <td>40.74±24.41 / 26.76±22.52</td>
      <td>7.51±5.58 / 5.88±4.70</td>
    </tr>
    <tr>
      <td>STAC (BIBM 2024)</td>
      <td>0.1554±0.0626 / 0.1130±0.0532</td>
      <td>0.0970±0.0439 / 0.0660±0.0327</td>
      <td>77.78±22.22 / 62.06±38.75</td>
      <td>29.70±12.14 / 24.75±19.73</td>
    </tr>
    <tr>
      <td>MetaSSL (TMI 2025)</td>
      <td>0.5062±0.1516 / 0.4474±0.1390</td>
      <td>0.3851±0.1477 / 0.3194±0.1209</td>
      <td>48.42±11.77 / 45.37±7.30</td>
      <td>14.27±2.77 / 16.00±3.86</td>
    </tr>
    <tr>
      <td>DiffRect (MICCAI 2024)</td>
      <td>0.6482±0.1135 / 0.5931±0.2408</td>
      <td>0.6911±0.1711 / 0.5730±0.2647</td>
      <td>112.30±68.19 / 111.04±68.70</td>
      <td>23.78±17.18 / 29.74±25.45</td>
    </tr>
    <tr>
      <td>Text-SemiSeg (MICCAI 2025)</td>
      <td>0.5522±0.1585 / 0.4726±0.1892</td>
      <td>0.4751±0.1947 / 0.3561±0.1929</td>
      <td>211.86±40.95 / 197.27±34.28</td>
      <td>66.83±15.10 / 79.63±20.59</td>
    </tr>
    <tr>
      <td>SynFoC (CVPR 2025)</td>
      <td><span style="color:orange">0.6856±0.3608</span> / <span style="color:orange">0.7611±0.3394</span></td>
      <td><span style="color:orange">0.7784±0.3536</span> / <span style="color:orange">0.9223±0.0536</span></td>
      <td><span style="color:orange">16.75±28.01</span> / <span style="color:orange">6.64±7.95</span></td>
      <td><span style="color:orange">3.01±4.47</span> / <span style="color:red">0.96±0.57</span></td>
    </tr>
    <tr style="background-color:#e0e0e0; font-weight:bold;">
      <td>APPR (Ours)</td>
      <td><span style="color:red">0.9133±0.1155</span> / <span style="color:red">0.9158±0.1387</span></td>
      <td><span style="color:red">0.9796±0.0898</span> / <span style="color:red">0.9573±0.1123</span></td>
      <td><span style="color:red">2.88±0.93</span> / <span style="color:red">4.89±2.44</span></td>
      <td><span style="color:red">1.23±1.01</span> / <span style="color:orange">2.58±1.88</span></td>
    </tr>
    <tr>
      <td rowspan="8" style="font-weight:bold;">10%</td>
      <td>CGS (TMI 2025)</td>
      <td>0.7571±0.2617 / 0.7726±0.2383</td>
      <td>0.9360±0.0724 / 0.9006±0.0949</td>
      <td>3.47±4.10 / 16.65±23.11</td>
      <td>1.18±1.69 / 3.42±3.95</td>
    </tr>
    <tr>
      <td>DECSeg (PR 2025)</td>
      <td><span style="color:orange">0.9043±0.0563</span> / <span style="color:orange">0.8820±0.0672</span></td>
      <td><span style="color:orange">0.9366±0.0292</span> / <span style="color:red">0.9638±0.0238</span></td>
      <td><span style="color:orange">2.31±2.38</span> / <span style="color:orange">1.95±1.58</span></td>
      <td><span style="color:orange">0.86±0.93</span> / <span style="color:red">0.39±0.23</span></td>
    </tr>
    <tr>
      <td>STAC (BIBM 2024)</td>
      <td>0.2197±0.0934 / 0.0861±0.0795</td>
      <td>0.1267±0.0600 / 0.0470±0.0476</td>
      <td>63.26±16.18 / 45.25±16.20</td>
      <td>31.98±12.37 / 22.29±8.15</td>
    </tr>
    <tr>
      <td>MetaSSL (TMI 2025)</td>
      <td>0.5481±0.0825 / 0.5319±0.1413</td>
      <td>0.4341±0.0805 / 0.4039±0.1171</td>
      <td>40.89±3.44 / 41.81±3.63</td>
      <td>12.41±2.49 / 14.17±4.29</td>
    </tr>
    <tr>
      <td>DiffRect (MICCAI 2024)</td>
      <td>0.8453±0.0826 / 0.8047±0.1713</td>
      <td>0.8625±0.1056 / 0.9128±0.0800</td>
      <td>57.27±70.09 / 63.08±76.72</td>
      <td>13.11±14.02 / 10.02±13.48</td>
    </tr>
    <tr>
      <td>Text-SemiSeg (MICCAI 2025)</td>
      <td>0.7976±0.0988 / 0.7115±0.1598</td>
      <td>0.6985±0.1434 / 0.5889±0.1824</td>
      <td>206.87±27.15 / 244.58±32.16</td>
      <td>55.43±9.72 / 85.73±20.72</td>
    </tr>
    <tr>
      <td>SynFoC (CVPR 2025)</td>
      <td>0.7230±0.3360 / 0.7828±0.2911</td>
      <td>0.9355±0.0770 / 0.9216±0.0478</td>
      <td>12.10±18.15 / 5.15±5.70</td>
      <td>0.87±0.80 / 0.91±0.52</td>
    </tr>
    <tr style="background-color:#e0e0e0; font-weight:bold;">
      <td>APPR (Ours)</td>
      <td><span style="color:red">0.9238±0.0989</span> / <span style="color:red">0.9023±0.1354</span></td>
      <td><span style="color:red">0.9411±0.0989</span> / <span style="color:orange">0.9543±0.1192</span></td>
      <td><span style="color:red">2.01±0.83</span> / <span style="color:red">1.82±1.09</span></td>
      <td><span style="color:red">0.81±0.56</span> / <span style="color:orange">0.51±0.33</span></td>
    </tr>
    <tr>
      <td rowspan="8" style="font-weight:bold;">20%</td>
      <td>CGS (TMI 2025)</td>
      <td>0.9006±0.0776 / 0.8718±0.0880</td>
      <td>0.9487±0.0351 / 0.9013±0.0548</td>
      <td>13.59±18.82 / 27.60±34.45</td>
      <td>1.67±2.06 / 5.19±5.15</td>
    </tr>
    <tr>
      <td>DECSeg (PR 2025)</td>
      <td><span style="color:orange">0.9271±0.0382</span> / <span style="color:orange">0.9350±0.0319</span></td>
      <td>0.9633±0.0164 / 0.9622±0.0191</td>
      <td><span style="color:orange">1.45±1.68</span> / <span style="color:red">0.82±0.49</span></td>
      <td><span style="color:orange">0.28±0.20</span> / <span style="color:orange">0.19±0.08</span></td>
    </tr>
    <tr>
      <td>STAC (BIBM 2024)</td>
      <td>0.7798±0.1079 / 0.6424±0.1881</td>
      <td>0.6725±0.1353 / 0.5891±0.1896</td>
      <td>19.83±24.34 / 16.20±27.08</td>
      <td>4.42±5.29 / 6.20±13.75</td>
    </tr>
    <tr>
      <td>MetaSSL (TMI 2025)</td>
      <td>0.6474±0.0826 / 0.5862±0.0817</td>
      <td>0.5093±0.1021 / 0.4344±0.0755</td>
      <td>42.84±6.97 / 46.98±14.12</td>
      <td>11.55±2.21 / 14.01±3.17</td>
    </tr>
    <tr>
      <td>DiffRect (MICCAI 2024)</td>
      <td>0.9025±0.0435 / 0.8774±0.1136</td>
      <td><span style="color:red">0.9711±0.0154</span> / <span style="color:red">0.9730±0.0207</span></td>
      <td>7.07±6.50 / 3.85±3.01</td>
      <td>1.23±0.76 / 1.07±0.73</td>
    </tr>
    <tr>
      <td>Text-SemiSeg (MICCAI 2025)</td>
      <td>0.7453±0.1196 / 0.7661±0.1137</td>
      <td>0.6543±0.1707 / 0.6615±0.1329</td>
      <td>167.89±71.82 / 185.51±52.25</td>
      <td>42.99±18.13 / 44.33±13.67</td>
    </tr>
    <tr>
      <td>SynFoC (CVPR 2025)</td>
      <td>0.9146±0.0678 / 0.9119±0.0518</td>
      <td>0.9149±0.0538 / 0.9256±0.0530</td>
      <td>3.01±2.15 / 2.72±1.69</td>
      <td>0.94±0.69 / 0.90±0.52</td>
    </tr>
    <tr style="background-color:#e0e0e0; font-weight:bold;">
      <td>APPR (Ours)</td>
      <td><span style="color:red">0.9341±0.2012</span> / <span style="color:red">0.9502±0.1578</span></td>
      <td><span style="color:orange">0.9703±0.0212</span> / <span style="color:orange">0.9721±0.0321</span></td>
      <td><span style="color:red">1.21±1.33</span> / <span style="color:orange">1.15±0.32</span></td>
      <td><span style="color:red">0.21±0.19</span> / <span style="color:red">0.19±0.07</span></td>
    </tr>
    <tr>
      <td rowspan="8" style="font-weight:bold;">50%</td>
      <td>CGS (TMI 2025)</td>
      <td>0.8879±0.1328 / 0.8818±0.1189</td>
      <td>0.9136±0.1277 / <span style="color:orange">0.9624±0.0233</span></td>
      <td>10.55±25.17 / 5.00±10.71</td>
      <td>3.53±8.00 / <span style="color:orange">0.63±0.90</span></td>
    </tr>
    <tr>
      <td>DECSeg (PR 2025)</td>
      <td><span style="color:orange">0.9200±0.0514</span> / 0.8982±0.0800</td>
      <td>0.9429±0.0227 / 0.8978±0.0580</td>
      <td><span style="color:orange">1.40±1.17</span> / 8.37±20.05</td>
      <td><span style="color:orange">0.30±0.15</span> / 1.16±1.33</td>
    </tr>
    <tr>
      <td>STAC (BIBM 2024)</td>
      <td>0.7533±0.0625 / 0.6062±0.1148</td>
      <td>0.6162±0.0795 / 0.4626±0.1195</td>
      <td>15.17±23.88 / 11.44±25.54</td>
      <td>2.74±3.60 / 4.02±8.79</td>
    </tr>
    <tr>
      <td>MetaSSL (TMI 2025)</td>
      <td>0.6591±0.0673 / 0.5979±0.0776</td>
      <td>0.5207±0.0822 / 0.4502±0.0818</td>
      <td>41.35±3.32 / 42.57±3.72</td>
      <td>11.39±1.68 / 13.17±2.89</td>
    </tr>
    <tr>
      <td>DiffRect (MICCAI 2024)</td>
      <td>0.9039±0.0379 / 0.8904±0.0758</td>
      <td>0.9393±0.0512 / 0.9520±0.0370</td>
      <td>12.07±13.35 / 7.78±7.56</td>
      <td>2.11±1.96 / 1.90±1.07</td>
    </tr>
    <tr>
      <td>Text-SemiSeg (MICCAI 2025)</td>
      <td>0.8913±0.0282 / 0.8463±0.0972</td>
      <td>0.8421±0.0552 / 0.7758±0.1369</td>
      <td>77.50±66.81 / 85.80±51.76</td>
      <td>17.75±11.39 / 26.30±21.90</td>
    </tr>
    <tr>
      <td>SynFoC (CVPR 2025)</td>
      <td>0.8442±0.1702 / <span style="color:orange">0.9181±0.0537</span></td>
      <td><span style="color:orange">0.9554±0.0408</span> / 0.9317±0.0504</td>
      <td>4.29±4.59 / <span style="color:orange">2.43±1.76</span></td>
      <td>1.04±0.91 / 0.79±0.56</td>
    </tr>
    <tr style="background-color:#e0e0e0; font-weight:bold;">
      <td>APPR (Ours)</td>
      <td><span style="color:red">0.9387±0.1989</span> / <span style="color:red">0.9588±0.1032</span></td>
      <td><span style="color:red">0.9733±0.0583</span> / <span style="color:red">0.9789±0.0792</span></td>
      <td><span style="color:red">1.18±0.98</span> / <span style="color:red">1.01±0.35</span></td>
      <td><span style="color:red">0.21±0.14</span> / <span style="color:red">0.18±0.08</span></td>
    </tr>
  </tbody>
</table>
<p align="center"><em>Table 1: Quantitative comparison between APPR and state-of-the-art methods.</em></p>

### Qualitative Visualization Results
<p align="center">
  <img src="figures/visualization.png" alt="Visualization Results" width="900">
</p>
<p align="center"><em>Figure 3: Qualitative visualization of maxillary sinus segmentation results.</em></p>

### Ablation Study & Heatmap Analysis
<p align="center">
<img src="figures/heatmap.png" alt="Ablation Heatmap Results" width="800">
</p>
<p align="center"><em>Figure 4: Heatmap analysis for module effectiveness validation.</em></p>

<!--### Ablation study under 20% labeled data

| Method | Dice_fg | Left Dice | Right Dice | Mean Dice | Mean IoU | Mean PPV | Mean HD95 ↓ | Mean ASSD ↓ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| E1 Baseline | 0.2197 | 0.2288 | 0.2099 | 0.2194 | 0.1247 | 0.1268 | 45.0352 | 10.6486 |
| E2 S1 only | 0.6048 | - | - | - | - | - | - | - |
| E3 S2 only | 0.0782 | 0.0801 | 0.0761 | 0.0781 | 0.0409 | 0.0415 | 45.9121 | 11.2694 |
| E4 S3 only | 0.8914 | 0.8991 | 0.8816 | 0.8904 | 0.8045 | 0.9641 | 2.1667 | 0.5079 |
| E5 S1+S2 | 0.3376 | 0.4064 | 0.2441 | 0.3253 | 0.2021 | 0.7054 | 21.9025 | 6.2135 |
| E6 S1+S3 | 0.9034 | 0.9129 | 0.8806 | 0.8968 | 0.8308 | 0.9561 | 3.4096 | 1.5254 |
| E7 S2+S3 | 0.8974 | 0.9027 | 0.8810 | 0.8919 | 0.8226 | 0.9413 | 3.9960 | 2.4420 |
| **E8 Full** | **0.9258** | **0.9279** | **0.9230** | **0.9255** | **0.8624** | **0.9727** | **1.7719** | **0.4116** | -->
### Comprehensive Ablation Study (20% Labeled Data)
<p><strong> Comprehensive ablation study results for the left and right maxillary sinuses. Stage I, Stage II, and Stage III denote anatomy-aware pretraining, semantic misalignment rectification, and collaborative pseudo-label rectification, respectively. Since E2 performs foreground-oriented pretraining without explicit left/right multi-class supervision, only Dice<sub>fg</sub> is reported.</p>

<table border="1" cellpadding="6" cellspacing="0" style="width:100%; font-size:14px; border-collapse:collapse; text-align:center; margin:1em 0;">
  <thead>
    <tr style="border-bottom: 2px solid #333;">
      <th rowspan="2">ID</th>
      <th colspan="3">Stage</th>
      <th rowspan="2">Dice<sub>fg</sub></th>
      <th colspan="4">Left / Right Maxillary Sinus</th>
    </tr>
    <tr style="border-bottom: 1px solid #333;">
      <th>I</th>
      <th>II</th>
      <th>III</th>
      <th>Dice ↑</th>
      <th>PPV ↑</th>
      <th>HD95 ↓</th>
      <th>ASSD ↓</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>E1</td>
      <td></td>
      <td></td>
      <td></td>
      <td>0.2197</td>
      <td>0.2288±0.0695 / 0.2099±0.0657</td>
      <td>0.1328±0.0448 / 0.1208±0.0421</td>
      <td>45.21±8.29 / 44.86±9.04</td>
      <td>10.49±1.81 / 10.80±2.00</td>
    </tr>
    <tr>
      <td>E2</td>
      <td>✅</td>
      <td></td>
      <td></td>
      <td>0.6048</td>
      <td>- / -</td>
      <td>- / -</td>
      <td>- / -</td>
      <td>- / -</td>
    </tr>
    <tr>
      <td>E3</td>
      <td></td>
      <td>✅</td>
      <td></td>
      <td>0.0782</td>
      <td>0.0801±0.0279 / 0.0761±0.0291</td>
      <td>0.0426±0.0157 / 0.0405±0.0163</td>
      <td>46.47±6.99 / 45.36±6.99</td>
      <td>11.28±1.45 / 11.26±1.49</td>
    </tr>
    <tr>
      <td>E4</td>
      <td></td>
      <td></td>
      <td>✅</td>
      <td><em>0.8914</em></td>
      <td><em>0.8991±0.0287 / 0.8816±0.0458</em></td>
      <td><em>0.9651±0.0288 / 0.9631±0.0649</em></td>
      <td><em>2.04±1.02 / 2.30±1.93</em></td>
      <td><em>0.50±0.24 / 0.52±0.32</em></td>
    </tr>
    <tr>
      <td>E5</td>
      <td>✅</td>
      <td>✅</td>
      <td></td>
      <td>0.3376</td>
      <td>0.4064±0.1042 / 0.2441±0.1260</td>
      <td>0.7358±0.2296 / 0.6751±0.3057</td>
      <td>22.37±8.23 / 21.43±7.30</td>
      <td>5.53±2.64 / 6.90±4.08</td>
    </tr>
    <tr>
      <td>E6</td>
      <td>✅</td>
      <td></td>
      <td>✅</td>
      <td><em>0.9034</em></td>
      <td><em>0.9129±0.0331 / 0.8806±0.0554</em></td>
      <td><em>0.9586±0.0225 / 0.9535±0.0895</em></td>
      <td><em>2.99±1.71 / 3.83±4.85</em></td>
      <td><em>1.52±0.33 / 1.53±0.63</em></td>
    </tr>
    <tr>
      <td>E7</td>
      <td></td>
      <td>✅</td>
      <td>✅</td>
      <td>0.8974</td>
      <td>0.9027±0.0224 / 0.8810±0.0306</td>
      <td>0.9490±0.0255 / 0.9336±0.0482</td>
      <td>3.71±1.29 / 4.28±2.42</td>
      <td>2.40±0.21 / 3.49±0.42</td>
    </tr>
    <tr style="font-weight:bold; border-top: 1px solid #333;">
      <td>E8 (Full)</td>
      <td>✅</td>
      <td>✅</td>
      <td>✅</td>
      <td>0.9258</td>
      <td>0.9279±0.0253 / 0.9230±0.0291</td>
      <td>0.9722±0.0278 / 0.9733±0.0195</td>
      <td>1.98±2.83 / 1.56±0.59</td>
      <td>0.41±0.32 / 0.41±0.22</td>
    </tr>
  </tbody>
</table>
<p align="center"><em>Table 2:</strong> Comprehensive ablation study results for the left and right maxillary sinuses under 20% labeled data.</em></p>

### Key takeaways:
- **Stage III** is the main performance driver for structure completeness and boundary refinement.
- **Stage I** provides the strongest warm-start for stable foreground localization.
- **Full APRNet** achieves the best overall accuracy and the smallest bilateral performance gap.

<!--Key takeaways:

- **Stage III** is the main performance driver for structure completeness and boundary refinement.
- **Stage I** provides the strongest warm-start for stable foreground localization.
- **Full APPR** achieves the best overall accuracy and the smallest bilateral performance gap. -->

## Citation

If you use this repository in your work, please cite the APRNet paper:

```bibtex
@article{aprnet2025,
  title   = {APRNet: Anatomy-aware Pretraining and Pseudo-label Rectification for Semi-Supervised Maxillary Sinus Segmentation},
  author  = {Anonymous},
  journal = {Under preparation},
  year    = {2025}
}
```
