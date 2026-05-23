# Proximal Nonstationary Transformer

A time series forecasting framework featuring the **Proximal Nonstationary Transformer** — a variant of the [Nonstationary Transformer](https://arxiv.org/abs/2205.14415) (NeurIPS 2022) with configurable window normalization strategies. The repository also includes 16 additional forecasting models and the AutoCon contrastive learning method from [Self-Supervised Contrastive Learning for Long-term Forecasting](https://openreview.net/forum?id=nBCuRujqK7) (ICLR 2024).

Built on top of [TSLib](https://github.com/thuml/Time-Series-Library).

## Key Features

- **Proximal Nonstationary Transformer**: Extends the Nonstationary Transformer by supporting multiple normalization modes — subtract last observed value (`LastVal`), median (`Median`), mean (`Mean`), ReVIN, and decomposition-based normalization
- **17 built-in models**: TimesNet, Autoformer, Transformer, Nonstationary Transformer, Proximal Nonstationary Transformer, DLinear, NLinear, FEDformer, Informer, LightTS, ETSformer, PatchTST, Pyraformer, MICN, Crossformer, FiLM, AutoConNet
- **AutoCon contrastive learning**: Autocorrelation-based self-supervised contrastive loss and CNN-based architecture (AutoConNet)
- **Gold price benchmark**: Includes GLD dataset and scripts for systematic evaluation across multiple prediction lengths and lookback windows

## Environment Setup

```
python >= 3.8
torch >= 1.7.1
numpy == 1.23.5
pandas
statsmodels
scikit-learn
einops
sympy
numba
```

Install dependencies:
```bash
pip install torch numpy==1.23.5 pandas statsmodels scikit-learn einops sympy numba
```

## Models

| Model | Paper | Source |
|-------|-------|--------|
| Proximal Nonstationary Transformer | — (this repo) | `models/Proximal_Nonstationary_Transformer.py` |
| Nonstationary Transformer | NeurIPS 2022 | `models/Nonstationary_Transformer.py` |
| Transformer | NeurIPS 2017 | `models/Transformer.py` |
| Autoformer | NeurIPS 2021 | `models/Autoformer.py` |
| FEDformer | ICML 2022 | `models/FEDformer.py` |
| Informer | AAAI 2021 | `models/Informer.py` |
| TimesNet | ICLR 2023 | `models/TimesNet.py` |
| PatchTST | ICLR 2023 | `models/PatchTST.py` |
| DLinear | AAAI 2023 | `models/DLinear.py` |
| NLinear | AAAI 2023 | `models/NLinear.py` |
| Crossformer | ICLR 2023 | `models/Crossformer.py` |
| ETSformer | ICLR 2023 | `models/ETSformer.py` |
| LightTS | NeurIPS 2022 | `models/LightTS.py` |
| Pyraformer | ICLR 2022 | `models/Pyraformer.py` |
| MICN | ICLR 2023 | `models/MICN.py` |
| FiLM | NeurIPS 2022 | `models/FiLM.py` |
| AutoConNet / AutoConCI | ICLR 2024 | `models/AutoConNet.py` |

## Usage

### Proximal Nonstationary Transformer

```bash
python run.py \
  --model Proximal_Nonstationary_Transformer \
  --wnorm LastVal \
  --data ETTh2 \
  --seq_len 336 \
  --pred_len 720 \
  --d_model 16 \
  --d_ff 16 \
  --e_layers 3 \
  --enc_in 1 \
  --c_out 1 \
  --batch_size 64 \
  --learning_rate 0.005 \
  --feature S \
  --itr 1
```

The `--wnorm` parameter controls the normalization strategy:
- `LastVal` — subtract the last observed value (default, proximal normalization)
- `Mean` — subtract the mean (same as original Nonstationary Transformer)
- `Median` — subtract the median

### AutoCon (Self-Supervised Contrastive Learning)

```bash
python run_autocon.py \
  --AutoCon \
  --AutoCon_multiscales 720 \
  --AutoCon_wnorm LastVal \
  --AutoCon_lambda 0.1 \
  --data ETTh2 \
  --seq_len 336 \
  --pred_len 720 \
  --model AutoConNet \
  --d_model 16 \
  --d_ff 16 \
  --e_layers 3 \
  --enc_in 1 \
  --c_out 1 \
  --feature S \
  --batch_size 64 \
  --learning_rate 0.005 \
  --itr 1
```

### Other Models

Replace `--model` with any model name from the table above. Example:

```bash
python run.py --model PatchTST --data ETTh2 --seq_len 336 --pred_len 720 ...
```

### GLD (Gold Price) Benchmark

```bash
bash scripts/GLD.sh
bash scripts/AutoCon_GLD.sh
```

### Evaluate Saved Predictions

```bash
python metrics.py
```

## Key Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--model` | Model name | `Proximal_Nonstationary_Transformer` |
| `--wnorm` | Normalization mode | `LastVal` |
| `--data` | Dataset name | `custom` |
| `--seq_len` | Input sequence length | `96` |
| `--pred_len` | Prediction length | `96` |
| `--d_model` | Model dimension | `512` |
| `--e_layers` | Encoder layers | `2` |
| `--d_layers` | Decoder layers | `1` |
| `--d_ff` | FFN dimension | `2048` |
| `--enc_in` | Input channels | `7` |
| `--c_out` | Output channels | `7` |
| `--feature` | Feature type (`M`/`S`/`MS`) | `M` |
| `--itr` | Experiment repetitions | `1` |
| `--train_ratio` | Train split ratio (custom data) | `0.7` |
| `--AutoCon` | Enable AutoCon loss | `False` |
| `--AutoCon_lambda` | AutoCon loss weight | `0.1` |
| `--AutoCon_multiscales` | AutoCon multi-scale window | `pred_len` |

## Directory Structure

```
├── run.py                  # Main entry point (standard training)
├── run_autocon.py          # AutoCon contrastive learning entry point
├── models/                 # 17 model implementations
├── layers/                 # Attention, embeddings, encoders/decoders
├── exp/                    # Experiment classes (training/testing logic)
├── data_provider/          # Data loading and preprocessing
├── utils/                  # Metrics, early stopping, time features, DTW loss
├── scripts/                # Shell scripts for running experiments
├── dataset/                # Dataset files (GLD.csv)
├── checkpoints/            # Saved model weights
├── results/                # Prediction outputs (.npy)
└── logs/                   # Training logs
```

## Datasets

Pre-configured datasets: `ETTh1`, `ETTh2`, `ETTm1`, `ETTm2`, `electricity`, `traffic`, `weather`, `exchange_rate`, `national_illness`, and `custom` (for user-provided CSV files like `GLD.csv`).

## License

MIT License. See [LICENSE](LICENSE).

## Acknowledgments

This repository builds on [TSLib](https://github.com/thuml/Time-Series-Library) and the [Self-Supervised Contrastive Forecasting](https://github.com/junwoopark92/Self-Supervised-Contrastive-Forecasting) repository.
