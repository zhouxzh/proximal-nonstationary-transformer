# Proximal Nonstationary Transformer

A time series forecasting framework featuring the **Proximal Nonstationary Transformer** — a variant of the [Nonstationary Transformer](https://arxiv.org/abs/2205.14415) (NeurIPS 2022) with configurable window normalization strategies. The repository also includes 12 additional forecasting models.

Built on top of [TSLib](https://github.com/thuml/Time-Series-Library).

## Key Features

- **Proximal Nonstationary Transformer**: Extends the Nonstationary Transformer by supporting multiple normalization modes — subtract last observed value (`LastVal`), median (`Median`), mean (`Mean`), ReVIN, and decomposition-based normalization
- **13 built-in models**: Autoformer, Transformer, Nonstationary Transformer, Proximal Nonstationary Transformer, NLinear, FEDformer, Informer, LightTS, ETSformer, Pyraformer, MICN, Crossformer, FiLM
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
| NLinear | AAAI 2023 | `models/NLinear.py` |
| Crossformer | ICLR 2023 | `models/Crossformer.py` |
| ETSformer | ICLR 2023 | `models/ETSformer.py` |
| LightTS | NeurIPS 2022 | `models/LightTS.py` |
| Pyraformer | ICLR 2022 | `models/Pyraformer.py` |
| MICN | ICLR 2023 | `models/MICN.py` |
| FiLM | NeurIPS 2022 | `models/FiLM.py` |

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

### Other Models

Replace `--model` with any model name from the table above. Example:

```bash
python run.py --model NLinear --data ETTh2 --seq_len 336 --pred_len 720 ...
```

### GLD (Gold Price) Benchmark

```bash
bash scripts/GLD.sh
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
## Directory Structure

```
├── run.py                  # Main entry point (standard training)
├── models/                 # 13 model implementations
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
