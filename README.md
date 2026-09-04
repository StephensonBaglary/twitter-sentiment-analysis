# Tuberculosis Trend Analysis using Python and LSTM

Analysis of two decades of WHO tuberculosis (TB) burden data, followed by
an LSTM (TensorFlow/Keras) that forecasts next-year country-level TB
incidence, trained as a single shared model across all countries at once
(a "global panel" approach).

## Dataset

`data/who_tb_data.csv` — WHO Global Tuberculosis Programme burden
estimates: **5,117 country-year records, 215 countries, 2000–2023 (24
years)**. Columns include estimated incidence and mortality (overall and
HIV-stratified) per 100,000 population, case detection rate, case fatality
ratio, and population, broken down by WHO region.

Source: [WHO Global Tuberculosis Programme](https://www.who.int/teams/global-tuberculosis-programme/data),
mirrored via the [tidytuesday project](https://github.com/rfordatascience/tidytuesday/blob/main/data/2025/2025-11-11/readme.md).

## Project structure

```
tuberculosis-trend-analysis/
├── data/
│   └── who_tb_data.csv
├── src/
│   ├── 01_eda.py                 # global trends, HIV co-infection, detection vs. mortality
│   └── 02_lstm_forecasting.py    # LSTM incidence forecasting vs. naive baseline
├── outputs/                      # generated on run: figures, predictions, metrics
├── requirements.txt
└── README.md
```

## How to run

```bash
pip install -r requirements.txt
python src/01_eda.py
python src/02_lstm_forecasting.py
```

## 1. Exploratory Data Analysis (`01_eda.py`)

Three findings, each population-weighted across all 215 countries:

**Mortality has fallen much faster than new-case incidence.** From 2000 to
2023, the global TB mortality rate fell from 42.6 to 15.1 per 100,000
(**-64.5%**), while incidence fell from 180.6 to 134.5 per 100,000
(**-25.5%**). Fewer people are dying of TB per case than in 2000, mainly
because treatment access and drug regimens have improved faster than
transmission has actually been reduced.

**Africa's TB burden is disproportionately HIV-linked.** Averaged across
2000-2023, **39.4%** of TB deaths in the WHO Africa region were in people
co-infected with HIV, versus a 9.5% average across all other WHO regions —
roughly **4x**. Every other region sits in single digits to low twenties.

**Case detection rate and mortality are negatively correlated**, as
expected: r = **-0.53** across 4,800 country-year records with both
fields present. Countries that catch a higher share of their TB cases
tend to see substantially lower mortality — the strongest lever in this
dataset for reducing TB deaths isn't a new drug, it's finding the cases
that already exist.

Figures (mortality-vs-incidence trend line, regional HIV co-infection bar
chart, detection-vs-mortality scatter) are saved to `outputs/figures/`.

## 2. LSTM Incidence Forecasting (`02_lstm_forecasting.py`)

**Setup:** a single LSTM is trained across all countries' time series at
once (rather than fitting 200+ separate per-country models). Each country's
incidence series is min-max scaled to its own range, split into 5-year
sliding windows, and the model predicts the *change* from the last known
year to the next year (not the raw next-year level — see below for why).
The most recent 3 years per country are held out as a test set (a genuine
future-forecasting split, not a random shuffle), evaluated against a naive
persistence baseline (predict next year = this year).

### Results (this run)

| Model | MAE (incidence per 100k) |
|---|---|
| LSTM | **5.71** |
| Naive persistence (predict last year's value) | **5.68** |

The LSTM essentially **matches** the naive baseline; it does not beat it.

**This is a real and worth-reporting finding, not a bug.** TB incidence
estimates move very little year-to-year — the median absolute year-over-
year change across the whole dataset is only about 5% of the value — which
makes "predict last year's number" an unusually strong baseline for any
1-step-ahead model to beat. The first version of this model (a larger,
unregularized 2-layer LSTM trained for a fixed 40 epochs) actually scored
*worse* than the naive baseline (MAE 7.83 vs. 5.68, i.e. -38%): it was
overfitting noise in the training deltas that didn't generalize to future
years. Shrinking the network to 4 LSTM units, adding L2 weight decay and
50% dropout, switching to MAE loss, and stopping on validation loss
(instead of a fixed epoch count) is what brought it back down to matching
persistence rather than actively hurting it — that debugging process is
recorded in the script's `build_model()` docstring.

Concrete next steps that would give the model a realistic shot at actually
beating persistence: exogenous covariates that lead incidence (case
detection rate, funding/program data), a longer lookback window, or
pooling countries by WHO region so the model isn't forced to generalize
a single set of weights across radically different epidemic dynamics (a
country in Western Pacific and a country in Africa do not behave the same
way year to year).

Predictions and metrics from this run are saved to
`outputs/tb_forecast_predictions.csv` and `outputs/forecast_metrics.txt`.

> **Note on numbers:** the figures above come from actually running this
> code against the dataset in `data/`. If you're comparing against a
> different write-up of this project with different numbers, that version
> likely used a different WHO data release or forecasting setup — rerun
> the scripts to reproduce the numbers above from scratch.
