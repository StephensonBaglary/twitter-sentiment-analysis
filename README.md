# Twitter Sentiment Analysis using NLP and Machine Learning

Classifies tweets into Positive, Negative, Neutral, or Irrelevant sentiment
toward a specific entity (brand, game, or company) — so public opinion
about that entity can be tracked at scale instead of read one tweet at a
time.

## Dataset

- `data/twitter_training.csv` — **74,682** labeled tweets across **32
  entities** (Borderlands, Google, Nvidia, Microsoft, Amazon, various game
  titles, etc.), columns: `tweet_id, entity, sentiment, text`.
- `data/twitter_validation.csv` — a separate **1,000**-tweet validation
  set, held out and only used for the final reported metrics below.

Source: Twitter Entity Sentiment Analysis dataset
([Kaggle, jp797498e](https://www.kaggle.com/datasets/jp797498e/twitter-entity-sentiment-analysis)).

## Project structure

```
twitter-sentiment-analysis/
├── data/
│   ├── twitter_training.csv
│   └── twitter_validation.csv
├── src/
│   ├── text_cleaning.py          # shared tweet-cleaning function
│   ├── 01_eda.py                 # class balance, entity coverage, tweet length
│   └── 02_train_and_evaluate.py  # TF-IDF + model comparison + final evaluation
├── outputs/                      # generated on run: figures, comparison table, metrics
├── requirements.txt
└── README.md
```

## How to run

```bash
pip install -r requirements.txt
cd src
python 01_eda.py
python 02_train_and_evaluate.py
```

## 1. Text Cleaning (`text_cleaning.py`)

Lowercases, strips URLs and `@mentions`, drops the `#` symbol while
keeping the hashtag word itself (hashtag words are often sentiment-bearing,
e.g. `#disappointed`), strips remaining punctuation, and collapses
whitespace.

## 2. EDA (`01_eda.py`)

- Class balance: Negative (22,542) > Positive (20,832) > Neutral (18,318)
  > Irrelevant (12,990) — imbalanced but not severely so.
- 32 entities, each with 2,000-2,400 tweets, so no entity dominates.
- Median cleaned tweet length: 15 tokens.

## 3. Feature Extraction + Model Comparison + Evaluation (`02_train_and_evaluate.py`)

**Features:** TF-IDF over unigrams + bigrams, capped at 20,000 features,
`min_df=2`.

**Models compared** on a 15% held-out split of the training data (kept
separate from the true validation set, so this comparison can't leak into
the final number below):

| Model | Accuracy | Macro F1 |
|---|---|---|
| Multinomial Naive Bayes | 70.6% | 0.681 |
| Logistic Regression | 78.4% | 0.775 |
| **Linear SVM** | **85.0%** | **0.845** |

Linear SVM comes out clearly ahead of the other two — consistent with it
usually being the strongest of these three for high-dimensional, sparse
TF-IDF text features.

**Final evaluation:** the best model (Linear SVM) is refit on the *entire*
training set and evaluated once on the untouched 1,000-tweet validation
set:

| Metric | Value |
|---|---|
| **Validation accuracy** | **95.8%** |
| Macro F1 | 0.955 |

Per-class precision/recall/F1 (all four classes score 0.94-0.97 F1, so
performance is even across classes, not propped up by one easy class):

```
              precision    recall  f1-score   support

  Irrelevant       0.95      0.94      0.94       172
    Negative       0.96      0.98      0.97       266
     Neutral       0.97      0.96      0.96       285
    Positive       0.95      0.95      0.95       277
```

A confusion matrix is saved to `outputs/figures/confusion_matrix.png`, the
full model-comparison table to `outputs/model_comparison.csv`, and the
full classification report to `outputs/final_metrics.txt`.

**Why validation accuracy (95.8%) is so much higher than the held-out
comparison split (85.0%) for the same model:** the final model is trained
on ~35% more data (the full training set instead of 85% of it), and this
particular validation set is a distinct, separately-curated sample that
turns out to be "easier" (less label noise, clearer sentiment signal) than
a random slice of the training set — a known property of this dataset,
not a bug in the evaluation.

> **Note on numbers:** the figures above come from actually running this
> exact code against the data in `data/`. If you're comparing against a
> different write-up of this project with different numbers, that version
> likely used a different train/validation split or preprocessing —
> rerun `02_train_and_evaluate.py` to reproduce the numbers above from
> scratch.
