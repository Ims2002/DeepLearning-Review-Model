# Trustpilot Review Analysis — Xero (Business Services)

**Deep Learning Practical Case · Machine Learning Module Final Project**

---

## Overview

This project implements an end-to-end **NLP pipeline** to analyze 100 Trustpilot reviews for [Xero](https://www.xero.com) (cloud accounting platform) and benchmark them against 5 competitors in the Business Services sector.

The analysis answers four business questions posed by the Customer Experience director:

1. Are the majority of reviews positive or negative? How does Xero compare to competitors?
2. What topics/themes do the reviews cover?
3. For each topic, is sentiment predominantly positive or negative? Where does Xero outperform or underperform competitors?
4. What are the priority areas for improvement?

---

## Tech Stack

| Component | Library / Model |
|-----------|----------------|
| Sentiment Analysis | `cardiffnlp/twitter-roberta-base-sentiment-latest` (RoBERTa) |
| Topic Modeling (classical) | NMF + TF-IDF (`scikit-learn`) |
| Topic Modeling (neural) | BERTopic (`sentence-transformers` + UMAP + HDBSCAN) |
| Embeddings | `all-MiniLM-L6-v2` (Sentence Transformers) |
| Data manipulation | `pandas`, `numpy` |
| Visualizations | `matplotlib`, `seaborn`, `wordcloud` |
| Pattern matching | `regex` (Unicode-aware superset of `re`) |

**Python version:** 3.10+

---

## Dataset

- **Source:** `trustpilot-reviews-123k.csv` — 123,181 reviews across 1,680 companies and 22 sectors
- **Target company:** `www.xero.com` (100 reviews)
- **Competitors analyzed:** `crunch.co.uk`, `www.takepayments.com`, `skrill.com`, `www.clearpay.co.uk`, `eposnow.com`

> ⚠️ **Methodological note:** Star ratings are **not used** as sentiment labels. The dataset is artificially balanced (exactly 20 reviews per star level per company), which invalidates stars as a supervised signal. Sentiment is inferred exclusively from review text via the RoBERTa model.

---

## Project Structure

```
Proyecto Módulo/
├── analisis_xero_trustpilot.ipynb   # Main analysis notebook
├── trustpilot-reviews-123k.csv      # Dataset (not tracked in git)
├── README.md                        # This file
├── README_personal.md               # Technical deep-dive notes
└── context/                         # Academic background (course materials)
    ├── 09_Deep_Learning_ev.pdf
    ├── 10_NLP_ev (1).pdf
    └── [notebooks de referencia del curso]
```

---

## How to Run

### 1. Create and activate a virtual environment

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate
```

### 2. Install dependencies

```bash
pip install transformers torch scikit-learn pandas numpy matplotlib seaborn wordcloud
pip install bertopic sentence-transformers umap-learn hdbscan
pip install regex jupyter
```

> First execution downloads the RoBERTa model (~500 MB) and `all-MiniLM-L6-v2` (~90 MB) from HuggingFace Hub. Subsequent runs use the local cache.

### 3. Place the dataset

Copy `trustpilot-reviews-123k.csv` to the same directory as the notebook.

### 4. Launch Jupyter and run all cells

```bash
jupyter notebook analisis_xero_trustpilot.ipynb
```

Run cells sequentially. Expected total runtime (CPU): **15–40 minutes** depending on hardware (the sentiment inference loop is the bottleneck).

---

## Notebook Structure

| Section | Description |
|---------|-------------|
| 0. Installation | Optional dependency install via `%pip` |
| 1. Libraries | Imports and global style settings |
| 2. Configuration | `TARGET`, `COMPETITORS`, `N_TOPICS` and other parameters |
| 3. Data Loading | CSV ingestion, descriptive stats, star-bias verification |
| 4. Linguistic Exploration | Pattern detection with `regex`, length distribution |
| 5. Text Cleaning | Preprocessing pipeline (URLs, whitespace, non-ASCII) |
| 6. Corpus Construction | Filtering by sector, train/analysis split |
| 7. Sentiment Analysis | RoBERTa inference, global and comparative visualizations |
| 8. Topic Modeling | NMF + TF-IDF (§8.1) and BERTopic (§8.2), cross-validation |
| 9. Sentiment × Topic | Net Sentiment Score (NSS) per topic, competitive benchmark |
| 10. Conclusions | Executive summary, priority improvement areas |

---

## Key Metric: Net Sentiment Score (NSS)

```
NSS = % positive reviews − % negative reviews  (per topic)
```

- NSS > 0 → the topic generates more satisfaction than dissatisfaction
- NSS < 0 → priority area for improvement

---

## License

Academic project — Evolve Machine Learning Programme. Dataset and course materials are proprietary.
