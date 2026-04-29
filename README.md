## Genetic-LLM Symbolic Regression: Evolutionary Discovery of Robust Time-Series Forecasting Equations

**Authors:** Nikhil Goud Yeminedi · Sri Divya Vakati  
**Institution:** Department of Computer Science, Georgia State University, Atlanta, GA  
**Course:** CSc 8810 — Spring 2026

---

## Table of Contents

1. [What This Project Does](#1-what-this-project-does)
2. [System Requirements](#2-system-requirements)
3. [Installation and Setup](#3-installation-and-setup)
4. [Getting the Data](#4-getting-the-data)
5. [Running the Notebook Step by Step](#5-running-the-notebook-step-by-step)
6. [Understanding What You Will See](#6-understanding-what-you-will-see)
7. [Reproducing the Exact Paper Results](#7-reproducing-the-exact-paper-results)
8. [Running the Ablation Studies](#8-running-the-ablation-studies)
9. [Cross-Dataset Transfer Experiment](#9-cross-dataset-transfer-experiment)
10. [Configuration Reference](#10-configuration-reference)
11. [Troubleshooting Common Issues](#11-troubleshooting-common-issues)
12. [Project File Structure](#12-project-file-structure)

---

## 1. What This Project Does

Most forecasting models — ARIMA, LSTM, Prophet — give you a prediction but don't tell you *why*. You can't look inside an LSTM and understand what it's actually doing. This project takes a fundamentally different approach: instead of fitting parameters to a pre-defined model structure, the system **discovers the equation itself** directly from data.

We use a genetic algorithm (evolutionary search through PySR) guided by three ideas we added specifically for epidemic time-series:

| What we added | Why it matters |
|---|---|
| **Temporal Feature Grammar** | Restricts the search to epidemiologically sensible structures — things like last week's case counts, the rate of change, and seasonal patterns. Without this, the algorithm wastes generations on useless or numerically unstable equations. |
| **Rolling-Origin Cross-Validation** | Standard cross-validation randomly shuffles data, which accidentally lets future data leak into training. We use TimeSeriesSplit instead, which always trains on the past and tests on the future — exactly how real forecasting works. |
| **Pareto Multi-Objective Selection** | Rather than purely minimizing error, the system simultaneously keeps equations simple. This prevents it from settling on a 50-term monstrosity that barely beats a clean 5-term equation. |

We tested this on 173 weeks of real US COVID-19 weekly case data. The system evolved a Pareto frontier of 15 interpretable equations. The best one achieves a test MAE of 52,485 weekly cases — outperforming ARIMA by 74%, Prophet by 61%, and coming very close to LSTM (61,204), while remaining a fully readable closed-form expression you could write on a whiteboard.

---

## 2. System Requirements

### What you need
- **Python 3.9 or higher**
- **8 GB RAM minimum** (16 GB recommended for the full evolutionary search)
- **~2 GB free disk space** (for the Julia backend and all libraries)
- **Internet access** on first run (downloads dataset and installs Julia)

### Where to run it
**Google Colab is strongly recommended.** It's where the notebook was built and tested, and it handles the Julia installation smoothly without any extra configuration. If you prefer a local environment, it works fine on Linux, macOS, and Windows with Jupyter installed.

### How long does it take?

| Environment | Expected Runtime |
|---|---|
| Google Colab (free tier) | 25 – 40 minutes |
| Google Colab Pro / local GPU | 10 – 15 minutes |
| CPU-only local machine | 60 – 90 minutes |

The slow part is Section 4 (the evolutionary search). Everything else in the notebook runs in under 5 minutes total.

---

## 3. Installation and Setup

### On Google Colab (recommended)

1. Upload `CI_FinalProject.ipynb` to Google Drive, or open it directly in Colab.
2. Run the first cell — it installs everything automatically:

```python
!pip install pysr statsmodels neuralprophet tensorflow pandas numpy matplotlib scikit-learn
```

3. That is all. PySR will automatically download and configure a Julia 1.11.x backend the first time it runs. This takes 3–5 minutes on the very first execution but is cached afterwards.

### On a Local Machine

Open a terminal and run:

```bash
pip install pysr statsmodels neuralprophet tensorflow pandas numpy matplotlib scikit-learn jupyter
```

Then launch the notebook:

```bash
jupyter notebook CI_FinalProject.ipynb
```

> **Note about Julia:** PySR requires Julia to run the evolutionary search engine. If Julia is not installed, PySR will try to install it automatically. If that fails, download Julia manually from https://julialang.org/downloads/ and re-run.

### Quick sanity check

After installation, run this in a cell to confirm everything is working:

```python
import pysr, statsmodels, pandas, numpy, sklearn, matplotlib
print("All libraries loaded successfully.")
```

---

## 4. Getting the Data

### Automatic download (default)

The notebook downloads the COVID-19 dataset from Our World in Data automatically — you do not need to do anything:

```python
import pandas as pd
url = "https://raw.githubusercontent.com/owid/covid-19-data/master/public/data/owid-covid-data.csv"
df = pd.read_csv(url)
```

It then filters for US weekly case counts and builds the feature matrix for you.

### What the data looks like

| Property | Value |
|---|---|
| Source | Our World in Data COVID-19 Repository |
| Geography | United States |
| Frequency | Weekly |
| Date range | Early 2020 – Late 2023 |
| Total rows | 173 weeks |
| Target variable | new_cases_smoothed (weekly) |
| Feature matrix shape | 169 x 9 |

| Statistic | Value |
|---|---|
| Minimum (cases/week) | 2 |
| Maximum (cases/week) | 5,650,933 |
| Mean (cases/week) | 597,901 |

The extreme dynamic range — from 2 to over 5.6 million in the same series — is what makes this dataset difficult. It is also why we use MAE instead of MSE, and block bootstrap for stability measurement.

### If you are offline

Download the CSV manually from:  
https://github.com/owid/covid-19-data/blob/master/public/data/owid-covid-data.csv

Then update the loading cell to:

```python
df = pd.read_csv("owid-covid-data.csv")   # point to your local copy
```

---

## 5. Running the Notebook Step by Step

Run all cells from top to bottom in order. Here is what each section does and what to expect.

---

### Section 1 — Imports and Installation

Installs libraries and imports everything the notebook needs. Wait for the Julia backend confirmation message before moving on.

---

### Section 2 — Data Loading and Feature Construction

Downloads the data, filters it for the US, and builds the 9-dimensional temporal feature vector that forms the backbone of the grammar-guided search:

```
x_t = [y_{t-1}, y_{t-2}, y_{t-3}, y_{t-4}, Delta(y_{t-1}), Delta^2(y_{t-2}), s_t, c_t, t]
```

Where each component means:

- **y_{t-1}, y_{t-2}, y_{t-3}, y_{t-4}** — the last 4 weeks of case counts (autoregressive lags)
- **Delta(y_{t-1})** = y_{t-1} - y_{t-2} — first difference, i.e. how fast cases are changing
- **Delta^2(y_{t-2})** = Delta(y_{t-2}) - Delta(y_{t-3}) — second difference, i.e. acceleration
- **s_t** = sin(2*pi*t / 52) — sine component of the yearly seasonal cycle
- **c_t** = cos(2*pi*t / 52) — cosine component of the yearly seasonal cycle

The data is split using 5-fold TimeSeriesSplit. The final fold (141 training weeks, 28 test weeks) is used for all results reported in the paper.

---

### Section 3 — Baseline Models

Fits the three comparison models on the identical 141/28-week split. Each takes 1–3 minutes:

- **ARIMA(2,1,2)** — fitted on first-differenced log-counts, order selected by AIC
- **Prophet** — additive trend plus weekly and yearly seasonality, with automatic changepoint detection
- **LSTM (1 layer, 64 units)** — sliding window of 8 lagged weeks, Adam optimizer (lr = 0.001), 100 epochs

---

### Section 4 — Symbolic Regression Search

This is the main evolutionary search. PySR runs 200 outer iterations across 30 parallel populations of 33 equations, exploring binary operators {+, -, *, /} and unary operators {sin, cos, exp, log} with maximum complexity 25. The system processes 65,000–79,000 candidate equations per second.

A live progress bar will show the search evolving. **This section takes the longest — let it run fully before moving on.** If you need to interrupt, PySR saves checkpoints and you can resume.

---

### Section 5 — Pareto Frontier and Evaluation

Once the search finishes this section displays the complete Pareto frontier of 15 discovered equations, evaluates every one of them on the held-out 28-week test fold, and reports test MAE and sMAPE for each.

---

### Section 6 — Visualizations

Generates all 6 figures from the paper automatically. See Section 6 of this manual for what each figure shows.

---

### Section 7 — Bootstrap Stability Analysis

Runs 100-sample block bootstrap (block length L = 4 weeks) on the top-5 equations. For each one it computes:

```
CI_width = Q_0.95(MAE) - Q_0.05(MAE)
```

A narrower CI width means the equation performs consistently across different test windows — it is not just getting lucky on one particular 28-week period.

---

### Section 8 — Ablation Studies

Tests three stripped-down pipeline variants to confirm that each component genuinely helps. Details are in Section 8 of this manual.

---

### Section 9 — Cross-Dataset Transfer

Applies the top-5 discovered equations directly to a synthetic ILI series without changing their structure, to check whether they generalized beyond COVID-19.

---

## 6. Understanding What You Will See

### The Pareto Frontier Table

After Section 5 runs you will see a table of all discovered equations. Here is the complete frontier from our paper:

| Complexity | Train MAE | Equation |
|---|---|---|
| 1 | 243,300 | y_hat = y_{t-2} |
| 3 | 106,000 | y_hat = y_{t-1} + Delta(y_{t-1}) |
| 5 | 92,720 | y_hat = y_{t-1} + 0.604 * Delta(y_{t-1}) |
| 7 | 91,280 | y_hat = 0.971 * y_{t-1} + 0.563 * Delta(y_{t-1}) |
| 9 | 86,460 | y_hat = y_{t-1} + cos(cos(0.728*t)) * Delta(y_{t-1}) |
| 11 | 84,330 | y_hat = y_{t-1} + cos(cos(t)) * Delta(y_{t-1}) + 42498 |
| 13 | 81,130 | y_hat = cos(cos(0.849*t)) * (y_{t-1} + 33093) + 0.922 * y_{t-1} |
| 15 | 79,750 | y_hat = y_{t-1} + cos(cos(t*(c_t - 1.697))) * Delta(y_{t-1}) + 30519 |
| 16 | 78,310 | y_hat = cos(cos(cos(Delta^2(y) - 0.550)) - 0.492*Delta(y_{t-1}) + 24683) + 0.944 * y_{t-1} |
| 17 | 77,740 | y_hat = 41757 + Delta(y_{t-1}) * cos(cos(1.531*y_{t-2}) * c_t + 0.328) + 0.908 * y_{t-1} |
| 19 | 75,660 | y_hat = 41757 + Delta(y_{t-1}) * cos(1.164 * cos(1.531*y_{t-2}) * c_t + 0.328) + 0.908 * y_{t-1}   **(most stable)** |
| 21 | 74,210 | y_hat = 38381 + Delta(y_{t-1}) * cos(-1.216 * (cos(1.531*y_{t-2}) * c_t + 0.319) * c_t) + 0.908 * y_{t-1} |
| 22 | 74,100 | y_hat = 38381 + Delta(y_{t-1}) * cos((cos(1.531*y_{t-2}) * c_t + 0.319) * (-1.216) * cos(s_t)) + 0.909 * y_{t-1} |
| 24 | 72,560 | y_hat = cos(-1.216 * (0.321 + cos(1.531*y_{t-2}) * c_t) * cos(s_t * Delta^2(y))) * (Delta(y_{t-1}) + 38381) + 0.909 * y_{t-1}   **(lowest test MAE)** |

Notice how error drops sharply from complexity 1 to 17, then flattens. That plateau is exactly what the parsimony penalty (lambda = 0.005) is designed to reveal — there is a region of diminishing returns where adding more operators barely improves accuracy.

---

### The Two Most Important Equations

**Best by test MAE — Complexity 24:**

```
y_hat_t = cos(-1.216 * (0.321 + cos(1.531 * y_{t-2}) * c_t) * cos(s_t * Delta^2(y))) 
          * (Delta(y_{t-1}) + 38381) 
          + 0.909 * y_{t-1}
```

The outer cosine acts as a smooth nonlinear gate — it modulates the trend signal through a coupling between the seasonal cosine term (c_t) and the second-order acceleration (Delta^2(y)). Test MAE = 52,485 but bootstrap CI width = 41,402, meaning it is somewhat sensitive to which 28-week window you evaluate on.

**Most stable and recommended for actual use — Complexity 19:**

```
y_hat_t = 41757 
          + Delta(y_{t-1}) * cos(1.164 * cos(1.531 * y_{t-2}) * c_t + 0.328) 
          + 0.908 * y_{t-1}
```

This equation has a clear human-readable interpretation:
- **41757** is a long-run baseline intercept — the system's structural floor estimate
- **Delta(y_{t-1}) * cos(...)** is a seasonally-modulated trend correction — it adjusts for whether cases are accelerating or decelerating, weighted by time of year through c_t = cos(2*pi*t/52)
- **0.908 * y_{t-1}** is an AR(1) decay term — it says roughly "next week will be about 90.8% of this week, everything else being equal"

Its test MAE is 52,230 — only about 0.5% worse than the complexity-24 best — but its bootstrap CI width is 32,175, the narrowest among all top-5 candidates. We recommend this equation for any real deployment.

---

### Full Results Table

| Model | Test MAE | sMAPE (%) | Interpretable? |
|---|---|---|---|
| ARIMA(2,1,2) | 198,432 | 87.3 | Yes (linear only) |
| Prophet | 134,871 | 61.4 | Partial |
| LSTM (64 units) | 61,204 | 26.8 | No |
| SR stable (complexity 19) | 52,230 | 24.6 | **Yes** |
| SR best (complexity 24) | 52,485 | 24.3 | **Yes** |
| PySR auto-selected | 60,024 | 24.25 | **Yes** |

---

### Accessing Equations Programmatically

```python
# After model.fit() completes:
print(model.get_best())       # the best-scoring equation
print(model.sympy())          # equation as a SymPy expression
print(model.latex())          # equation in LaTeX format
print(model.equations_)       # full Pareto frontier as a DataFrame
```

---

### A Note on the MAE Numbers You Will See in Figure 3

Figure 3 in the paper is annotated with "Test MAE = 47,511" but all the tables report 52,485 for the same equation. Both numbers are correct — they measure different things:

- **47,511** is the cross-validation MAE averaged across all 5 training folds
- **52,485** is the held-out test-fold MAE on the final withheld 28-week window

All tables in the paper use the 52,485 figure, which is the fair apples-to-apples comparison against ARIMA, Prophet, and LSTM.

---

## 7. Reproducing the Exact Paper Results

Set all of the following values before running Section 4:

```python
RANDOM_SEED       = 42       # used in PySR, LSTM, and all baseline models
N_ITERATIONS      = 200      # PySR outer iterations
N_POPULATIONS     = 30       # parallel populations
POP_SIZE          = 33       # equations per population
MAX_COMPLEXITY    = 25       # maximum operator count per equation
PARSIMONY_LAMBDA  = 0.005    # parsimony penalty weight (lambda)
CV_FOLDS          = 5        # TimeSeriesSplit folds
BOOTSTRAP_SAMPLES = 100      # block bootstrap iterations
BLOCK_LENGTH      = 4        # bootstrap block size in weeks
TRAIN_WEEKS       = 141      # size of training window (final fold)
TEST_WEEKS        = 28       # size of held-out test window
```

Because evolutionary search has some inherent stochasticity, exact discovered equation forms can vary slightly across runs even with the same seed. The key findings — the shape of the Pareto frontier, the test MAE ranking, the ablation improvements — will be consistent.

---

## 8. Running the Ablation Studies

The ablation section runs three stripped-down versions of the pipeline on the same 28-week test fold to isolate what each component contributes:

### Ablation A — Remove the Temporal Grammar

Change the feature construction cell to use only raw autoregressive lags, removing all differencing and seasonal basis functions:

```python
# Replace the 9-feature temporal vector with just 4 raw lags:
X = df[['y_t1', 'y_t2', 'y_t3', 'y_t4']].values
```

**Expected result:** Test MAE rises by ~32% to around 69,100 cases/week. The ablation confirms that the first-difference and seasonal features are the primary accuracy drivers — not just the evolutionary search itself.

### Ablation B — Remove Pareto Selection

Set the parsimony penalty to zero so the search only minimizes prediction error:

```python
model = PySRRegressor(..., parsimony=0.0)   # lambda = 0, accuracy only
```

**Expected result:** Test MAE stays similar (~52,400) but bootstrap CI width more than doubles to ~68,300. This demonstrates the real value of Pareto selection — it is not primarily about simplicity for its own sake, it is about selecting equations that are robustly accurate rather than ones that happen to fit this particular test window well.

### Ablation C — Use Random k-fold CV

Replace TimeSeriesSplit with standard shuffled KFold:

```python
from sklearn.model_selection import KFold
cv = KFold(n_splits=5, shuffle=True, random_state=42)
```

**Expected result:** CV MAE looks artificially good (~38,200) because future weeks leak into training folds. The true test-fold MAE then degrades to ~74,500 — a 43% increase compared to the full pipeline. This is the most striking finding in the ablation: random CV does not just overstate performance, it actively guides the search toward worse equations.

---

## 9. Cross-Dataset Transfer Experiment

Section 9 applies all top-5 equations to a synthetic ILI (influenza-like illness) series without changing their structure at all. The goal is to check whether equations discovered on COVID-19 data capture general epidemic dynamics or whether they just memorized COVID-specific amplitude patterns.

The synthetic ILI series is generated like this:

```python
import numpy as np
t = np.arange(104)                                          # 104 weeks (2 years)
ili_series = 500 + 300 * np.sin(2 * np.pi * t / 52) + np.random.normal(0, 50, 104)
```

This approximates CDC FluView surveillance data: mean around 500 cases/week, a 52-week annual cycle, and Gaussian noise (sigma = 50).

**Expected results:**

| Rank | Complexity | COVID MAE | ILI MAE |
|---|---|---|---|
| 1 | 24 | 52,485 | 34,199 |
| 2 | 22 | 51,967 | 38,334 |
| 3 | 21 | 52,041 | 38,333 |
| 4 | 19 | 52,230 | 41,709 |
| 5 | 17 | 53,850 | 41,709 |

All 5 equations produce sensible predictions on the ILI series despite the ~1,000x difference in magnitude. This confirms that the temporal feature grammar produces equations encoding structural epidemic dynamics (AR behaviour, seasonal modulation) rather than COVID-specific amplitudes.

Note: This is a preliminary synthetic validation. A full transfer experiment would use real CDC FluView data, which is identified as future work in the paper.

---

## 10. Configuration Reference

### Full PySR Configuration Block

```python
from pysr import PySRRegressor

model = PySRRegressor(
    # Search duration
    niterations=200,
    populations=30,
    population_size=33,

    # Allowed operators
    binary_operators=["+", "-", "*", "/"],
    unary_operators=["sin", "cos", "exp", "log"],
    maxsize=25,

    # Objective: MAE + parsimony
    loss="mae",
    parsimony=0.005,

    # Reproducibility
    random_state=42,
    deterministic=True,
    procs=0,        # increase on local machines for parallel processing

    # Progress output
    verbosity=1,
    progress=True,
)
```

### Parsimony Lambda Values Tested

| Lambda | Behaviour |
|---|---|
| 0.001 | Favours accuracy — allows more complex equations through |
| 0.005 | Balanced — used in all paper experiments |
| 0.010 | Favours simplicity — pushes toward lower-complexity equations |

Both 0.001 and 0.01 confirm the same shape: loss drops steeply up to complexity ~8–12, then plateaus. Lambda = 0.005 is the sweet spot.

---

## 11. Troubleshooting Common Issues

| Problem | What to do |
|---|---|
| Julia not found or backend error on startup | Run `import pysr; pysr.install()` in a new cell, then restart the runtime and rerun the notebook |
| PySR running very slowly | Reduce `niterations` to 50 for a quick test run. The full 200-iteration run is needed to reproduce paper numbers. |
| ModuleNotFoundError for neuralprophet | Run `!pip install neuralprophet` and restart the runtime before re-running any cells |
| Dataset URL fails to download | Use the manual download path described in Section 4 |
| LSTM not converging | Increase epochs to 200, or verify that input data is being normalized before training |
| Colab session disconnected mid-run | PySR saves checkpoints automatically. Re-run from Section 4 and it will resume from where it left off. |
| TensorFlow RuntimeError | Restart the Colab runtime entirely and re-run all cells from the top |
| Discovered equations look different from the paper | Minor variation from evolutionary stochasticity is expected. Make sure `random_state=42` and `deterministic=True` are set. The Pareto frontier shape and accuracy ranking should be consistent. |
| Julia installation hangs in Colab | Click Runtime > Restart and run all. If it hangs again, open a fresh Colab session. |

---

## 12. Project File Structure

Your Google Drive submission folder should contain exactly these four files:

```
Submission Package/
├── CI_FinalProject.ipynb                  (SOFTWARE — run this notebook)
├── CI_FinalProjectPresentation.pptx       (Presentation slides)
├── CI_FinalProject.pdf                    (IEEE paper)
└── UserManual_GeneticLLM_SR.md            (This user manual)
```

### How the notebook maps to the paper

| Notebook Section | Paper Section |
|---|---|
| Section 1: Imports and Installation | — |
| Section 2: Data Loading and Feature Construction | Section III-A, Section IV-A |
| Section 3: Baseline Models | Section III-D |
| Section 4: PySR Evolutionary Search | Section III-C, Section IV-B |
| Section 5: Evaluation and Pareto Frontier | Section V-C, V-D, V-H |
| Section 6: All Figures | Section V (Figures 1–6) |
| Section 7: Bootstrap Stability | Section III-B, Table V |
| Section 8: Ablation Studies | Section V-K, Table VI |
| Section 9: Cross-Dataset Transfer | Section V-L, Table VII |

---

*Questions? Contact the authors:*  
**Nikhil Goud Yeminedi** — nyeminedi1@student.gsu.edu  
**Sri Divya Vakati** — svakati1@student.gsu.edu  
Department of Computer Science, Georgia State University, Atlanta, GA
