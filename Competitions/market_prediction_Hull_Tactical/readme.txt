# Hull Tactical Market Prediction (Kaggle)

It builds and evaluates models for the daily S&P 500 excess return and for the position sizing that follows from it.

> **Status:** active research. The main finding so far is that **the size of daily moves is predictable and their direction is not**. The current work turns that into a volatility-based position.

---

## 1. Goal

Each day t, submit a position **s_t ∈ [0, 2]** in the S&P 500.

| s_t | Meaning |
|---|---|
| 0 | All in cash. Earns the risk-free rate f. |
| 1 | Buy and hold (100 % in the index). |
| 1 < s ≤ 2 | Leveraged. The extra part is borrowed at the risk-free rate in the score. |

Strategy return on day t: `strat_t = f_t·(1 − s_t) + s_t·r_t`

- r_t: forward return of the S&P 500 (day t to t+1).
- f_t: risk-free rate (cash reward and borrowing cost).
- Excess return of the strategy: `s_t·(r_t − f_t)`.

**Position rule used in this project:** `s_t = clip(1 + k·ŷ_t, 0, 2)`, where ŷ_t is the predicted excess return (my k = 100, the host's = 400).

### Competition metric

1. **Sharpe** = (geometric mean of strategy excess return) / (std, ddof = 1) × √252.
2. **Volatility penalty:** `vol_pen = 1 + max(0, vol_ratio − 1.2)`, where `vol_ratio` is the strategy volatility divided by the market volatility.
3. **Return penalty:** `ret_pen = 1 + gap²/100`, with `gap = max(0, (mkt_geo − strat_geo)·100·252)`. It punishes under-performing the market.
4. **Final score:** `adjusted = Sharpe / (vol_pen · ret_pen)`.

Why this matters:

- Scaling the whole position up or down by a constant does **not** raise the Sharpe. The daily Sharpe is approximately `μ/σ − cσ/2` (volatility drag).
- The penalties punish scaling up (volatility) and scaling down (return gap).
- Only real timing skill helps, or moving the position against volatility.

---

## 2. Data and variables

**Source:** Kaggle competition data. 9,048 daily rows of anonymized features plus targets.

| Group | Meaning | Examples |
|---|---|---|
| **D** | Dummy / binary indicators | D1 … |
| **E** | Macro-economic features | E4, E6, E13 … |
| **I** | Interest-rate features | I6 … |
| **M** | Market dynamics / technical | M4, M6, M9, M15 … |
| **P** | Price / valuation | P1, P3, P4, P12, P13 … |
| **S** | Sentiment | S9, S11, S12 … |
| **V** | Volatility | V6 … |

### Targets and columns

| Column | Definition |
|---|---|
| `forward_returns` | S&P 500 return from day t to t+1 (the future at row t). |
| `risk_free_rate` | Risk-free rate f. |
| `market_forward_excess_returns` (**`target`**) | Excess return = forward return minus a rolling-mean benchmark, as defined by the host. |
| `lagged_*` | Test-format columns. On row t they hold the **day t−1** values. |

### Engineered volatility features (`vol_features.csv`)

Volatility = standard deviation of daily returns (how large the moves are). All features use only returns up to t−1, so they are causal.

| Feature | Formula |
|---|---|
| `abs_ret_lag1` | \|r_{t−1}\| |
| `vol_5`, `vol_22`, `vol_63` | Rolling std of r over the last 5, 22, 63 days (trailing windows on `forward_returns.shift(1)`). |
| `vol_ewm_094` | √(EWMA of r², α = 0.06), i.e. RiskMetrics-style decay 0.94. |
| `vol_ratio_5_63`, `vol_ratio_22_63` | Short-term volatility divided by long-term volatility. |

### Alignment rules (to avoid leakage)

- Train row t: features(t) and `forward_returns[t]` (the future). In test format the target is shifted, with `lagged_forward_returns = forward_returns.shift(1)`.
- Alignment check: correlation at offset 0 is **0.9999**.
- Anything fitted (scalers, KMeans, c, k, windows, hyperparameters) uses **train only** or is chosen on validation. The test set is used once.

---

## 3. Repository and workflow

```
Competitions/market_prediction_Hull_Tactical/scripts/
├── Baseline_model                                      linear baselines
├── data_exploration                                    splits + rolling volatility features
├── RandomForest_XGBRegressor                           tree-based baselines + importance
├── Feature_Engineering                                 noise filtering
├── Feature_Engineering_Part2                           regimes + quantile models + score
└── Volatility_absolute_ret_Feature_Engineering_Part2   Part2 + vol features, target = |y|
```

```
data_exploration ──► train/val/test splits (80/10/10, chronological)
      │                  │
      │                  ├──► Baseline_model ───────────────┐
      │                  └──► RandomForest_XGBRegressor ──► feat_imp_combined.csv
      │                                                       │
      │                                        Feature_Engineering
      │                          (drop noise → ds_train_no_noise / ds_test_no_noise)
      │                                                       │
      │                                        Feature_Engineering_Part2
      │                    (KMeans regimes, HGB mean + quantile models, conformal, score)
      │                                                       │
      └──► vol_features.csv ─────────────► Volatility_absolute_ret_Feature_Engineering_Part2
           full_series_targets.csv           (same pipeline + vol features, target = |y|)
```

### Saved outputs (file types)

| Extension | Used for |
|---|---|
| `.parquet` | Typed tables (keeps dtypes, fast). |
| `.csv` | Small readable tables. |
| `.joblib` | Fitted objects (models, KMeans). Load only your own files. Depends on the sklearn version. |
| `.json` | Settings and metrics (`meta_part2.json`). |

---

## 4. What has been done

1. **`data_exploration`**
   - Cleans the data and builds the chronological 80/10/10 split.
   - Builds the test-format `lagged_*` columns.
   - Runs three alignment tests and uses 1-row buffers between splits.
   - Writes `train_split.csv`, `val_split.csv` and `test_split.csv`.
   - Builds `vol_features.csv` (9,048 × 8) and `full_series_targets.csv`.
   - Checks the volatility features for leakage with a perturbation test, a manual window test and a split-end test.
2. **`Baseline_model`**
   - ElasticNetCV (α = 0.0020, l1_ratio = 0.1).
   - SGDRegressor, trained online on a sliding window to mimic deployment.
   - Zero imputation plus missing-value indicators.
3. **`RandomForest_XGBRegressor`**
   - Random Forest, HistGradientBoosting and XGBoost.
   - Exports feature importance to `feat_imp_combined.csv`.
4. **`Feature_Engineering`**
   - Scores every feature with RF/XGB importance, Pearson, Spearman and mutual information (KSG estimator).
   - Tests mutual information against a block-bootstrap null that corrects for autocorrelation.
   - Crosses these methods to flag "likely noise".
   - Drops E6, I6, M15, P1, P13, P3, P4 and S9. S11 and V6 are kept with doubts.
   - Writes `ds_train_no_noise.csv`, `ds_test_no_noise.csv` and `dropped_noise_features.csv`.
5. **`Feature_Engineering_Part2`**
   - KMeans regimes on 48 features and 6,212 rows (k = 13).
   - HGB mean model.
   - HGB quantile models for q = 0.1, 0.5, 0.9.
   - Cross-validation (CV) versus test skill.
   - Conformal intervals per cluster.
   - Competition score.
   - Saves `ds_train_part2.parquet`, `ds_test_part2.parquet`, `ds_solution.csv`, `KMeans_pair.joblib`, `hgb_quantile_results.joblib`, `hgb_mean_model.joblib` and `meta_part2.json`.
6. **`Volatility_absolute_ret_Feature_Engineering_Part2` (v1)**
   - A copy of Part2 with the volatility features joined on `date_id`.
   - A switch `USE_ABS_TARGET = True` that predicts |y| instead of y.
   - A wider search range for k (2 to 24).

---

## 5. Results so far

| Notebook / model | Target | Test R² | Verdict |
|---|---|---|---|
| Naive train mean | y | −0.003 | Reference |
| ElasticNet | y | −0.068 | Worse than the mean. No linear signal. |
| Random Forest | y | 0.0009 | About the mean |
| HGB | y | −0.0112 | Worse than the mean |
| XGBoost | y | −0.0084 | Worse than the mean |
| HGB (Part2, no vol features) | y | −0.0060 | No signal |
| **HGB (+ vol features)** | **\|y\|** | **0.2096** (CV 0.1744) | **Clear signal** |

Main findings:

- **Direction is not predictable** with these features. Every model is at or below the naive baseline (RMSE ≈ 0.0105 to 0.0108).
- **Magnitude is predictable.** Spearman correlation between the volatility features and |target| is 0.34 to 0.37, strongest for `vol_ewm_094` (0.369), `vol_22` (0.358) and `vol_63` (0.341).
- This is volatility clustering (Engle 1982, Bollerslev 1986): large moves follow large moves.
- The task is best named **volatility forecasting with absolute return as proxy**. For a normal return, E|y| ≈ 0.8σ.

---

## 6. Current challenges

**Open technical issues**

1. **Skill mismatch in the volatility notebook.** The `cv_vs_test_skill` table (test skill 0.0200 / 0.0820 / 0.2085) disagrees with the `pinball_skill` output (0.0580 / 0.0004 / 0.1240) for q = 0.1 / 0.5 / 0.9. The code looks identical and the cause is not found yet.
2. **Competition score cells are invalid for |y|.** They assume a signed return. They must be replaced by volatility sizing.
3. **Stale text** from Part2 remains in the volatility notebook (summary and conclusion).
4. **Rerun risk.** Running the join cell twice creates `_right` duplicate columns. Use the idempotent join cell or Restart & Run All. Duplicates would double the KMeans weight of the vol features.

**Methodological challenges**

- **Single short test window** (about 3.6 years). The Sharpe standard error is about `√((1 + SR²/2)/Y) ≈ 0.57` (Lo 2002), so small differences are not significant.
- **Clusters may just track time periods** and not true regimes.
- **Cluster tests are in-sample** (Kruskal for location, Levene/Fligner for variance) and overstated by autocorrelation.
- **No benchmark yet.** The pure volatility baseline (HAR, Corsi 2009) has not been run, so the gain from boosting is unproven.
- **Volatility sizing is untested.** There are no results yet for the competition metric.

---

## 7. Next steps

1. Define the volatility target: |y_t| as the main target, a 5-day realized volatility as a check.
2. Run the HAR baseline (daily, weekly and monthly volatility averages).
3. Run boosting on volatility, comparing with and without clusters.
4. Size the position: `s = clip(c / σ̂, 0, 2)`, with c set on validation so the mean position is about 1. Check `vol_ratio ≤ 1.2` and a return gap near 0 (volatility-managed portfolios, Moreira & Muir 2017; debate in Cederburg et al. 2020).
5. Run an honest walk-forward evaluation with a paired test against buy and hold.
6. Optional extensions:
   - A direction tilt on top of the volatility sizing.
   - A distributional model, y = μ + σε.
   - A "director" model that chooses between strategies.
7. Cleanup: resolve the skill mismatch, update the text cells, check the unclassified features (E13, E4, M6, M9, P12, S12) and the M4 interpretation, and decide on a separate `Part3` notebook that reads the saved Part2 outputs.

---

## 8. Reproducibility

- Run order: `data_exploration` → `Baseline_model` / `RandomForest_XGBRegressor` → `Feature_Engineering` → `Feature_Engineering_Part2` → `Volatility_absolute_ret_Feature_Engineering_Part2`.
- On Kaggle, outputs go to `/kaggle/working`. The volatility notebook reads `vol_features.csv` from the `data-exploration` notebook output.
- Use **Restart & Run All** before saving a version.
- Python with pandas, polars, scikit-learn, XGBoost and joblib. Pin the sklearn version when loading `.joblib` files.

---

## 9. References

- Engle (1982). ARCH.
- Bollerslev (1986). GARCH.
- Corsi (2009). HAR model.
- Lo (2002). The statistics of Sharpe ratios.
- Moreira & Muir (2017). Volatility-managed portfolios.
- Cederburg et al. (2020). On the performance of volatility-managed portfolios.
- Kraskov, Stögbauer & Grassberger (2004). Mutual information estimator.
