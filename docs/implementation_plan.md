# Implementation Plan — Phase 7: RMT/PCA Eigenvectors and Filtered Correlation Matrices

## Project

**Repository:** `b3-econophysics-ai`  
**Context:** Econophysics, financial dependencies, complex networks and AI applied to Brazilian B3 stocks.  
**Current phase:** Phase 7 — Random Matrix Theory and PCA.

---

## Current validated status

The project has already completed and validated the following outputs:

### Stylized facts

- Figure 1D — Stylized facts for `PETR4`, `VALE3`, `BBDC4`.
- Table 1 — Descriptive statistics for selected Brazilian stock returns.

### Static correlation structure

- Figure 2 — Pearson correlation distribution for the `core_historical` universe.
- Figure 3 — Within-sector vs between-sector correlation distribution.
- `core_historical` universe:
  - `N = 58` assets.
  - `N_pairs = 1653`.
  - Mean pairwise correlation approximately `0.24`.
  - Median pairwise correlation approximately `0.23`.

### Sectoral dependency result

Within-sector correlations are stronger than between-sector correlations:

```text
Within-sector:
  n_pairs = 275
  mean    = 0.3433
  median  = 0.3082

Between-sector:
  n_pairs = 1378
  mean    = 0.2249
  median  = 0.2076

Mann-Whitney U:
  H1: within-sector > between-sector
  p-value = 1.33389e-24
```

### Dynamic correlation structure

* Figure 4 — Rolling average market correlation.
* Figure 5 — Dynamic pairwise correlations.

Main validated conclusion:

```text
Brazilian market synchronization is time-varying and increases sharply during systemic stress episodes, especially around the 2008 global financial crisis and the 2020 COVID shock.
```

---

# Phase 7 current status

The initial RMT script has been implemented and executed:

```text
scripts/python/07_rmt_pca.py
```

The RMT analysis uses a complete-case matrix from:

```text
outputs/tables/core_historical_returns_wide_1998_2025.csv
```

Important methodological rule:

```text
For RMT/PCA, do not use the pairwise correlation matrix.
Use a common-sample return matrix after complete-case filtering.
```

---

## Current RMT results

After complete-case filtering:

```text
T = 1527 trading days
N = 58 assets
Q = T / N ≈ 26.33
```

Marčenko-Pastur bounds:

```text
lambda_max ≈ 1.4278
```

Dominant eigenvalue:

```text
lambda_1 ≈ 21.6505
```

Number of eigenvalues above the Marčenko-Pastur upper bound:

```text
5 eigenvalues above lambda_max
```

Interpretation so far:

```text
lambda_1 captures a strong Market Mode.
lambda_2 to lambda_5 likely encode sectoral or group-level structures.
```

The first eigenvalue explains approximately:

```text
lambda_1 / N = 21.6505 / 58 ≈ 37.3%
```

of the total correlation-matrix variance.

Generated files:

```text
outputs/tables/rmt_summary_core_historical_1998_2025.csv
outputs/tables/rmt_eigenvalues_core_historical_1998_2025.csv
outputs/figures/vector/figure_6_rmt_eigenvalue_spectrum.pdf
outputs/figures/preview/figure_6_rmt_eigenvalue_spectrum.png
```

---

# Objective of this implementation plan

The next phase should not immediately perform PCA filtering.

Before reconstructing filtered matrices, we must interpret the eigenvectors associated with the eigenvalues above the Marčenko-Pastur noise threshold.

The next implementation goal is:

```text
Interpret the top eigenvectors and identify whether they represent market-wide or sectoral structures.
```

---

# Phase 7.1 — Eigenvector loading analysis

## Script

Extend the existing script:

```text
scripts/python/07_rmt_pca.py
```

or create a dedicated script:

```text
scripts/python/07b_rmt_eigenvectors.py
```

Recommendation:

```text
Keep everything in scripts/python/07_rmt_pca.py for now.
```

If the file becomes too large, split later.

---

## Inputs

```text
outputs/tables/core_historical_returns_wide_1998_2025.csv
outputs/tables/assets_sector_map.csv
outputs/tables/rmt_eigenvalues_core_historical_1998_2025.csv
```

The script should recompute the correlation matrix and eigenvectors from the same complete-case matrix used in the RMT spectrum.

---

## Methodology

1. Load `core_historical_returns_wide_1998_2025.csv`.
2. Convert `date` to datetime.
3. Drop the `date` column.
4. Apply `dropna()` to obtain a complete-case return matrix.
5. Standardize returns if necessary.
6. Compute the correlation matrix.
7. Compute eigenvalues and eigenvectors.
8. Sort eigenvalues in descending order.
9. Identify eigenvalues above `lambda_max`.
10. For each significant eigenvector, compute:

* loading;
* absolute loading;
* signed rank;
* sector;
* subsector;
* company name.

---

## Important eigenvector convention

Because eigenvector signs are arbitrary, the sign of each eigenvector may flip between executions.

To stabilize interpretation:

```text
For eigenvector 1, enforce the sign so that the sum of loadings is positive.
For other eigenvectors, signs are arbitrary and should be interpreted by relative groups, not absolute direction alone.
```

Implementation idea:

```python
if eigenvector.sum() < 0:
    eigenvector = -eigenvector
```

For eigenvectors 2–5, keep signs but interpret both positive and negative sides.

---

## Expected outputs

### 1. Top eigenvector loadings

```text
outputs/tables/rmt_top_eigenvector_loadings_core_historical_1998_2025.csv
```

Columns:

```text
eigen_rank
eigenvalue
above_mp_lambda_max
symbol
company_name
sector
subsector
segment
loading
abs_loading
loading_rank
```

For each relevant eigenvector, include all 58 assets, not only the top assets. Sorting can be done later.

---

### 2. Eigenvector sector summary

```text
outputs/tables/rmt_eigenvector_sector_summary_core_historical_1998_2025.csv
```

Columns:

```text
eigen_rank
eigenvalue
sector
n_assets
sum_abs_loading
mean_abs_loading
max_abs_loading
dominant_symbol
dominant_loading
```

Purpose:

```text
Quantify which sectors dominate each significant eigenvector.
```

---

### 3. Figure 7 — top eigenvectors

```text
outputs/figures/vector/figure_7_rmt_top_eigenvectors.pdf
outputs/figures/preview/figure_7_rmt_top_eigenvectors.png
```

Optional:

```text
outputs/figures/vector/figure_7_rmt_top_eigenvectors.svg
```

---

# Figure 7 design

## Purpose

Visualize the loadings of the significant eigenvectors above the Marčenko-Pastur noise threshold.

## Layout

Recommended layout:

```text
5 panels vertically stacked or arranged in a 2x3 grid.
Each panel corresponds to one eigenvector:
  Eigenvector 1 — Market Mode
  Eigenvector 2
  Eigenvector 3
  Eigenvector 4
  Eigenvector 5
```

## Plot type

Use bar plots of loadings by symbol.

Recommended:

```text
x-axis: symbols
y-axis: eigenvector loading
bar color: sector or neutral gray
```

If coloring by sector becomes visually noisy, use neutral bars and annotate sector summaries separately.

## Style

Use the same publication style as previous figures:

```text
serif font
thin axes
no large global title
subtle grid or no grid
PDF/SVG export
```

---

# Expected interpretation

## Eigenvector 1

Expected behavior:

```text
Most loadings should share the same sign.
The eigenvector should be broadly distributed across assets.
```

Interpretation:

```text
Market Mode / systemic Brazilian equity-market factor.
```

Potential article statement:

```text
The first eigenvector is broadly distributed across assets and captures a market-wide mode, consistent with the large eigenvalue far above the Marčenko-Pastur noise band.
```

---

## Eigenvectors 2–5

Expected behavior:

```text
Loadings should concentrate in economically interpretable sectors or groups.
```

Possible structures:

```text
Basic Materials / commodities
Financials
Utilities
Oil Gas and Biofuels
Industrials
```

Interpretation:

```text
Sectoral or group-level factors that survive the random-matrix noise threshold.
```

Potential article statement:

```text
The remaining eigenvectors outside the Marčenko-Pastur interval encode economically interpretable sectoral structures, linking the spectral analysis to the observed within-sector correlation patterns.
```

---

# Phase 7.2 — Filtered correlation matrix reconstruction

Do not implement this before validating Phase 7.1.

After eigenvector interpretation, reconstruct filtered matrices.

## Future outputs

```text
outputs/tables/correlation_market_mode_core_historical_1998_2025.csv
outputs/tables/correlation_group_mode_core_historical_1998_2025.csv
outputs/tables/correlation_filtered_core_historical_1998_2025.csv
outputs/tables/correlation_cleaned_without_noise_core_historical_1998_2025.csv
```

---

## Matrix definitions

Let:

```text
C = empirical correlation matrix
lambda_k = eigenvalue k
u_k = eigenvector k
lambda_max = Marčenko-Pastur upper bound
```

### Market mode matrix

```text
C_market = lambda_1 * u_1 * u_1^T
```

Purpose:

```text
Capture the common market-wide component.
```

### Group/sector matrix

```text
C_group = sum(lambda_k * u_k * u_k^T) for k = 2,...,K
```

where:

```text
lambda_k > lambda_max
```

Purpose:

```text
Capture sectoral and group-level structures excluding the dominant market mode.
```

### Filtered matrix

```text
C_filtered = C_market + C_group
```

Purpose:

```text
Keep all statistically significant components above the random-matrix noise band.
```

### Sector-filtered matrix

```text
C_sector_filtered = C_group
```

Purpose:

```text
Use for network analysis when the market mode is too dominant.
```

---

# Phase 7.3 — Network preparation after RMT

After filtered matrices are validated, prepare them for networks.

Future use:

```text
Heatmap and dendrogram
MST
PMFG/TMFG
community detection
```

Recommended strategy:

```text
Build networks from both:
1. original correlation matrix
2. market-mode-removed / group-mode matrix
```

Reason:

```text
The original matrix shows the raw market structure.
The market-mode-removed matrix reveals more sector-specific information.
```

---

# Validation checklist for Phase 7.1

After running the updated RMT/PCA script, send the following files for validation:

```text
outputs/tables/rmt_summary_core_historical_1998_2025.csv
outputs/tables/rmt_eigenvalues_core_historical_1998_2025.csv
outputs/tables/rmt_top_eigenvector_loadings_core_historical_1998_2025.csv
outputs/tables/rmt_eigenvector_sector_summary_core_historical_1998_2025.csv
outputs/figures/preview/figure_7_rmt_top_eigenvectors.png
```

Also send the terminal output containing:

```text
N assets
T observations
Q = T / N
lambda_min
lambda_max
largest eigenvalue
number of eigenvalues above lambda_max
top positive and negative loadings for eigenvectors 1–5
sector concentration by eigenvector
```

---

# Validation commands

## Check eigenvalue summary

```powershell
python -c "import pandas as pd; print(pd.read_csv('outputs/tables/rmt_summary_core_historical_1998_2025.csv').to_string(index=False)); print(pd.read_csv('outputs/tables/rmt_eigenvalues_core_historical_1998_2025.csv').head(10).to_string(index=False))"
```

## Check top eigenvector loadings

```powershell
python -c "import pandas as pd; df=pd.read_csv('outputs/tables/rmt_top_eigenvector_loadings_core_historical_1998_2025.csv'); print(df.columns.tolist()); print('\nEIGENVECTOR 1 TOP ABS'); print(df[df['eigen_rank']==1].sort_values('abs_loading', ascending=False).head(20).to_string(index=False)); print('\nEIGENVECTOR 2 TOP POSITIVE'); print(df[df['eigen_rank']==2].sort_values('loading', ascending=False).head(15).to_string(index=False)); print('\nEIGENVECTOR 2 TOP NEGATIVE'); print(df[df['eigen_rank']==2].sort_values('loading').head(15).to_string(index=False))"
```

## Check sector concentration

```powershell
python -c "import pandas as pd; df=pd.read_csv('outputs/tables/rmt_eigenvector_sector_summary_core_historical_1998_2025.csv'); print(df.sort_values(['eigen_rank','sum_abs_loading'], ascending=[True,False]).to_string(index=False))"
```

---

# ROADMAP updates after successful Phase 7.1 validation

After eigenvector loading analysis is implemented and validated:

```markdown
## Phase 7 — Random Matrix Theory and PCA

- [x] Compute eigenvalues of correlation matrix
- [x] Compute Marcenko-Pastur bounds
- [x] Identify market mode
- [x] Identify eigenvalues outside the random-matrix noise band
- [x] Analyze eigenvectors above the noise band
- [x] Generate top-eigenvector loading figure
- [ ] Reconstruct market-mode matrix
- [ ] Reconstruct group/sector-mode matrix
- [ ] Reconstruct filtered correlation matrix
- [ ] Compare original and filtered matrices
```

---

# Expected scientific contribution of Phase 7

This phase should support the following claims:

1. The Brazilian equity market has a strong market-wide mode.
2. The largest eigenvalue is far above the random-matrix noise threshold.
3. Several additional eigenvalues survive beyond the Marčenko-Pastur upper bound.
4. The first eigenvector captures broad market synchronization.
5. The remaining significant eigenvectors encode sectoral or group-level structures.
6. RMT provides a principled way to separate structured information from random noise in B3 stock correlations.
7. Filtered correlation matrices can be used to build more interpretable financial networks.

---

# Next action

Implement the eigenvector analysis inside:

```text
scripts/python/07_rmt_pca.py
```

Generate:

```text
Figure 7 — RMT top eigenvector loadings
```

Then validate the sectoral interpretation before reconstructing filtered matrices.
