# ROADMAP — B3 Econophysics AI

## Status legend

- [ ] Pending
- [~] In progress
- [x] Done
- [!] Attention / methodological risk
- [-] Deferred

---

## Phase 0 — Project setup

- [x] Create project structure
- [x] Create README.md
- [x] Fill requirements.txt
- [x] Fill config/clickhouse.toml
- [x] Configure Python import path
- [x] Test local execution from project root
- [x] Initialize Git repository
- [x] Create .gitignore
- [x] Connect repository to GitHub `dreyvinixz/b3-econophysics-ai`

---

## Phase 1 — ClickHouse connection

- [x] Start QuantBase backend Docker stack
- [x] Confirm ClickHouse container is running
- [x] Confirm port 8123 is exposed
- [x] Test ClickHouse connection through Python client
- [x] Implement `src/db.py`
- [x] Implement `scripts/python/00_check_clickhouse.py`
- [x] Run `SELECT 1`
- [x] Run table summary for `quantbase.candles_1d`

---

## Phase 2 — Data audit

- [x] Describe `quantbase.candles_1d`
- [x] Confirm available columns
- [x] Confirm first and last date
- [x] Count total rows
- [x] Count unique symbols
- [x] Check duplicated rows by `symbol, date`
- [x] Check null values in `adj_close`
- [x] Check null values in `financial_volume`
- [x] Check invalid prices
- [x] Check invalid volume
- [~] Inspect `factor_status`
- [x] Confirm that `adj_close` is the correct price column for returns
- [x] Validate generated return table dimensions

---

## Phase 3 — Liquid universe selection

- [x] Generate clean top assets excluding BDRs
- [x] Build `core_historical` universe with `n_days >= 5000`
- [x] Build `modern_liquid` candidate universe with `n_days >= 1000`
- [x] Decide to keep Brazilian units
- [x] Decide to exclude BDRs using `specification NOT LIKE '%DR%'`
- [x] Fill `config/assets_universe.yaml`
- [x] Implement `scripts/python/01_list_liquid_assets.py`
- [x] Export universe tables to `outputs/tables`
- [x] Define `demo_assets`: PETR4, VALE3, BBDC4
- [x] Define `modern_liquid_top_50`

---

## Phase 4 — Returns

- [x] Load adjusted daily prices from ClickHouse
- [x] Compute log returns using `adj_close`
- [x] Create long returns DataFrame
- [x] Create wide returns matrix
- [x] Generate summary statistics by asset
- [x] Validate extreme returns
- [x] Approve `demo_assets` for first stylized-facts figure
- [!] Investigate extreme adjusted-return events in modern universe
- [!] Avoid full-period complete-case dropna for `modern_liquid_top_50`

---

## Phase 4.1 — Return quality control

- [ ] Create extreme-return flags
- [ ] Inspect `abs(log_return) > 0.50`
- [ ] Inspect `abs(log_return) > 1.00`
- [ ] Investigate KLBN11 December 2013 adjustment sequence
- [ ] Decide treatment for extreme corporate-action-like returns
- [~] Define modern analysis window, likely 2018-2025
- [ ] Export `extreme_returns_modern_top_50.csv`
- [ ] Export `critical_returns_modern_top_50.csv`

---

## Phase 5 — Stylized facts

- [x] Generate normalized price plot
- [x] Generate daily returns plot
- [x] Generate CCDF of absolute returns
- [x] Generate ACF of absolute returns
- [x] Add journal-style return offsets
- [x] Extend absolute-return ACF to 300 lags
- [x] Add pooled/all ACF curve
- [x] Add power-law reference line for CCDF tail
- [x] Export Figure 1 as PDF
- [x] Export Figure 1 as SVG
- [x] Export Figure 1 preview as PNG
- [x] Export Figure 1A, 2006-2021 comparison window
- [x] Export Figure 1B, 1998-2025 full Brazilian window
- [x] Export Figure 1C, journal-style 2006-2021 main candidate
- [x] Visually inspect Figure 1 PNG
- [x] Visually inspect Figure 1C journal-style PNG
- [x] Commit and push Figure 1 artifacts

---

## Phase 6 — Correlation structure

- [ ] Compute Pearson correlation matrix
- [ ] Compute Spearman correlation matrix
- [ ] Compute within-sector correlations
- [ ] Compute between-sector correlations
- [ ] Generate correlation histogram
- [ ] Compute rolling correlations
- [ ] Compute EWMA correlations

---

## Phase 7 — Random Matrix Theory and PCA

- [ ] Compute eigenvalues of correlation matrix
- [ ] Compute Marcenko-Pastur bounds
- [ ] Identify market mode
- [ ] Analyze eigenvectors
- [ ] Run PCA
- [ ] Compare PCA factors with market/sector structure

---

## Phase 8 — Heatmap and dendrogram

- [ ] Compute Mantegna distance matrix
- [ ] Run hierarchical clustering
- [ ] Reorder correlation matrix
- [ ] Generate ordered heatmap
- [ ] Generate dendrogram
- [ ] Export vector figure

---

## Phase 9 — Financial networks

- [ ] Build MST
- [ ] Compute centrality metrics
- [ ] Export MST as GraphML
- [ ] Export MST as GEXF
- [ ] Build PMFG or TMFG
- [ ] Detect communities
- [ ] Compare communities with B3 sectors

---

## Phase 10 — Dynamic networks

- [ ] Define rolling windows
- [ ] Compute dynamic correlation matrices
- [ ] Compute largest eigenvalue over time
- [ ] Compute average correlation over time
- [ ] Compute network metrics over time
- [ ] Compare crisis and non-crisis periods

---

## Phase 11 — Econometric benchmarks

- [ ] Estimate GARCH(1,1) for selected assets
- [ ] Compare EWMA and GARCH volatility
- [ ] Test DCC-GARCH on selected pairs/subsample
- [ ] Evaluate Granger causality network
- [ ] Evaluate variance decomposition / spillover network

---

## Phase 12 — AI models

- [ ] Define predictive task
- [ ] Implement PCA baseline
- [ ] Implement Autoencoder
- [ ] Compare Autoencoder latent factors with PCA
- [ ] Choose LSTM, Transformer, or GNN extension
- [ ] Train first AI model
- [ ] Compare AI model with classical baseline

---

## Phase 13 — Article writing

- [ ] Write abstract
- [ ] Write introduction
- [ ] Write related work
- [ ] Write data section
- [ ] Write methodology
- [ ] Write results
- [ ] Write discussion
- [ ] Write conclusion
- [ ] Format references
- [ ] Compile LaTeX
