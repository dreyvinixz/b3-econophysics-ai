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
- [ ] Create README.md
- [ ] Fill requirements.txt
- [ ] Fill config/clickhouse.toml
- [ ] Configure Python import path
- [ ] Test local execution from project root
- [ ] Initialize Git repository
- [ ] Create .gitignore

---

## Phase 1 — ClickHouse connection

- [ ] Start QuantBase backend Docker stack
- [ ] Confirm ClickHouse container is running
- [ ] Confirm port 8123 is exposed
- [ ] Test `curl http://localhost:8123/ping`
- [ ] Implement `src/db.py`
- [ ] Implement `scripts/python/00_check_clickhouse.py`
- [ ] Run `SELECT 1`
- [ ] Run table summary for `quantbase.candles_1d`

---

## Phase 2 — Data audit

- [ ] Describe `quantbase.candles_1d`
- [ ] Confirm available columns
- [ ] Confirm first and last date
- [ ] Count total rows
- [ ] Count unique symbols
- [ ] Check duplicated rows by `symbol, date`
- [ ] Check null values in `adj_close`
- [ ] Check null values in `financial_volume`
- [ ] Check invalid prices
- [ ] Check invalid volume
- [ ] Inspect `factor_status`
- [ ] Confirm that `adj_close` is the correct price column for returns

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

---

## Phase 4 — Returns

- [x] Load adjusted daily prices from ClickHouse
- [x] Compute log returns using `adj_close`
- [ ] Validate extreme returns
- [x] Create long returns DataFrame
- [x] Create wide returns matrix
- [x] Generate summary statistics by asset

---

## Phase 5 — Stylized facts

- [ ] Generate normalized price plot
- [ ] Generate daily returns plot
- [ ] Generate CCDF of absolute returns
- [ ] Generate ACF of absolute returns
- [ ] Export Figure 1 as PDF
- [ ] Export Figure 1 as SVG
- [ ] Export Figure 1 preview as PNG

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