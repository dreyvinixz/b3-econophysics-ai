# 🧠 QuantBase Project Architecture

> Auto-generated architecture mapping  
> Last update: 2026-05-26 22:22:20 (Horário de Brasília)

---

## Critérios de leitura do mapa

- **[📄 Página]**: páginas React
- **[🧩 Seção]**: blocos grandes da interface
- **[🎨 Componente]**: componentes React reutilizáveis
- **[🧱 Layout]**: estruturas de composição/layout
- **[🪝 Hook]**: hooks customizados
- **[🗂️ Dados]**: mocks, constantes e estruturas estáticas
- **[🧾 Tipagem]**: tipos, interfaces e contratos
- **[🎨 Estilo]**: CSS/SCSS
- **[🛠️ Config]**: arquivos de configuração
- **[🧪 Teste]**: testes
- **[🤖 Script]**: scripts e automações
- **[⚙️ Utilitário]**: lógica auxiliar
- **[📦 Módulo]**: módulo genérico quando não houver sinal suficiente

---

## 📁 Árvore do Projeto

```text
b3-econophysics-ai/
├── article/
│   ├── literature_review/
│   │   ├── docs/
│   │   │   └── 2302.08208v1_copy.pdf  # [📄 Arquivo] Arquivo do projeto (2302.08208v1_copy)
│   │   └── literature_search.py  # [∅ Sem conteúdo]
│   ├── main.tex  # [📄 Arquivo] Arquivo do projeto (main)
│   ├── references.bib  # [📄 Arquivo] Arquivo do projeto (references)
│   └── sections.tex  # [📄 Arquivo] Arquivo do projeto (sections)
├── config/
│   ├── assets_universe.yaml  # [∅ Sem conteúdo]
│   └── clickhouse.toml  # [📄 Arquivo] Arquivo do projeto (clickhouse)
├── notebooks/
│   ├── 01_exploration.ipynb  # [📄 Arquivo] Arquivo do projeto (01_exploration)
│   └── 02_article_figures.ipynb  # [📄 Arquivo] Arquivo do projeto (02_article_figures)
├── outputs/  [∅ Sem conteúdo]
│   ├── figures/  [∅ Sem conteúdo]
│   │   ├── preview/  [∅ Sem conteúdo]
│   │   └── vector/  [∅ Sem conteúdo]
│   ├── networks/  [∅ Sem conteúdo]
│   │   ├── gexf/  [∅ Sem conteúdo]
│   │   └── graphml/  [∅ Sem conteúdo]
│   └── tables/  [∅ Sem conteúdo]
├── scripts/
│   ├── python/
│   │   ├── 00_check_clickhouse.py  # [∅ Sem conteúdo]
│   │   ├── 01_list_liquid_assets.py  # [∅ Sem conteúdo]
│   │   ├── 02_compute_returns.py  # [∅ Sem conteúdo]
│   │   ├── 03_stylized_facts.py  # [∅ Sem conteúdo]
│   │   ├── 04_correlation_analysis.py  # [∅ Sem conteúdo]
│   │   ├── 05_rmt_pca.py  # [∅ Sem conteúdo]
│   │   ├── 06_heatmap_dendrogram.py  # [∅ Sem conteúdo]
│   │   ├── 07_mst_network.py  # [∅ Sem conteúdo]
│   │   ├── 08_pmfg_network.py  # [∅ Sem conteúdo]
│   │   ├── 09_dynamic_networks.py  # [∅ Sem conteúdo]
│   │   ├── 10_garch_baselines.py  # [∅ Sem conteúdo]
│   │   └── 11_ai_models.py  # [∅ Sem conteúdo]
│   └── r/  [∅ Sem conteúdo]
├── src/
│   ├── __init__.py  # [∅ Sem conteúdo]
│   ├── correlations.py  # [∅ Sem conteúdo]
│   ├── data_loader.py  # [∅ Sem conteúdo]
│   ├── db.py  # [∅ Sem conteúdo]
│   ├── networks.py  # [∅ Sem conteúdo]
│   ├── plotting.py  # [∅ Sem conteúdo]
│   ├── returns.py  # [∅ Sem conteúdo]
│   ├── rmt.py  # [∅ Sem conteúdo]
│   └── utils.py  # [∅ Sem conteúdo]
├── generate_project_map.py  # [🤖 Script] Script de automação (generate_project_map) | ⚡ analyze_python_ast, analyze_ts_js, build_tree, classify_css
├── requirements.txt  # [📄 Arquivo] Arquivo do projeto (requirements)
└── ROADMAP.md  # [📘 Documento] Documentação (ROADMAP)
```

---
## Observações

Este README é inferido automaticamente. A classificação melhorou bastante,
mas ainda depende da qualidade estrutural do código e dos nomes de arquivos.

Quanto mais consistentes forem:
- nomes de arquivos
- comentários de topo
- exports
- separação por responsabilidade

mais preciso o mapa fica.
