import ast
import re
from pathlib import Path
from datetime import datetime, timezone, timedelta

# =========================
# CONFIGURAÇÕES
# =========================
ROOT_DIR = Path(".")
OUTPUT_FILE = "README.md"

IGNORE_DIRS = {
    ".git", "__pycache__", "node_modules", ".venv",
    "venv", "dist", "build", ".idea", ".vscode",
    "coverage", "sites", ".next", ".turbo"
}

TEXT_FILE_EXTENSIONS = {
    ".py", ".ts", ".tsx", ".js", ".jsx", ".json",
    ".css", ".scss", ".md", ".yml", ".yaml", ".html"
}

# =========================
# UTILITÁRIOS
# =========================

def safe_read_text(file_path: Path, max_chars: int | None = None) -> str:
    try:
        content = file_path.read_text(encoding="utf-8")
        if max_chars is not None:
            return content[:max_chars]
        return content
    except Exception:
        return ""

def normalize_spaces(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()

def first_meaningful_line(text: str) -> str:
    for line in text.splitlines():
        stripped = line.strip()
        if stripped:
            return stripped
    return ""

def is_binary_like(file_path: Path) -> bool:
    return file_path.suffix.lower() not in TEXT_FILE_EXTENSIONS

# =========================
# DESCRIÇÕES
# =========================

def get_python_description(file_path: Path) -> str:
    try:
        content = safe_read_text(file_path)
        if not content:
            return ""

        tree = ast.parse(content)
        doc = ast.get_docstring(tree)
        if doc:
            return normalize_spaces(doc.splitlines()[0])

        for line in content.splitlines():
            stripped = line.strip()
            if stripped.startswith("#"):
                return stripped.lstrip("#").strip()
            if stripped:
                break
    except Exception:
        pass
    return ""

def get_js_ts_description(file_path: Path) -> str:
    try:
        content = safe_read_text(file_path, max_chars=2500)
        if not content:
            return ""

        jsdoc = re.search(r"/\*\*([\s\S]*?)\*/", content)
        if jsdoc:
            cleaned = re.sub(r"^\s*\*\s?", "", jsdoc.group(1), flags=re.MULTILINE)
            line = first_meaningful_line(cleaned)
            if line:
                return normalize_spaces(line)

        block = re.search(r"/\*([\s\S]*?)\*/", content)
        if block:
            cleaned = block.group(1)
            line = first_meaningful_line(cleaned)
            if line:
                return normalize_spaces(line)

        for line in content.splitlines():
            stripped = line.strip()
            if stripped.startswith("//"):
                return stripped.lstrip("/").strip()
            if stripped:
                break
    except Exception:
        pass
    return ""

def get_css_description(file_path: Path) -> str:
    content = safe_read_text(file_path, max_chars=1500)
    if not content:
        return ""
    match = re.search(r"/\*([\s\S]*?)\*/", content)
    if match:
        line = first_meaningful_line(match.group(1))
        if line:
            return normalize_spaces(line)
    return ""

# =========================
# ANÁLISE TÉCNICA
# =========================

def analyze_python_ast(file_path: Path) -> list[str]:
    defs = set()
    try:
        tree = ast.parse(safe_read_text(file_path))
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                if not node.name.startswith("_"):
                    defs.add(node.name)
    except Exception:
        pass
    return sorted(defs)[:4]

def analyze_ts_js(file_path: Path) -> list[str]:
    defs = set()
    content = safe_read_text(file_path)
    if not content:
        return []

    patterns = [
        r"export\s+default\s+function\s+([A-Z]\w+)",
        r"export\s+function\s+([A-Z]\w+|\w+)",
        r"export\s+const\s+([A-Z]\w+|\w+)",
        r"export\s+class\s+([A-Z]\w+|\w+)",
        r"export\s+interface\s+([A-Z]\w+|\w+)",
        r"export\s+type\s+([A-Z]\w+|\w+)",
    ]

    for pattern in patterns:
        defs.update(re.findall(pattern, content))

    return sorted(defs)[:4]

def extract_imports_ts_js(file_path: Path) -> list[str]:
    content = safe_read_text(file_path, max_chars=6000)
    if not content:
        return []

    imports = re.findall(r'import\s+.*?\s+from\s+[\'"](.+?)[\'"]', content)
    return imports[:8]

def extract_css_selectors(file_path: Path) -> list[str]:
    content = safe_read_text(file_path, max_chars=5000)
    if not content:
        return []

    selectors = re.findall(r"([.#][a-zA-Z0-9_-]+)\s*\{", content)
    unique = []
    for item in selectors:
        if item not in unique:
            unique.append(item)
    return unique[:6]

# =========================
# CLASSIFICAÇÃO SEMÂNTICA
# =========================

def classify_python(file_path: Path, content: str) -> tuple[str, str]:
    name = file_path.name.lower()
    parts = [p.lower() for p in file_path.parts]

    if "test" in name or any("test" in p for p in parts):
        return "[🧪 Teste]", "Teste automatizado"
    if name in {"conftest.py"}:
        return "[🧪 Teste]", "Configuração de testes"
    if "script" in parts or name.startswith("generate_") or name.startswith("build_"):
        return "[🤖 Script]", "Script de automação"
    if "api" in parts or "router" in name or "route" in name:
        return "[🌐 API]", "Módulo de API"
    if "config" in name or "settings" in name:
        return "[🛠️ Config]", "Configuração Python"

    try:
        tree = ast.parse(content)
        classes = [n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]
        functions = [n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
        imports = [n for n in ast.walk(tree) if isinstance(n, (ast.Import, ast.ImportFrom))]
        has_main = any(
            isinstance(n, ast.If)
            and isinstance(n.test, ast.Compare)
            for n in ast.walk(tree)
        )

        if classes and functions:
            return "[🧠 Lógica]", "Módulo de lógica e estruturas"
        if classes:
            return "[🧱 Classe]", "Classe ou objeto principal"
        if has_main and not classes:
            return "[🤖 Script]", "Script executável"
        if imports and functions:
            return "[⚙️ Utilitário]", "Funções utilitárias"
    except Exception:
        pass

    return "[📦 Módulo]", "Módulo Python"

def classify_ts_js(file_path: Path, content: str) -> tuple[str, str]:
    name = file_path.name
    name_lower = name.lower()
    parts = [p.lower() for p in file_path.parts]
    ext = file_path.suffix.lower()

    if "test" in name_lower or ".spec." in name_lower or ".test." in name_lower:
        return "[🧪 Teste]", "Teste frontend"
    if name_lower in {"vite.config.ts", "vitest.config.ts", "tailwind.config.ts", "postcss.config.js"}:
        return "[🛠️ Config]", "Configuração do projeto"
    if ext in {".json", ".yml", ".yaml"}:
        return "[🛠️ Config]", "Arquivo de configuração"

    imports = extract_imports_ts_js(file_path)

    is_react_component = bool(
        re.search(r"export\s+default\s+function\s+[A-Z]", content)
        or re.search(r"const\s+[A-Z]\w*\s*=\s*\(", content)
        or re.search(r"React\.memo\(", content)
        or re.search(r"return\s*\(\s*<", content, re.DOTALL)
    )

    has_hooks = bool(re.search(r"\buse(State|Effect|Memo|Ref|Callback|Reducer)\b", content))
    has_types = bool(re.search(r"export\s+(type|interface)\s+\w+", content))
    has_route = "router" in name_lower or "routes" in name_lower or "react-router" in content
    has_styles_import = any(
        imp.endswith(".css") or imp.endswith(".scss") for imp in imports
    )
    is_hook_file = name_lower.startswith("use") and ext in {".ts", ".tsx"}
    is_data_file = bool(re.search(r"export\s+const\s+\w+\s*=\s*\[", content)) and not is_react_component
    is_type_file = name_lower.endswith(".types.ts") or name_lower == "types.ts"
    is_style_helper = ext in {".css", ".scss"}

    if is_style_helper:
        return "[🎨 Estilo]", "Folha de estilo"
    if is_hook_file:
        return "[🪝 Hook]", "Hook customizado"
    if has_route:
        return "[🧭 Rota]", "Definição de rotas"
    if is_type_file or (has_types and not is_react_component and not has_hooks):
        return "[🧾 Tipagem]", "Tipos e contratos"
    if is_data_file or "data" in name_lower:
        return "[🗂️ Dados]", "Dados estáticos ou mocks"
    if is_react_component:
        if "page" in name_lower:
            return "[📄 Página]", "Página React"
        if "section" in name_lower:
            return "[🧩 Seção]", "Seção da interface"
        if "layout" in name_lower or "shell" in name_lower:
            return "[🧱 Layout]", "Estrutura de layout"
        if has_hooks and has_styles_import:
            return "[🎨 Componente]", "Componente React com estado/efeitos"
        return "[🎨 Componente]", "Componente React"
    if has_hooks:
        return "[⚙️ Utilitário]", "Lógica de frontend"
    return "[📦 Módulo]", "Módulo TypeScript/JavaScript"

def classify_css(file_path: Path, content: str) -> tuple[str, str]:
    name_lower = file_path.name.lower()

    if "tailwind" in name_lower:
        return "[🛠️ Config]", "Configuração de estilo"
    if "globals" in name_lower or "global" in name_lower:
        return "[🎨 Estilo]", "Estilos globais"

    selectors = extract_css_selectors(file_path)
    if any(sel.startswith(".qb-market") for sel in selectors):
        return "[🎨 Estilo]", "Estilos da seção Market"
    if selectors:
        return "[🎨 Estilo]", "Folha de estilo de interface"
    return "[🎨 Estilo]", "Folha de estilo"

def classify_file(file_path: Path) -> tuple[str, str]:
    ext = file_path.suffix.lower()
    content = safe_read_text(file_path, max_chars=12000)

    if ext == ".py":
        return classify_python(file_path, content)
    if ext in {".ts", ".tsx", ".js", ".jsx", ".json", ".yml", ".yaml"}:
        return classify_ts_js(file_path, content)
    if ext in {".css", ".scss"}:
        return classify_css(file_path, content)
    if file_path.name.lower() == "dockerfile":
        return "[🛠️ Config]", "Containerização"
    if ext in {".md"}:
        return "[📘 Documento]", "Documentação"
    return "[📄 Arquivo]", "Arquivo do projeto"

# =========================
# DESCRIÇÃO SEMÂNTICA
# =========================

def infer_description(file_path: Path) -> str:
    ext = file_path.suffix.lower()

    if ext == ".py":
        description = get_python_description(file_path)
        if description:
            return description
    elif ext in {".ts", ".tsx", ".js", ".jsx"}:
        description = get_js_ts_description(file_path)
        if description:
            return description
    elif ext in {".css", ".scss"}:
        description = get_css_description(file_path)
        if description:
            return description

    name = file_path.stem
    category, fallback = classify_file(file_path)
    return f"{fallback} ({name})"

def get_tech_context(file_path: Path) -> str:
    ext = file_path.suffix.lower()

    if ext == ".py":
        defs = analyze_python_ast(file_path)
    elif ext in {".ts", ".tsx", ".js", ".jsx"}:
        defs = analyze_ts_js(file_path)
    elif ext in {".css", ".scss"}:
        defs = extract_css_selectors(file_path)
    else:
        defs = []

    if defs:
        return f"| ⚡ {', '.join(defs[:4])}"
    return ""

def get_file_details(file_path: Path) -> str:
    category, inferred_role = classify_file(file_path)
    description = infer_description(file_path)
    tech_context = get_tech_context(file_path)

    if description == inferred_role:
        base = f"{category} {description}"
    else:
        base = f"{category} {description}"

    return f"{base} {tech_context}".strip()

# =========================
# DETECÇÃO DE VAZIO
# =========================

def is_effectively_empty(directory: Path) -> bool:
    try:
        entries = [
            e for e in directory.iterdir()
            if e.name not in IGNORE_DIRS and not e.name.startswith(".")
        ]
    except PermissionError:
        return True

    if not entries:
        return True

    for entry in entries:
        if entry.is_file():
            if entry.name != OUTPUT_FILE:
                return False
        elif entry.is_dir():
            if not is_effectively_empty(entry):
                return False

    return True

def is_file_empty(file_path: Path) -> bool:
    if is_binary_like(file_path):
        return False

    try:
        if file_path.stat().st_size == 0:
            return True
        return safe_read_text(file_path).strip() == ""
    except Exception:
        return False

# =========================
# ÁRVORE
# =========================

def build_tree(directory: Path, prefix: str = "") -> list[str]:
    tree_lines: list[str] = []
    try:
        entries = sorted(
            [
                e for e in directory.iterdir()
                if e.name not in IGNORE_DIRS and not e.name.startswith(".")
            ],
            key=lambda x: (x.is_file(), x.name.lower())
        )
    except PermissionError:
        return []

    visible_entries = []
    for entry in entries:
        if entry.is_dir():
            visible_entries.append(entry)
        elif entry.name != OUTPUT_FILE:
            visible_entries.append(entry)

    for i, entry in enumerate(visible_entries):
        connector = "└── " if i == len(visible_entries) - 1 else "├── "

        if entry.is_dir():
            label = f"{entry.name}/"
            if is_effectively_empty(entry):
                label += "  [∅ Sem conteúdo]"

            tree_lines.append(f"{prefix}{connector}{label}")
            tree_lines.extend(
                build_tree(
                    entry,
                    prefix + ("    " if i == len(visible_entries) - 1 else "│   ")
                )
            )
        else:
            if is_file_empty(entry):
                tree_lines.append(f"{prefix}{connector}{entry.name}  # [∅ Sem conteúdo]")
            else:
                tree_lines.append(f"{prefix}{connector}{entry.name}  # {get_file_details(entry)}")

    return tree_lines

# =========================
# README FRONTEND
# =========================

def generate_frontend_intro(now: str) -> str:
    return f"""# 🧠 QuantBase Project Architecture

> Auto-generated architecture mapping  
> Last update: {now} (Horário de Brasília)

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
{ROOT_DIR.resolve().name}/
"""

def generate_readme() -> None:
    fuso_br = timezone(timedelta(hours=-3))
    now = datetime.now(fuso_br).strftime("%Y-%m-%d %H:%M:%S")
    tree = build_tree(ROOT_DIR)

    content = generate_frontend_intro(now)
    content += "\n".join(tree)
    content += "\n```\n"

    content += """
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
"""

    Path(OUTPUT_FILE).write_text(content, encoding="utf-8")
    print("✅ README atualizado com sucesso!")

if __name__ == "__main__":
    generate_readme()