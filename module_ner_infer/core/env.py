from pathlib import Path


def find_project_root(start_path: Path, marker: str = ".git") -> Path:
    current_path = start_path.resolve()
    while current_path != current_path.parent:
        if (current_path / marker).exists():
            return current_path
        current_path = current_path.parent
    raise FileNotFoundError(f"Project root with marker '{marker}' not found.")


# Environment
PROJECT_ROOT_DIR = find_project_root(Path(__file__))
CURRENT_ENV = "dev"
DIC_PATH = "/opt/homebrew/lib/mecab/dic/mecab-ko-dic"


# APIs
ORIGINS_PROD = [
    "http://localhost",
    "http://localhost:8000",
]
