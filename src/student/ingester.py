from pathlib import Path

from src.student.chunkers.markdown_chunker import MarkdownChunker
from src.student.chunkers.python_chunker import PythonChunker
from src.student.models import MinimalSource


class Ingester:
    def __init__(self, max_chunk_size: int = 2000) -> None:
        self.py_chunker = PythonChunker(max_chunk_size)
        self.md_chunker = MarkdownChunker(max_chunk_size)

    def ingest(self, repo_path: str) -> list[MinimalSource]:
        chunks = []
        for file in Path(repo_path).rglob("*"):
            try:
                content = file.read_text(encoding="utf-8")
            except (UnicodeDecodeError, PermissionError, OSError):
                print(f"Skipping {file}")
                continue
            if file.suffix == ".md":
                chunks.extend(self.md_chunker.chunk(str(file), content))
            if file.suffix == ".py":
                chunks.extend(self.py_chunker.chunk(str(file), content))

        return chunks
