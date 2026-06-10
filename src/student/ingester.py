from pathlib import Path

from student.chunkers.markdown_chunker import MarkdownChunker
from student.chunkers.python_chunker import PythonChunker
from student.models import MinimalSource


class Ingester:
    """Reads all .py and .md files from a repo and splits them into chunks."""

    def __init__(self, max_chunk_size: int = 2000) -> None:
        """Initialize the ingester with chunkers."""
        self.py_chunker = PythonChunker(max_chunk_size)
        self.md_chunker = MarkdownChunker(max_chunk_size)

    def ingest(self, repo_path: str) -> tuple[list[MinimalSource], list[str]]:
        """Read all files and return chunks with their text contents.
        """
        chunks: list[MinimalSource] = []
        contents: list[str] = []

        for file in Path(repo_path).rglob("*"):
            try:
                content = file.read_text(encoding="utf-8")
            except (UnicodeDecodeError, PermissionError, OSError):
                continue

            if file.suffix == ".md":
                new_chunks = self.md_chunker.chunk(str(file), content)
            elif file.suffix == ".py":
                new_chunks = self.py_chunker.chunk(str(file), content)
            elif file.suffix in (".rst", ".txt"):
                new_chunks = self.md_chunker.chunk(str(file), content)
            else:
                continue

            chunks.extend(new_chunks)
            for ch in new_chunks:
                contents.append(
                    content[ch.first_character_index:ch.last_character_index]
                )

        return (chunks, contents)
