import pickle
from pathlib import Path

import bm25s

from src.student.models import MinimalSource


class BM25Indexer:
    """Builds a BM25 index from a list of chunks and saves it to disk."""

    def __init__(self) -> None:
        """Initialize the indexer."""
        self.retriever = bm25s.BM25()
        self.chunks: list[MinimalSource] = []

    def build(self, chunks: list[MinimalSource], contents: list[str]) -> None:
        """Build the BM25 index from chunks and their text contents."""
        self.chunks = chunks
        tokenized = bm25s.tokenize(contents)
        self.retriever.index(tokenized)

    def save(self, directory: str) -> None:
        """Save the BM25 index and chunks list to disk."""
        save_path = Path(directory)
        save_path.mkdir(parents=True, exist_ok=True)

        self.retriever.save(str(save_path / "bm25_index"))

        with open(save_path / "chunks.pkl", "wb") as file:
            pickle.dump(self.chunks, file)
