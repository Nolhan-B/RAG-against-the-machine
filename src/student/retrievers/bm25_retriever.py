import pickle
from pathlib import Path

import bm25s

from src.student.models import MinimalSource


class BM25Retriever:
    """Loads a BM25 index from disk and retrieves relevant chunks for a query."""

    def __init__(self, index_directory: str) -> None:
        """Load the BM25 index and chunks from disk.
        """
        index_path = Path(index_directory)

        try:
            self.retriever = bm25s.BM25.load(
                str(index_path / "bm25_index"),
                load_corpus=False,
            )
        except (FileNotFoundError, OSError) as error:
            raise FileNotFoundError(
                f"BM25 index not found in {index_directory}. "
                "Run the index command first."
            ) from error

        try:
            with open(index_path / "chunks.pkl", "rb") as file:
                self.chunks: list[MinimalSource] = pickle.load(file)
        except (FileNotFoundError, OSError) as error:
            raise FileNotFoundError(
                f"Chunks file not found in {index_directory}. "
                "Run the index command first."
            ) from error

    def search(self, query: str, k: int = 5) -> list[MinimalSource]:
        """Search for the top-k most relevant chunks for a query."""
        if not query.strip() or k == 0:
            return []

        actual_k = min(k, len(self.chunks))
        tokenized_query = bm25s.tokenize([query])
        results, _ = self.retriever.retrieve(tokenized_query, k=actual_k)
        indices = results[0].tolist()

        retrieved = []
        for index in indices:
            retrieved.append(self.chunks[index])

        return retrieved
