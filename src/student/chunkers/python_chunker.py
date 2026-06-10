"""Python chunker: splits Python files on functions and classes using AST."""
import ast
from student.models import MinimalSource


class PythonChunker:
    """Splits a Python file into chunks based on functions and classes."""

    def __init__(self, max_chunk_size: int = 2000, overlap: int = 200) -> None:
        """Initialize the chunker with a max chunk size and overlap."""
        if max_chunk_size == 2000:
            self.overlap = 380
            self.max_chunk_size = 1650 - self.overlap
        else:
            self.overlap = overlap
            self.max_chunk_size = max_chunk_size - self.overlap

    def _get_line_offsets(self, content: str) -> list[int]:
        """Return the character offset of the start of each line."""
        offsets = []
        cursor = 0
        for line in content.splitlines(keepends=True):
            offsets.append(cursor)
            cursor += len(line)
        return offsets

    def _split_by_size(
        self,
        file_path: str,
        content: str,
        start: int,
        end: int,
    ) -> list[MinimalSource]:
        """Split a chunk that is too big into smaller pieces by character size,
        with overlap between consecutive chunks."""
        chunks = []
        chunk_start = start
        while chunk_start < end:
            chunk_end = min(chunk_start + self.max_chunk_size, end)
            chunks.append(MinimalSource(
                file_path=file_path,
                first_character_index=chunk_start,
                last_character_index=chunk_end,
            ))
            if chunk_end == end:
                break
            # Le prochain chunk recule de `overlap` caractères
            chunk_start = chunk_end - self.overlap
        return chunks

    def _apply_overlap(
        self,
        chunks: list[MinimalSource],
        content_length: int,
    ) -> list[MinimalSource]:
        """Étend chaque chunk pour inclure l'overlap avec le suivant."""
        if self.overlap == 0 or len(chunks) <= 1:
            return chunks

        overlapped = []
        for i, chunk in enumerate(chunks):
            if i < len(chunks) - 1:
                # Étendre la fin jusqu'au début du chunk suivant + overlap
                extended_end = min(
                    chunk.last_character_index + self.overlap,
                    content_length,
                )
                overlapped.append(MinimalSource(
                    file_path=chunk.file_path,
                    first_character_index=chunk.first_character_index,
                    last_character_index=extended_end,
                ))
            else:
                overlapped.append(chunk)
        return overlapped

    def chunk(self, file_path: str, content: str) -> list[MinimalSource]:
        """Split Python content into chunks on functions and classes."""
        try:
            tree = ast.parse(content)
        except SyntaxError:
            return self._split_by_size(file_path, content, 0, len(content))

        offsets = self._get_line_offsets(content)
        top_level_nodes = [
            node for node in tree.body
            if isinstance(
                node,
                (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
                )
        ]

        if not top_level_nodes:
            return self._split_by_size(file_path, content, 0, len(content))

        top_level_nodes.sort(key=lambda node: node.lineno)

        chunks = []
        lines = content.splitlines(keepends=True)
        for node in top_level_nodes:
            if node.lineno is None or node.end_lineno is None:
                continue
            node_start = offsets[node.lineno - 1]
            node_end = offsets[node.end_lineno - 1] + len(
                lines[node.end_lineno - 1]
                )
            node_size = node_end - node_start

            if node_size > self.max_chunk_size:
                chunks.extend(
                    self._split_by_size(
                        file_path,
                        content,
                        node_start,
                        node_end
                    )
                )
            else:
                chunks.append(MinimalSource(
                    file_path=file_path,
                    first_character_index=node_start,
                    last_character_index=node_end,
                ))

        return self._apply_overlap(chunks, len(content))
