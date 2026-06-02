"""Python chunker: splits Python files on functions and classes using AST."""

import ast

from student.models import MinimalSource


class PythonChunker:
    """Splits a Python file into chunks based on functions and classes."""

    def __init__(self, max_chunk_size: int = 2000) -> None:
        """Initialize the chunker with a max chunk size."""
        self.max_chunk_size = max_chunk_size

    def _get_line_offsets(self, content: str) -> list[int]:
        """Return the character offset of the start of each line.

        For example, if content is "hello\nworld\n",
        offsets will be [0, 6, 12].
        """
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
        """Split a chunk that is too big into smaller pieces by character
        size."""
        chunks = []
        chunk_start = start

        while chunk_start < end:
            chunk_end = min(chunk_start + self.max_chunk_size, end)
            chunks.append(MinimalSource(
                file_path=file_path,
                first_character_index=chunk_start,
                last_character_index=chunk_end,
            ))
            chunk_start = chunk_end

        return chunks

    def chunk(self, file_path: str, content: str) -> list[MinimalSource]:
        """Split Python content into chunks on functions and classes.

        Falls back to size-based splitting if the file cannot be parsed.
        If a function or class is bigger than max_chunk_size,
        splits it further."""
        try:
            tree = ast.parse(content)
        except SyntaxError:
            return self._split_by_size(file_path, content, 0, len(content))

        offsets = self._get_line_offsets(content)

        top_level_nodes = []
        for node in tree.body:
            if isinstance(
                    node,
                    (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
            ):
                top_level_nodes.append(node)

        if not top_level_nodes:
            return self._split_by_size(file_path, content, 0, len(content))

        top_level_nodes.sort(key=lambda node: node.lineno)

        chunks = []
        for node in top_level_nodes:
            if node.lineno is None or node.end_lineno is None:
                continue
            node_start = offsets[node.lineno - 1]
            node_end = offsets[node.end_lineno - 1] + len(
                content.splitlines(keepends=True)[node.end_lineno - 1]
            )
            node_size = node_end - node_start

            if node_size > self.max_chunk_size:
                split_chunks = self._split_by_size(
                    file_path, content, node_start, node_end
                )
                chunks.extend(split_chunks)
            else:
                chunks.append(MinimalSource(
                    file_path=file_path,
                    first_character_index=node_start,
                    last_character_index=node_end,
                ))

        return chunks
