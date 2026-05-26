"""Markdown chunker: splits markdown files on headers."""

from student.models import MinimalSource


class MarkdownChunker:
    """Splits a markdown file into chunks based on headers."""

    def __init__(self, max_chunk_size: int = 2000) -> None:
        """Initialize the chunker with a max chunk size."""
        self.max_chunk_size = max_chunk_size

    def chunk(self, file_path: str, content: str) -> list[MinimalSource]:
        """Split markdown content into chunks and return a list
        of MinimalSource.

        Cuts on headers (lines starting with #).
        If a section is too big, cuts it further by size.
        """
        chunks = []
        current_start = 0
        current_text = ""

        lines = content.splitlines(keepends=True)
        cursor = 0

        for line in lines:
            is_header = line.startswith("#")

            current_size = len(current_text) + len(line)
            would_overflow = current_size > self.max_chunk_size
            if (is_header or would_overflow) and current_text.strip():
                chunk_end = current_start + len(current_text)
                chunks.append(MinimalSource(
                    file_path=file_path,
                    first_character_index=current_start,
                    last_character_index=chunk_end,
                ))
                current_start = cursor
                current_text = ""

            current_text += line
            cursor += len(line)

        if current_text.strip():
            chunks.append(MinimalSource(
                file_path=file_path,
                first_character_index=current_start,
                last_character_index=current_start + len(current_text),
            ))

        return chunks
