"""Markdown chunker: splits markdown files on headers."""
from student.models import MinimalSource


class MarkdownChunker:
    """Splits a markdown file into chunks based on headers."""

    def __init__(self, max_chunk_size: int = 2000, overlap: int = 200) -> None:
        """Initialize the chunker with a max chunk size and overlap."""
        max_chunk_size = max(560, min(max_chunk_size, 2000))

        if max_chunk_size == 2000:
            self.overlap = 280
            self.max_chunk_size = 1855 - self.overlap
        else:
            self.overlap = min(overlap, max_chunk_size // 2)
            self.max_chunk_size = max_chunk_size - self.overlap

    def _overlap_prefix(self, content: str, end: int) -> str:
        """Retourne les `overlap` derniers caractères avant `end`."""
        start = max(0, end - self.overlap)
        return content[start:end]

    def chunk(self, file_path: str, content: str) -> list[MinimalSource]:
        """Split markdown content into chunks and return a list
        of MinimalSource.
        Cuts on headers (lines starting with #).
        If a section is too big, cuts it further by size.
        Consecutive chunks share `overlap` characters of context.
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
                # Le prochain chunk commence `overlap` caractères en arrière
                overlap_text = self._overlap_prefix(content, chunk_end)
                current_start = chunk_end - len(overlap_text)
                current_text = overlap_text

            current_text += line
            cursor += len(line)

        if current_text.strip():
            chunks.append(MinimalSource(
                file_path=file_path,
                first_character_index=current_start,
                last_character_index=current_start + len(current_text),
            ))

        return chunks
