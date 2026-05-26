"""Generator: uses Qwen3-0.6B to generate answers from retrieved chunks."""
import re
from pathlib import Path

from transformers import AutoModelForCausalLM, AutoTokenizer

from student.models import MinimalSource


class Generator:
    """Loads Qwen3-0.6B and generates answers from retrieved context."""

    MODEL_NAME = "Qwen/Qwen3-0.6B"
    MAX_NEW_TOKENS = 512
    MAX_CONTEXT_CHARS = 6000

    def __init__(self) -> None:
        """Load the tokenizer and model from Hugging Face."""
        self.tokenizer = AutoTokenizer.from_pretrained(self.MODEL_NAME)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.MODEL_NAME,
            torch_dtype="auto",
            device_map="auto",
        )

    def _read_chunk_content(
        self, source: MinimalSource, repo_path: str
    ) -> str:
        """Read the actual text of a chunk from disk.

        Args:
            source: the MinimalSource with file path and char positions.
            repo_path: root path of the repository.

        Returns:
            the text content of the chunk.
        """
        file_path = Path(repo_path) / source.file_path
        try:
            content = file_path.read_text(encoding="utf-8")
            start = source.first_character_index
            end = source.last_character_index

            return content[start:end]
        except (FileNotFoundError, OSError, UnicodeDecodeError):
            return ""

    def _build_context(
        self, sources: list[MinimalSource], repo_path: str
    ) -> str:
        """Concatenate chunk texts into a context string within char limits.

        Args:
            sources: list of retrieved MinimalSource chunks.
            repo_path: root path of the repository.

        Returns:
            a single string with all chunk contents joined.
        """
        context_parts = []
        total_chars = 0

        for source in sources:
            chunk_text = self._read_chunk_content(source, repo_path)
            if not chunk_text:
                continue

            if total_chars + len(chunk_text) > self.MAX_CONTEXT_CHARS:
                break

            context_parts.append(
                f"Source: {source.file_path}\n{chunk_text}"
            )
            total_chars += len(chunk_text)

        return "\n\n---\n\n".join(context_parts)

    def generate(
        self,
        question: str,
        sources: list[MinimalSource],
        repo_path: str,
    ) -> str:
        """Generate an answer for a question given retrieved sources.

        Args:
            question: the question to answer.
            sources: list of retrieved MinimalSource chunks.
            repo_path: root path of the repository.

        Returns:
            a generated answer string.
        """
        context = self._build_context(sources, repo_path)

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a helpful assistant answering questions about the vLLM codebase. "
                    "Answer based ONLY on the provided sources. "
                    "Be concise and self-contained. Mention the source file(s) you draw from."
                ),
            },
            {
                "role": "user",
                "content": f"Sources:\n{context}\n\nQuestion: {question}",
            },
        ]

        inputs = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        ).to(self.model.device)

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=self.MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=self.tokenizer.eos_token_id,
            repetition_penalty=1.3,
        )

        new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        answer = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        answer = re.sub(r"<think>.*?</think>", "", answer, flags=re.DOTALL).strip()

        return answer.strip()
