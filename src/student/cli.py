# import json
from pathlib import Path

from tqdm import tqdm

from student.evaluators.evaluator import Evaluator
from student.generators.generator import Generator
from student.indexers.bm25_indexer import BM25Indexer
from student.ingester import Ingester
from student.models import (
    AnsweredQuestion,
    MinimalAnswer,
    MinimalSearchResults,
    RagDataset,
    StudentSearchResults,
    StudentSearchResultsAndAnswer,
)
from student.retrievers.bm25_retriever import BM25Retriever

PROCESSED_DIR = "data/processed"
RAW_DIR = "data/raw/vllm-0.10.1"


class CLI:
    """RAG pipeline CLI — index, search, answer, evaluate."""

    @staticmethod
    def _parse_int(value: object, name: str) -> int | None:
        """Parse value as int, print an error and return None if it fails."""
        try:
            return int(value)  # type: ignore[call-overload, no-any-return]
        except (ValueError, TypeError):
            print(f"Error: '{name}' must be an integer, got {value!r}.")
            return None

    @staticmethod
    def _clamp_k(k: int) -> int:
        """Clamp k to [1, 100]."""
        if k < 1:
            print("Error: k must be at least 1, clamping to 1.")
            return 1
        if k > 100:
            print("Warning: k too high, clamping to 100.")
            return 100
        return k

    @staticmethod
    def _resolve_file(path: object, name: str) -> Path | None:
        """Check that path is a string and points to an existing file."""
        if not isinstance(path, str):
            print(
                f"Error: '{name}' must be a string path, "
                f"got {type(path).__name__!r}."
            )
            return None
        p = Path(path)
        if not p.exists():
            print(f"Error: {name} file not found: {path!r}.")
            return None
        if not p.is_file():
            print(f"Error: {name} path is not a file: {path!r}.")
            return None
        return p

    def index(self, max_chunk_size: object = 2000) -> None:
        """Index the vLLM repository into a BM25 index.

        Args:
            max_chunk_size: maximum number of characters per chunk.
        """
        parsed = self._parse_int(max_chunk_size, "max_chunk_size")
        if parsed is None:
            return
        if parsed <= 0:
            print("Error: max_chunk_size must be greater than 0.")
            return
        if parsed > 2000:
            print("Error: max_chunk_size can't be greater than 2000.")
            return

        print(f"Ingesting repository: {RAW_DIR}")
        ingester = Ingester(max_chunk_size=parsed)
        chunks, contents = ingester.ingest(RAW_DIR)

        if not chunks:
            print(
                f"No files found in {RAW_DIR}. "
                "Check that the repository is present."
            )
            return

        print(f"Indexing {len(chunks)} chunks...")
        indexer = BM25Indexer()
        indexer.build(chunks, contents)
        indexer.save(PROCESSED_DIR)

        print(f"Ingestion complete! Indices saved under {PROCESSED_DIR}/")

    def search(
        self,
        query: object = "",
        k: object = 5,
        save_directory: object = "data/output/search_results",
    ) -> None:
        """Search the index for a single query and print results.

        Args:
            query: the search query.
            k: number of results to return.
            save_directory: directory to save results.
        """
        if not isinstance(query, str):
            print(
                f"Error: 'query' must be a string, "
                f"got {type(query).__name__!r}."
            )
            return
        if not query.strip():
            print("Error: empty query, no results.")
            return
        if not isinstance(save_directory, str):
            print(
                f"Error: 'save_directory' must be a string, "
                f"got {type(save_directory).__name__!r}."
            )
            return

        parsed_k = self._parse_int(k, "k")
        if parsed_k is None:
            return
        parsed_k = self._clamp_k(parsed_k)

        try:
            retriever = BM25Retriever(PROCESSED_DIR)
        except FileNotFoundError as error:
            print(f"Error: {error}")
            return

        results = retriever.search(query, k=parsed_k)

        for i, source in enumerate(results):
            print(
                f"[{i + 1}] {source.file_path} "
                f"(chars {source.first_character_index}"
                f"-{source.last_character_index})"
            )

        output = StudentSearchResults(
            search_results=[MinimalSearchResults(
                question_id="manual_query",
                question=query,
                retrieved_sources=results,
            )],
            k=parsed_k,
        )

        try:
            save_path = Path(save_directory)
            save_path.mkdir(parents=True, exist_ok=True)
            output_file = save_path / "manual_query.json"
            output_file.write_text(
                output.model_dump_json(indent=2), encoding="utf-8"
            )
            print(f"Saved search results to {output_file}")
        except OSError as error:
            print(f"Error saving results: {error}")

    def search_dataset(
        self,
        dataset_path: object = "",
        k: object = 5,
        save_directory: object = "data/output/search_results",
    ) -> None:
        """Search the index for all questions in a dataset.

        Args:
            dataset_path: path to the JSON dataset file.
            k: number of results per question.
            save_directory: directory to save results.
        """
        parsed_k = self._parse_int(k, "k")
        if parsed_k is None:
            return
        parsed_k = self._clamp_k(parsed_k)

        if not isinstance(save_directory, str):
            print(
                f"Error: 'save_directory' must be a string, "
                f"got {type(save_directory).__name__!r}."
            )
            return

        resolved = self._resolve_file(dataset_path, "dataset_path")
        if resolved is None:
            return

        try:
            dataset = RagDataset.model_validate_json(
                resolved.read_text(encoding="utf-8")
            )
        except (OSError, UnicodeDecodeError, ValueError) as error:
            print(f"Error loading dataset: {error}")
            return

        try:
            retriever = BM25Retriever(PROCESSED_DIR)
        except FileNotFoundError as error:
            print(f"Error: {error}")
            return

        search_results = []
        for question in tqdm(dataset.rag_questions, desc="Searching"):
            retrieved = retriever.search(question.question_str, k=parsed_k)
            search_results.append(MinimalSearchResults(
                question_id=question.question_id,
                question=question.question_str,
                retrieved_sources=retrieved,
            ))

        output = StudentSearchResults(
            search_results=search_results, k=parsed_k
        )

        try:
            save_path = Path(save_directory)
            save_path.mkdir(parents=True, exist_ok=True)
            output_file = save_path / resolved.name
            output_file.write_text(
                output.model_dump_json(indent=2), encoding="utf-8"
            )
            print(f"Saved student_search_results to {output_file}")
        except OSError as error:
            print(f"Error saving results: {error}")

    def answer(self, query: object = "", k: object = 5) -> None:
        """Answer a single question using retrieved context.

        Args:
            query: the question to answer.
            k: number of chunks to retrieve.
        """
        if not isinstance(query, str):
            print(
                f"Error: 'query' must be a string, "
                f"got {type(query).__name__!r}."
            )
            return
        if not query.strip():
            print("Error: empty query, no answer.")
            return

        parsed_k = self._parse_int(k, "k")
        if parsed_k is None:
            return
        parsed_k = self._clamp_k(parsed_k)

        try:
            retriever = BM25Retriever(PROCESSED_DIR)
        except FileNotFoundError as error:
            print(f"Error: {error}")
            return

        sources = retriever.search(query, k=parsed_k)

        generator = Generator()
        answer = generator.generate(query, sources, RAW_DIR)
        print(f"\nAnswer: {answer}")

    def answer_dataset(
        self,
        student_search_results_path: object = "",
        save_directory: object = "data/output/search_results_and_answer",
    ) -> None:
        """Generate answers for all questions in a search results file.

        Args:
            student_search_results_path: path to the search results JSON file.
            save_directory: directory to save results with answers.
        """
        if not isinstance(save_directory, str):
            print(
                f"Error: 'save_directory' must be a string, "
                f"got {type(save_directory).__name__!r}."
            )
            return

        resolved = self._resolve_file(
            student_search_results_path, "student_search_results_path"
        )
        if resolved is None:
            return

        try:
            search_results = StudentSearchResults.model_validate_json(
                resolved.read_text(encoding="utf-8")
            )
        except (OSError, UnicodeDecodeError, ValueError) as error:
            print(f"Error loading search results: {error}")
            return

        print(f"Loaded {len(search_results.search_results)} questions.")
        generator = Generator()

        answers = []
        for result in tqdm(
            search_results.search_results, desc="Generating answers"
        ):
            answer_text = generator.generate(
                result.question_str, result.retrieved_sources, RAW_DIR
            )
            answers.append(MinimalAnswer(
                question_id=result.question_id,
                question_str=result.question_str,  # type: ignore[call-arg]
                retrieved_sources=result.retrieved_sources,
                answer=answer_text,
            ))

        output = StudentSearchResultsAndAnswer(
            search_results=answers,
            k=search_results.k,
        )

        try:
            save_path = Path(save_directory)
            save_path.mkdir(parents=True, exist_ok=True)
            output_file = save_path / resolved.name
            output_file.write_text(
                output.model_dump_json(indent=2), encoding="utf-8"
            )
            print(f"Saved student_search_results_and_answer to {output_file}")
        except OSError as error:
            print(f"Error saving results: {error}")

    def evaluate(
        self,
        student_answer_path: object = "",
        dataset_path: object = "",
        k: object = 5,
    ) -> None:
        """Evaluate search results against ground truth annotations.

        Args:
            student_answer_path: path to the student search results JSON.
            dataset_path: path to the answered dataset JSON.
            k: number of results that were retrieved.
        """
        parsed_k = self._parse_int(k, "k")
        if parsed_k is None:
            return
        parsed_k = self._clamp_k(parsed_k)

        resolved_answers = self._resolve_file(
            student_answer_path, "student_answer_path"
        )
        if resolved_answers is None:
            return

        resolved_dataset = self._resolve_file(dataset_path, "dataset_path")
        if resolved_dataset is None:
            return

        try:
            search_results = StudentSearchResults.model_validate_json(
                resolved_answers.read_text(encoding="utf-8")
            )
        except (OSError, UnicodeDecodeError, ValueError) as error:
            print(f"Error loading student answers: {error}")
            return

        try:
            dataset = RagDataset.model_validate_json(
                resolved_dataset.read_text(encoding="utf-8")
            )
        except (OSError, UnicodeDecodeError, ValueError) as error:
            print(f"Error loading dataset: {error}")
            return

        ground_truth = [
            question for question in dataset.rag_questions
            if isinstance(question, AnsweredQuestion)
        ]

        if not ground_truth:
            print("Error: no answered questions found in dataset.")
            return

        evaluator = Evaluator()
        recall = evaluator.recall_at_k(search_results, ground_truth)

        print("Evaluation Results")
        print("=" * 40)
        print(f"Questions evaluated: {len(ground_truth)}")
        print(f"Recall@{parsed_k}: {recall:.3f}")
