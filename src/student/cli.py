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

    def index(self, max_chunk_size: int = 2000) -> None:
        """Index the vLLM repository into a BM25 index.

        Args:
            max_chunk_size: maximum number of characters per chunk.
        """
        if max_chunk_size <= 0:
            print("max_chunk_size must be greater than 0.")
            return

        print(f"Ingesting repository: {RAW_DIR}")
        ingester = Ingester(max_chunk_size=max_chunk_size)
        chunks, contents = ingester.ingest(RAW_DIR)

        if not chunks:
            print(f"No files found in {RAW_DIR}."
                  "Check that the repository is present.")
            return

        print(f"Indexing {len(chunks)} chunks...")
        indexer = BM25Indexer()
        indexer.build(chunks, contents)
        indexer.save(PROCESSED_DIR)

        print(f"Ingestion complete! Indices saved under {PROCESSED_DIR}/")

    def search(self, query: str, k: int = 5) -> None:
        """Search the index for a single query and print results.

        Args:
            query: the search query.
            k: number of results to return.
        """
        if not query.strip():
            print("Empty query, no results.")
            return

        if k == 0:
            print("k=0, no results.")
            return

        if k > 100:
            k = 100

        try:
            retriever = BM25Retriever(PROCESSED_DIR)
        except FileNotFoundError as error:
            print(f"Error: {error}")
            return

        results = retriever.search(query, k=k)

        for i, source in enumerate(results):
            print(
                f"[{i + 1}] {source.file_path} "
                f"(chars {source.first_character_index}"
                f"-{source.last_character_index})"
            )

    def search_dataset(
        self,
        dataset_path: str,
        k: int = 5,
        save_directory: str = "data/output/search_results",
    ) -> None:
        """Search the index for all questions in a dataset.

        Args:
            dataset_path: path to the JSON dataset file.
            k: number of results per question.
            save_directory: directory to save results.
        """
        try:
            dataset = RagDataset.model_validate_json(
                Path(dataset_path).read_text(encoding="utf-8")
            )
        except (
                FileNotFoundError,
                OSError,
                UnicodeDecodeError,
                ValueError
        ) as error:
            print(f"Error loading dataset: {error}")
            return

        try:
            retriever = BM25Retriever(PROCESSED_DIR)
        except FileNotFoundError as error:
            print(f"Error: {error}")
            return

        search_results = []
        for question in tqdm(dataset.rag_questions, desc="Searching"):
            retrieved = retriever.search(question.question_str, k=k)
            search_results.append(MinimalSearchResults(
                question_id=question.question_id,
                question_str=question.question_str,
                retrieved_sources=retrieved,
            ))

        output = StudentSearchResults(search_results=search_results, k=k)

        save_path = Path(save_directory)
        save_path.mkdir(parents=True, exist_ok=True)
        filename = Path(dataset_path).name

        output_file = save_path / filename
        output_file.write_text(
            output.model_dump_json(indent=2), encoding="utf-8"
        )
        print(f"Saved student_search_results to {output_file}")

    def answer(self, query: str, k: int = 5) -> None:
        """Answer a single question using retrieved context.

        Args:
            query: the question to answer.
            k: number of chunks to retrieve.
        """
        if not query.strip():
            print("Empty query, no answer.")
            return

        if k == 0:
            print("k=0, no context to answer from.")
            return

        try:
            retriever = BM25Retriever(PROCESSED_DIR)
        except FileNotFoundError as error:
            print(f"Error: {error}")
            return

        sources = retriever.search(query, k=k)

        print("Loading model...")
        generator = Generator()
        answer = generator.generate(query, sources, RAW_DIR)

        print(f"\nAnswer: {answer}")

    def answer_dataset(
        self,
        student_search_results_path: str,
        save_directory: str = "data/output/search_results_and_answer",
    ) -> None:
        """Generate answers for all questions in a search results file.

        Args:
            student_search_results_path: path to the search results JSON file.
            save_directory: directory to save results with answers.
        """
        try:
            search_results = StudentSearchResults.model_validate_json(
                Path(student_search_results_path).read_text(encoding="utf-8")
            )
        except (FileNotFoundError, OSError, UnicodeDecodeError, ValueError) as error:
            print(f"Error loading search results: {error}")
            return

        print(f"Loaded {len(search_results.search_results)} questions.")
        print("Loading model...")
        generator = Generator()

        answers = []
        for result in tqdm(
                search_results.search_results,
                desc="Generating answers"
        ):
            answer_text = generator.generate(
                result.question_str, result.retrieved_sources, RAW_DIR
            )
            answers.append(MinimalAnswer(
                question_id=result.question_id,
                question_str=result.question ,
                retrieved_sources=result.retrieved_sources,
                answer=answer_text,
            ))

        output = StudentSearchResultsAndAnswer(
            search_results=answers,
            k=search_results.k,
        )

        save_path = Path(save_directory)
        save_path.mkdir(parents=True, exist_ok=True)
        filename = Path(student_search_results_path).name

        output_file = save_path / filename
        output_file.write_text(
            output.model_dump_json(indent=2), encoding="utf-8"
        )
        print(
            f"Saved student_search_results_and_answer to {output_file}"
        )

    def evaluate(
        self,
        student_answer_path: str,
        dataset_path: str,
        k: int = 5,
    ) -> None:
        """Evaluate search results against ground truth annotations.

        Args:
            student_answer_path: path to the student search results JSON.
            dataset_path: path to the answered dataset JSON.
            k: number of results that were retrieved.
        """
        try:
            search_results = StudentSearchResults.model_validate_json(
                Path(student_answer_path).read_text(encoding="utf-8")
            )
        except (
                FileNotFoundError,
                OSError,
                UnicodeDecodeError,
                ValueError
        ) as error:
            print(f"Error loading dataset: {error}")
            return

        try:
            dataset = RagDataset.model_validate_json(
                Path(dataset_path).read_text(encoding="utf-8")
            )
        except (
                FileNotFoundError,
                OSError,
                UnicodeDecodeError,
                ValueError
        ) as error:
            print(f"Error loading dataset: {error}")
            return

        ground_truth = [
            question for question in dataset.rag_questions
            if isinstance(question, AnsweredQuestion)
        ]

        evaluator = Evaluator()
        recall = evaluator.recall_at_k(search_results, ground_truth)

        print("Evaluation Results")
        print("=" * 40)
        print(f"Questions evaluated: {len(ground_truth)}")
        print(f"Recall@{k}: {recall:.3f}")
