from src.student.models import (AnsweredQuestion,
                                MinimalSource,
                                StudentSearchResults)


class Evaluator:
    """Computes recall@k by comparing retrieved sources to ground truth."""

    MIN_OVERLAP_RATIO = 0.05

    def _compute_overlap(
        self, retrieved: MinimalSource, ground_truth: MinimalSource
    ) -> float:
        """Compute the overlap ratio between two sources from the same file.
        """
        if retrieved.file_path != ground_truth.file_path:
            return 0.0

        overlap_start = max(
            retrieved.first_character_index,
            ground_truth.first_character_index,
        )
        overlap_end = min(
            retrieved.last_character_index,
            ground_truth.last_character_index,
        )
        overlap_size = max(0, overlap_end - overlap_start)

        ground_truth_size = (
            ground_truth.last_character_index
            - ground_truth.first_character_index
        )

        if ground_truth_size == 0:
            return 0.0

        return overlap_size / ground_truth_size

    def _is_found(
        self,
        ground_truth_source: MinimalSource,
        retrieved_sources: list[MinimalSource],
    ) -> bool:
        """Check if a ground truth source is found in the retrieved sources.

        A source is considered found if any retrieved source overlaps
        by at least 5% with the ground truth source.

        Args:
            ground_truth_source: the correct source to find.
            retrieved_sources: the sources returned by the retriever.

        Returns:
            True if the source is found, False otherwise.
        """
        for retrieved in retrieved_sources:
            overlap = self._compute_overlap(retrieved, ground_truth_source)
            if overlap >= self.MIN_OVERLAP_RATIO:
                return True
        return False

    def recall_at_k(
        self,
        search_results: StudentSearchResults,
        ground_truth_questions: list[AnsweredQuestion],
    ) -> float:
        """Compute the average recall@k over all questions.

        Args:
            search_results: the retriever output for all questions.
            ground_truth_questions: the answered questions
            with correct sources.

        Returns:
            average recall@k as a float between 0.0 and 1.0.
        """
        ground_truth_map = {
            question.question_id: question
            for question in ground_truth_questions
        }

        total_recall = 0.0
        evaluated = 0

        for result in search_results.search_results:
            ground_truth = ground_truth_map.get(result.question_id)
            if ground_truth is None or not ground_truth.sources:
                continue

            found = 0
            for gt_source in ground_truth.sources:
                if self._is_found(gt_source, result.retrieved_sources):
                    found += 1

            total_recall += found / len(ground_truth.sources)
            evaluated += 1

        if evaluated == 0:
            return 0.0

        return total_recall / evaluated
