"""Pydantic models for the RAG pipeline."""

import uuid
from typing import List

from pydantic import BaseModel, Field, ConfigDict


class MinimalSource(BaseModel):
    """A chunk of source code or documentation retrieved
    from the knowledge base."""

    file_path: str
    first_character_index: int
    last_character_index: int


class UnansweredQuestion(BaseModel):
    """A question without an answer or sources."""
    model_config = ConfigDict(populate_by_name=True)

    question_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    question_str: str = Field(alias="question")


class AnsweredQuestion(UnansweredQuestion):
    """A question with its ground truth sources and answer."""

    sources: List[MinimalSource]
    answer: str


class RagDataset(BaseModel):
    """A dataset of RAG questions, answered or not."""

    rag_questions: List[AnsweredQuestion | UnansweredQuestion]


class MinimalSearchResults(BaseModel):
    """Search results for a single question."""
    model_config = ConfigDict(populate_by_name=True, arbitrary_types_allowed=True)

    question_id: str
    question_str: str = Field(alias="question")
    retrieved_sources: List[MinimalSource]


class MinimalAnswer(MinimalSearchResults):
    """Search results plus a generated answer for a single question."""

    answer: str


class StudentSearchResults(BaseModel):
    """All search results for a dataset."""

    search_results: List[MinimalSearchResults]
    k: int


class StudentSearchResultsAndAnswer(StudentSearchResults):
    """All search results with generated answers for a dataset."""

    search_results: List[MinimalAnswer]  # type: ignore[assignment]
    k: int
