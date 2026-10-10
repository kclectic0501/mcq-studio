from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, field_validator


class Difficulty(StrEnum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


class Chunk(BaseModel):
    id: str
    document: str
    page: int = Field(ge=1)
    text: str


class Request(BaseModel):
    count: int = Field(default=10, ge=1, le=50)
    difficulty: Difficulty = Difficulty.MEDIUM


class Draft(BaseModel):
    model_config = ConfigDict(extra="forbid")
    question: str = Field(min_length=10, max_length=1000)
    options: list[str] = Field(min_length=4, max_length=4)
    answer_index: int = Field(ge=0, le=3)
    explanation: str = Field(min_length=10, max_length=2000)
    difficulty: Difficulty
    quote: str = Field(min_length=10, max_length=2000)

    @field_validator("question", "explanation", "quote")
    @classmethod
    def nonblank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Text cannot be blank")
        return value.strip()

    @field_validator("options")
    @classmethod
    def distinct_options(cls, values: list[str]) -> list[str]:
        cleaned = [value.strip() for value in values]
        if any(not value for value in cleaned):
            raise ValueError("Options cannot be blank")
        if len({value.casefold() for value in cleaned}) != 4:
            raise ValueError("Options must be distinct")
        return cleaned


class Critique(BaseModel):
    passed: bool
    issues: list[str] = Field(default_factory=list)


class MCQ(Draft):
    source: Chunk


class Quiz(BaseModel):
    requested: int
    questions: list[MCQ]
    warnings: list[str] = Field(default_factory=list)
