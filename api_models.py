"""Queue job input types for the Laya prediction worker."""

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue, field_validator


class ChoiceQuestion(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    type: Literal["choice"]
    instructions: str
    criteria: (
        Annotated[dict[str, str | None], Field(min_length=2)]
        | Annotated[list[str], Field(min_length=2)]
    )

    @field_validator("criteria")
    @classmethod
    def distinct_labels(cls, criteria):
        if isinstance(criteria, list) and len(set(criteria)) != len(criteria):
            raise ValueError("Choice labels must be distinct")
        return criteria


class ScoreQuestion(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    type: Literal["score"]
    instructions: str
    criteria: Annotated[list[str], Field(min_length=2)]


class NoulQuestion(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    type: Literal["noul"]
    instructions: str
    criteria: dict[Literal["false", "true"], str | None] | None = None


Question = Annotated[
    ChoiceQuestion | ScoreQuestion | NoulQuestion, Field(discriminator="type")
]


class PredictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    state: dict[str, JsonValue] | str | list[JsonValue]
    questions: Annotated[dict[str, Question], Field(min_length=1)]
