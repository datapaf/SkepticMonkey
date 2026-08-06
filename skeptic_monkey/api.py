"""SkepticMonkey API: generate text and score line-level uncertainty."""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from skeptic_monkey.service import (
    DEFAULT_HEAD_DIR,
    LLM_NAME,
    LineEstimateResult,
    estimate_line_uncertainty,
    load_model,
)

DEFAULT_LLM_ID = os.environ.get("SKEPTIC_MONKEY_LLM_ID", LLM_NAME)
DEFAULT_HEAD_PATH = os.environ.get("SKEPTIC_MONKEY_HEAD_PATH", DEFAULT_HEAD_DIR)
DEFAULT_HOST = os.environ.get("SKEPTIC_MONKEY_HOST", "0.0.0.0")
DEFAULT_PORT = int(os.environ.get("SKEPTIC_MONKEY_PORT", "8000"))


class LineEstimateRequest(BaseModel):
    input_text: str = Field(..., description="Prompt / templated question")
    uq_head_path: Optional[str] = Field(
        None, description="Optional override for the uncertainty head path"
    )


class LineUncertaintyItem(BaseModel):
    text: str
    uncertainty: float


class LineEstimateResponse(BaseModel):
    input_text: str
    generation_text: str
    lines: List[LineUncertaintyItem]
    generation_tokens: List[int]
    model_path: str
    estimator: str


class AppState:
    model = None
    tokenizer = None
    llm_id: str = DEFAULT_LLM_ID
    head_path: str = DEFAULT_HEAD_PATH


state = AppState()


@asynccontextmanager
async def lifespan(app: FastAPI):
    state.llm_id = DEFAULT_LLM_ID
    state.head_path = DEFAULT_HEAD_PATH
    state.model, state.tokenizer = load_model(state.llm_id)
    yield


app = FastAPI(
    title="SkepticMonkey",
    description=(
        "Generate LLM responses and return line-level uncertainty scores "
        "for every line (code and prose)."
    ),
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/health")
def health():
    ready = state.model is not None and state.tokenizer is not None
    return {
        "status": "ok" if ready else "loading",
        "service": "SkepticMonkey",
        "llm_id": state.llm_id,
        "head_path": state.head_path,
    }


@app.post("/estimate/line", response_model=LineEstimateResponse)
def estimate_line(request: LineEstimateRequest):
    if state.model is None or state.tokenizer is None:
        raise HTTPException(status_code=503, detail="Model is not loaded yet")

    head_path = request.uq_head_path or state.head_path
    try:
        result: LineEstimateResult = estimate_line_uncertainty(
            request.input_text,
            state.model,
            state.tokenizer,
            uq_head_path=head_path,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return LineEstimateResponse(
        input_text=result.input_text,
        generation_text=result.generation_text,
        lines=[
            LineUncertaintyItem(text=line.text, uncertainty=line.uncertainty)
            for line in result.lines
        ],
        generation_tokens=result.generation_tokens,
        model_path=result.model_path,
        estimator=result.estimator,
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "skeptic_monkey.api:app",
        host=DEFAULT_HOST,
        port=DEFAULT_PORT,
        reload=False,
    )
