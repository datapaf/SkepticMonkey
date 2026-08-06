"""Model loading and line-level uncertainty estimation."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import groupby
from typing import List
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from lm_polygraph.utils.model import WhiteboxModel
from lm_polygraph.utils.generation_parameters import GenerationParameters
from lm_polygraph.utils.factory_estimator import FactoryEstimator
from lm_polygraph.utils.builder_enviroment_stat_calculator import (
    BuilderEnvironmentStatCalculator,
)
from lm_polygraph.utils.manager import UEManager
from lm_polygraph.utils.dataset import Dataset
from lm_polygraph.utils.factory_stat_calculator import StatCalculatorContainer

LLM_NAME = "deepseek-ai/deepseek-coder-1.3b-instruct"
DEFAULT_HEAD_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "heads",
    "instruct_line_diff_tp_hs",
)
MAX_NEW_TOKENS = 512


@dataclass
class StatCalculatorConfig:
    predict_token_uncertainties: bool
    uq_head_path: str
    args_generate: dict
    generations_cache_dir: str


@dataclass
class LineUncertainty:
    text: str
    uncertainty: float


@dataclass
class LineEstimateResult:
    input_text: str
    generation_text: str
    lines: List[LineUncertainty]
    generation_tokens: List[int]
    model_path: str
    estimator: str


def load_model(llm_id: str = LLM_NAME) -> tuple[WhiteboxModel, AutoTokenizer]:
    base_model = AutoModelForCausalLM.from_pretrained(
        llm_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(llm_id)
    gen_params = GenerationParameters(
        do_sample=False,
        max_new_tokens=MAX_NEW_TOKENS,
    )
    model = WhiteboxModel(
        base_model,
        tokenizer,
        model_path=llm_id,
        generation_parameters=gen_params,
    )
    return model, tokenizer


def _as_single_sample(value):
    """Unwrap a batch-of-1 result into the single sample payload."""
    if isinstance(value, list) and len(value) == 1 and isinstance(value[0], list):
        return value[0]
    if isinstance(value, list) and len(value) == 1 and not isinstance(
        value[0], (str, bytes)
    ):
        inner = value[0]
        if hasattr(inner, "tolist"):
            return inner.tolist()
        return inner
    if hasattr(value, "tolist"):
        return value.tolist()
    return value


def split_tokens_into_lines(token_ids: List[int], tokenizer) -> List[str]:
    tokens = tokenizer.convert_ids_to_tokens(token_ids)
    claim_tokens: List[List[str]] = []
    for key, group in groupby(tokens, key=lambda x: "Ċ" in x):
        if not key:
            claim_tokens.append(list(group))
        else:
            if claim_tokens:
                claim_tokens[-1].extend(list(group))
            else:
                claim_tokens.append(list(group))

    return [
        tokenizer.decode(
            tokenizer.convert_tokens_to_ids(claim),
            add_special_tokens=False,
        )
        for claim in claim_tokens
    ]


def estimate_line_uncertainty(
    input_text: str,
    model: WhiteboxModel,
    tokenizer,
    uq_head_path: str = DEFAULT_HEAD_DIR,
) -> LineEstimateResult:
    factory = FactoryEstimator()
    estimator = factory("luh.builder_LuhClaimEstimatorDummy", dict())
    sc1 = StatCalculatorContainer(
        name="Luh",
        cfg=StatCalculatorConfig(
            predict_token_uncertainties=False,
            uq_head_path=uq_head_path,
            args_generate={"max_new_tokens": MAX_NEW_TOKENS},
            generations_cache_dir="",
        ),
        stats=["greedy_tokens", "greedy_texts", "uhead_features", "claims"],
        dependencies=[],
        builder="luh.builder_CalculatorInferCodeLines",
    )
    sc2 = StatCalculatorContainer(
        name="luh_claim",
        stats=["uncertainty_claim_logits"],
        dependencies=["claims", "uhead_features"],
        builder="luh.builder_CalculatorApplyUQHead",
    )
    man = UEManager(
        Dataset([input_text], [""], batch_size=1),
        model,
        [estimator],
        available_stat_calculators=[sc1, sc2],
        builder_env_stat_calc=BuilderEnvironmentStatCalculator(model),
        generation_metrics=[],
        ue_metrics=[],
        processors=[],
        ignore_exceptions=False,
        verbose=False,
        max_new_tokens=model.generation_parameters.max_new_tokens,
    )
    man()

    ue = _as_single_sample(man.estimations[estimator.level, str(estimator)])
    texts = man.stats.get("greedy_texts", None)
    tokens = man.stats.get("greedy_tokens", None)

    generation_text = texts[0] if isinstance(texts, list) else texts
    generation_tokens: List[int] = []
    if tokens is not None and len(tokens) > 0:
        generation_tokens = list(tokens[0][:-1])

    line_texts = split_tokens_into_lines(generation_tokens, tokenizer)
    uncertainties = [float(x) for x in ue]

    if len(line_texts) != len(uncertainties):
        n = min(len(line_texts), len(uncertainties))
        line_texts = line_texts[:n]
        uncertainties = uncertainties[:n]

    lines = [
        LineUncertainty(text=text, uncertainty=score)
        for text, score in zip(line_texts, uncertainties)
    ]

    return LineEstimateResult(
        input_text=input_text,
        generation_text=generation_text or "",
        lines=lines,
        generation_tokens=generation_tokens,
        model_path=model.model_path or LLM_NAME,
        estimator=str(estimator),
    )
