"""Stage 4 -- ranking user dimensions down to the top-k gaps.

Every user dimension (explicit + implicit) is scored, all terms z-normalised:

    score = z(logprob) - lambda1 * z(max cos-sim to any system dimension)
                       - lambda2 * z(max cos-sim to any other user dimension)

  logprob   the DGA's own confidence in that dimension
  lambda1   suppresses dimensions the response already covered, surfacing the
            gaps the answer actually left open
  lambda2   suppresses near-duplicates

Embeddings are mean-pooled last hidden states from bge-large-en-v1.5 over
``f"{name}: {value}"``.
"""

from __future__ import annotations

import logging
import re
from difflib import SequenceMatcher

import torch

from src.common.llm import mean_pool

log = logging.getLogger(__name__)

VALUE_SPAN = re.compile(r'"value"\s*:\s*(.+?)\s*,\s*"justification"', re.DOTALL)


def embed_dims(dims, emb_model, emb_tokenizer):
    """Attach an `embed` vector to each dimension, in place."""
    for item in dims:
        item["embed"] = mean_pool(emb_model, emb_tokenizer, f"{item['name']}: {item['value']}")
    return dims


def get_dimension_logprobs(raw_text, logprobs, tokenizer, min_ratio=0.7):
    """Recover a mean logprob per emitted dimension value.

    Pulls every `"value": ... , "justification"` span out of the raw generation,
    then finds the first token window in the generation whose token ids match
    that span closely enough, and averages the logprobs over it.

    ``add_special_tokens=False`` on both tokenisations is load-bearing: Llama-3
    prepends BOS otherwise, which shifts every index by one against
    ``logprobs`` (one entry per *generated* token).
    """
    token_ids = tokenizer(raw_text, add_special_tokens=False)["input_ids"]
    if len(token_ids) != len(logprobs):
        log.debug(
            "token/logprob length mismatch (%d vs %d); matching over the overlap",
            len(token_ids),
            len(logprobs),
        )

    results = []
    for value in VALUE_SPAN.findall(raw_text):
        value = value.strip()
        if value.startswith('"') and value.endswith('"'):
            value = value[1:-1]

        value_tokens = tokenizer(value, add_special_tokens=False)["input_ids"]
        span = len(value_tokens)
        logprob = 0.0

        if span:
            for i in range(min(len(token_ids), len(logprobs)) - span + 1):
                window = token_ids[i : i + span]
                if SequenceMatcher(None, value_tokens, window).ratio() >= min_ratio:
                    logprob = sum(logprobs[i : i + span]) / span
                    break

        results.append({"value": value, "logprob": logprob})

    return results


def attach_logprobs(user_dims, raw_text, logprobs, tokenizer, min_ratio=0.7):
    """Match each parsed dimension to its closest emitted value span."""
    spans = get_dimension_logprobs(raw_text, logprobs, tokenizer, min_ratio)
    for dim in user_dims:
        if not spans:
            dim["logprob"] = 0.0
            continue
        target = _as_text(dim["value"])
        best = max(
            spans, key=lambda s: SequenceMatcher(None, target, _as_text(s["value"])).ratio()
        )
        dim["logprob"] = best["logprob"]
    return user_dims


def _as_text(value):
    return str(value) if isinstance(value, list) else value


def _zscore(tensor):
    std = tensor.std()
    if not torch.isfinite(std) or std < 1e-6:
        return torch.zeros_like(tensor)
    return (tensor - tensor.mean()) / std


def rank_user_dims(user_dims, system_dims, lambda1, lambda2, top_k=3):
    """Return the indices of the top-k user dimensions, best first."""
    if not user_dims:
        return []

    user_embeds = torch.tensor([d["embed"] for d in user_dims], dtype=torch.float32)

    if system_dims:
        system_embeds = torch.tensor(
            [d["embed"] for d in system_dims], dtype=torch.float32
        )
        sim_to_system = torch.tensor(
            [
                torch.cosine_similarity(u.unsqueeze(0), system_embeds).max()
                for u in user_embeds
            ]
        )
    else:
        sim_to_system = torch.zeros(len(user_dims))

    sim_among_users = torch.zeros(len(user_dims))
    for i, u in enumerate(user_embeds):
        others = torch.cat([user_embeds[:i], user_embeds[i + 1 :]])
        if len(others):
            sim_among_users[i] = torch.cosine_similarity(u.unsqueeze(0), others).max()

    logprob = torch.tensor([d.get("logprob", 0.0) for d in user_dims], dtype=torch.float32)

    scores = (
        _zscore(logprob)
        - lambda1 * _zscore(sim_to_system)
        - lambda2 * _zscore(sim_among_users)
    )
    return torch.argsort(scores, descending=True)[:top_k].tolist()
