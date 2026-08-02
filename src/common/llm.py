"""Local model loading, sampling, and per-token logprob extraction."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import torch
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

log = logging.getLogger(__name__)

DEFAULT_GEN = {
    "temperature": 0.7,
    "top_p": 0.9,
    "do_sample": True,
    "repetition_penalty": 1.1,
}


def device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


ADAPTER_CONFIG = "adapter_config.json"


def is_adapter(model_path):
    """True if the path is a LoRA adapter rather than a standalone checkpoint."""
    return (Path(model_path) / ADAPTER_CONFIG).is_file()


def _adapter_base(model_path):
    config = json.loads((Path(model_path) / ADAPTER_CONFIG).read_text())
    return config.get("base_model_name_or_path")


def load_tokenizer(model_path, base_model=None):
    """Tokenizer from the checkpoint, falling back to the base for bare adapters."""
    try:
        return AutoTokenizer.from_pretrained(model_path, use_fast=True)
    except (OSError, ValueError):
        fallback = base_model or _adapter_base(model_path)
        log.debug("no tokenizer at %s; using %s", model_path, fallback)
        return AutoTokenizer.from_pretrained(fallback, use_fast=True)


def load_causal_lm(model_path, base_model=None):
    """Load a checkpoint for generation.

    Accepts either a standalone (merged) checkpoint or a LoRA adapter
    directory. Adapters are applied to `base_model` and merged in memory, so
    callers see an ordinary model either way and generation speed is unaffected.
    """
    tokenizer = load_tokenizer(model_path, base_model)

    if is_adapter(model_path):
        from peft import PeftModel

        base = base_model or _adapter_base(model_path)
        if not base:
            raise ValueError(
                f"{model_path} is a LoRA adapter but no base model was given; "
                "set models.base in the config"
            )
        log.info("loading adapter %s onto %s", model_path, base)
        model = AutoModelForCausalLM.from_pretrained(
            base, dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
        )
        model = PeftModel.from_pretrained(model, model_path)
        model = model.merge_and_unload()
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.eval()
    return model, tokenizer


def load_embedder(model_path):
    """Load the bi-encoder used by the ranker (bge-large-en-v1.5)."""
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModel.from_pretrained(model_path)
    model.eval()
    return model, tokenizer


def generate(model, tokenizer, prompt, max_new_tokens=512, with_logprobs=False, **gen_kwargs):
    """Sample a continuation.

    Returns ``text`` normally, or ``(text, logprobs)`` when ``with_logprobs`` is
    set. ``logprobs`` holds one float per token of ``text`` -- special tokens are
    stripped from both sides together so the two stay index-aligned, which is
    what the ranker's span matching depends on.
    """
    kwargs = {**DEFAULT_GEN, **gen_kwargs}
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            return_dict_in_generate=True,
            output_logits=with_logprobs,
            pad_token_id=tokenizer.pad_token_id,
            **kwargs,
        )

    generated_ids = output.sequences[0, inputs.input_ids.shape[1] :]

    if not with_logprobs:
        return tokenizer.decode(generated_ids, skip_special_tokens=True)

    logits = torch.stack(output.logits, dim=1).float()  # [1, T, V]
    token_logprobs = torch.log_softmax(logits, dim=-1)[
        0, torch.arange(generated_ids.shape[0]), generated_ids
    ]

    # Drop trailing special tokens (EOS) so len(logprobs) matches the token
    # count of the *decoded* text.
    special = set(tokenizer.all_special_ids)
    keep = [i for i, tid in enumerate(generated_ids.tolist()) if tid not in special]
    generated_ids = generated_ids[keep]
    token_logprobs = token_logprobs[keep]

    text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return text, token_logprobs.tolist()


def mean_pool(model, tokenizer, text):
    """Mean-pooled last hidden state -- the embedding the ranker scores with."""
    enc = tokenizer(text, return_tensors="pt", truncation=True, padding=False)
    with torch.no_grad():
        hidden = model(**enc).last_hidden_state * enc.attention_mask.unsqueeze(-1)
        pooled = hidden.sum(1) / enc.attention_mask.sum(1).unsqueeze(-1)
    return pooled.squeeze(0).tolist()
