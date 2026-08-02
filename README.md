# ProPer Agents (ACL 2026)

*Proactivity Driven Personalized Agents for Advancing Knowledge Gap Navigation*
— ACL 2026.

[[paper]](https://aclanthology.org/2026.findings-acl.2082/)
· [[data]](https://huggingface.co/datasets/itsgupta/proper-agents-data)
· [[adapters]](https://huggingface.co/itsgupta/proper-agents-models)

Two finetuned Llama-3-8B agents that turn a flat answer into a proactive one.

**RGA** (Response Generating Agent) writes a base response. **DGA** (Dimension
Generating Agent) reads the query and proposes what it left unspecified, then
reads the response and reports what it explicitly covered. A ranker keeps the
three dimensions the response left open. The RGA rewrites with those in hand.

```
RGA writes  ->  DGA proposes what the question left unsaid
            ->  ranker picks the 3 gaps the answer left open
            ->  RGA rewrites
```

Three domains: `code`, `medical`, `pwab` (product recommendation).

## Terminology

| term | computed from | meaning |
|---|---|---|
| **explicit dimensions** | query alone | what the query *does* state |
| **implicit dimensions** | query alone | what the query *leaves unspecified* |
| **system dimensions** | query + response | what the response explicitly did — framing, scoping, deferral |

Implicit dimensions are a gap in the *question*, not in the answer. Which of
them the answer failed to cover is decided later, by the ranker.

## Layout

```
prompts/{domain}/  dga_explicit_implicit.txt   DGA user pass (query only)
                   dga_system.txt              DGA system pass (query + response)
                   rga_rewrite.txt             RGA rewrite
                   teacher_label.txt           teacher distillation prompt
prompts/judge.txt                              GPT judge, shared across domains

src/common/        io.py llm.py gpt_api.py domains.py
src/data/          label_gpt.py build_rga.py build_dga.py
                   shard.py migrate_dims_keys.py
src/train/         dataset_info.json rga.yaml dga.yaml train.sh
src/infer/         rga_base.py dga_user.py dga_system.py rga_rewrite.py
src/rank/          ranker.py
src/eval/          judge.py summarize.py

configs/{domain}.yaml   paths, checkpoints, ranker lambdas, generation lengths
data/download.sh        pulls the processed release from the Hugging Face Hub
data/samples/{domain}/  5 examples per domain, every stage's artefacts present
run_pipeline.sh         end-to-end inference + evaluation for one domain
```

Prompt slots are filled by substring replacement, not `str.format` — the
prompts embed literal JSON schemas full of braces.

## Quickstart

```bash
pip install -r requirements.txt
export OPENAI_API_KEY=...            # teacher + judge only
# export OPENAI_BASE_URL=...         # any OpenAI-compatible endpoint

bash run_pipeline.sh code            # runs against data/samples/code
```

`configs/{domain}.yaml` points at `data/samples/{domain}` so every stage is
runnable without a download. For the full release:

```bash
bash data/download.sh                     # dataset -> data/release/{domain}/
bash data/download.sh --with-models code  # + that domain's two adapters
# then set data_root: data/release/code in configs/code.yaml
```

The released checkpoints are **LoRA adapters** (~50 MB each), not merged 8B
weights — the base model is public and `src/common/llm.py` applies the adapter
and merges it in memory at load time, so nothing downstream has to care. All
six live in one Hub repo, `itsgupta/proper-agents-models`, one subfolder per
agent (`{domain}/rga`, `{domain}/dga`), so a single domain's pair can be fetched
without the rest. They land at `saves/{domain}/{rga,dga}` — also where
`src/train/train.sh` writes.

`models.rga` / `models.dga` in a config accept either an adapter directory or a
standalone merged checkpoint; `models.base` is only consulted for the former.

## Pipeline

### Training

Two independent branches off `raw/train.jsonl`.

**RGA** — no teacher involved. `src/data/build_rga.py` emits Alpaca-format
`query -> answer` pairs; the response is wrapped in `===START===` / `===END===`.

**DGA** — distilled from a teacher (GPT-5-nano in the paper).
`src/data/label_gpt.py` labels each example with
`{"user_aspects": [...], "solution_aspects": [...]}`, each aspect a
`{name, value, justification}`. The teacher sees the *gold* response, which the
student never will — privileged-information distillation.
`src/data/build_dga.py` then joins those labels to the raw split and emits two
rows per example, a `user` row and a `system` row, into one file and one
checkpoint.

> **The train/inference prompt gap is deliberate.** Training targets
> `user_aspects` / `solution_aspects`; at inference the model is prompted with
> `dga_explicit_implicit.txt` and asked for
> `explicit_dimensions` / `implicit_dimensions`. SFT teaches the model what a
> dimension *is*; the inference prompt reshapes the output. Do not "fix" this.

```bash
bash src/train/train.sh code data/release    # LoRA SFT, both agents
```

LoRA r=8, α=16, lr 1e-4, 1 epoch, cosine with 0.1 warmup, batch 1 × grad-accum
8, bf16, on `meta-llama/Meta-Llama-3-8B-Instruct`. Adapters are written to
`saves/{domain}/{rga,dga}`; there is no merge step.

### Inference

| stage | module | output |
|---|---|---|
| 1 | `src.infer.rga_base` | `raw/{domain}_test_rga_preds.jsonl` |
| 2 | `src.infer.dga_user` | `dims/user/{id}.json` (+ per-token logprobs) |
| 3 | `src.infer.dga_system` | `dims/system/{id}.txt` |
| 4 | `src.infer.rga_rewrite` | `outputs/llm_outputs.pkl` |
| 5 | `src.eval.judge` | `outputs/results.pkl` |
| 6 | `src.eval.summarize` | win rates and mean scores |

Stages 2 and 3 are the same checkpoint run twice under different prompts.

### Ranking

Every user dimension (explicit + implicit) is embedded with `bge-large-en-v1.5`
(mean-pooled last hidden state over `f"{name}: {value}"`) and scored, all terms
z-normalised:

```
score = z(logprob) − λ1 · z(max cos-sim to any system dim)
                   − λ2 · z(max cos-sim to any other user dim)
```

- `logprob` — the DGA's own confidence in that dimension, recovered by matching
  `"value": … , "justification"` spans in the raw generation back onto the
  per-token logprobs (`SequenceMatcher`, `min_ratio=0.7`)
- `λ1` — suppresses dimensions the response already covered, which is what
  turns "everything the query left unsaid" into "the gaps the answer left open"
- `λ2` — suppresses near-duplicates

Top-3, with `λ1 = 8`, `λ2 = 1` (`configs/{domain}.yaml`).

## Data format

A user-pass record — one row of `dims/user.jsonl`, or one `dims/user/{id}.json`:

```json
{"id": "...", "persona": "...(pwab only)", "query": "...",
 "raw_text": "<full generation, markers included>",
 "json_data": {"explicit_dimensions": [...], "implicit_dimensions": [...]},
 "logprobs": [<one float per generated token>]}
```

Artefacts generated before the terminology change store `missed_dimensions`
instead of `implicit_dimensions`. `src.common.io.normalise_dim_keys` accepts
both; `src/data/migrate_dims_keys.py` rewrites them in place.

The released data ships the three dimension collections as JSONL shards
(`labels/teacher/{split}.jsonl`, `dims/user.jsonl`, `dims/system.jsonl`) —
one row per example, keyed by `id`. The generators instead write one file per
example so a crashed run can resume; `src/data/shard.py pack|unpack` converts
between the two and every loader reads either. Teacher rows are
`{id, input, gpt_output}`.

Coverage is partial where a generation failed to parse. That is expected and
logged; downstream stages skip those ids.

## Credentials

No keys are stored in this repository. `OPENAI_API_KEY` is read from the
environment and `OPENAI_BASE_URL` selects the endpoint, so any
OpenAI-compatible provider works. Only the teacher and judge stages need it.

## Citation

If you use this code or data, please cite:

```bibtex
@inproceedings{kaur2026proper,
  title={PROPER Agents: Proactivity Driven Personalized Agents for Advancing Knowledge Gap Navigation},
  author={Kaur, Kirandeep and Gupta, Vinayak and Gupta, Aditya and Shah, Chirag},
  booktitle={Findings of the Association for Computational Linguistics: ACL 2026},
  pages={41951--41975},
  year={2026}
}
```
