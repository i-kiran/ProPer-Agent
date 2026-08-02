"""Filesystem, JSON and config helpers shared by every stage of the pipeline."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import yaml

log = logging.getLogger(__name__)

START, END = "===START===", "===END==="
START_JSON, END_JSON = "===START_JSON===", "===END_JSON==="


# ---------------------------------------------------------------- jsonl / text

def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_text(path):
    return Path(path).read_text(encoding="utf-8")


def write_text(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


# ------------------------------------------------------------ marker handling

def extract_between(text, start=START, end=END):
    """Return the span between two markers, or None if either is missing."""
    i = text.find(start)
    if i == -1:
        return None
    i += len(start)
    j = text.find(end, i)
    if j == -1:
        return None
    return text[i:j].strip()


def extract_json_block(text):
    """Parse the JSON payload out of a model or teacher generation.

    Tolerates ===START_JSON===/===END_JSON=== markers, ``` fences, and bare
    JSON with leading prose. Returns None when nothing parses.
    """
    candidate = extract_between(text, START_JSON, END_JSON)
    if candidate is None:
        candidate = text.strip()

    if candidate.startswith("```"):
        candidate = candidate.split("\n", 1)[-1]
        if candidate.rstrip().endswith("```"):
            candidate = candidate.rstrip()[: -len("```")]

    for attempt in (candidate, _widest_braces(candidate)):
        if not attempt:
            continue
        try:
            return json.loads(attempt)
        except json.JSONDecodeError:
            continue
    return None


def _widest_braces(text):
    i, j = text.find("{"), text.rfind("}")
    return text[i : j + 1] if i != -1 and j > i else None


# ------------------------------------------------------------- key migration

#: The DGA user pass originally emitted `missed_dimensions`. The released name
#: is `implicit_dimensions` -- a gap in the *question*, computed from the query
#: alone. Loaders accept both so pre-migration artefacts keep working.
LEGACY_DIM_KEYS = {"missed_dimensions": "implicit_dimensions"}


def normalise_dim_keys(json_data):
    """Rename legacy dimension keys in place-safe fashion; returns a new dict."""
    if not isinstance(json_data, dict):
        return json_data
    out = dict(json_data)
    for old, new in LEGACY_DIM_KEYS.items():
        if old in out and new not in out:
            out[new] = out.pop(old)
        else:
            out.pop(old, None)
    return out


def user_dims(record):
    """All user-side dimensions (explicit + implicit) from a dims/user record."""
    data = normalise_dim_keys(record.get("json_data", {}))
    return list(data.get("explicit_dimensions", [])) + list(
        data.get("implicit_dimensions", [])
    )


# -------------------------------------------------------------------- config

@dataclass
class Paths:
    """Canonical on-disk layout of a domain's data directory."""

    root: Path
    domain: str

    @property
    def raw(self):
        return self.root / "raw"

    @property
    def train_jsonl(self):
        return self.raw / "train.jsonl"

    @property
    def test_jsonl(self):
        return self.raw / "test.jsonl"

    def rga_sft(self, split):
        return self.raw / f"{self.domain}_rga_{split}.jsonl"

    def dga_sft(self, split):
        return self.raw / f"{self.domain}_dga_{split}.jsonl"

    @property
    def rga_preds(self):
        return self.raw / f"{self.domain}_test_rga_preds.jsonl"

    # Each of the three artefact collections exists in two interchangeable
    # forms: a directory of per-example files (what the generators write, so a
    # crashed run can resume) and a single JSONL shard (what gets distributed,
    # because ~13k small files is past what the Hub is comfortable with).
    # The readers below prefer the shard and fall back to the directory.

    def teacher_labels(self, split):
        return self.root / "labels" / "teacher" / split

    def teacher_labels_shard(self, split):
        return self.root / "labels" / "teacher" / f"{split}.jsonl"

    @property
    def dims_user(self):
        return self.root / "dims" / "user"

    @property
    def dims_user_shard(self):
        return self.root / "dims" / "user.jsonl"

    @property
    def dims_system(self):
        return self.root / "dims" / "system"

    @property
    def dims_system_shard(self):
        return self.root / "dims" / "system.jsonl"

    @property
    def outputs(self):
        return self.root / "outputs"

    @property
    def rewrites_pkl(self):
        return self.outputs / "llm_outputs.pkl"

    @property
    def results_pkl(self):
        return self.outputs / "results.pkl"


# ------------------------------------------------------- artefact collections

TEACHER_INPUT_MARKER = "=== INPUT ==="
TEACHER_OUTPUT_MARKER = "=== GPT OUTPUT ==="


def split_teacher_label(text):
    """Split a teacher label file into its (input, output) blocks."""
    if TEACHER_OUTPUT_MARKER not in text:
        return None, None
    head, output = text.split(TEACHER_OUTPUT_MARKER, 1)
    return head.replace(TEACHER_INPUT_MARKER, "", 1).strip(), output.strip()


def iter_teacher_labels(paths, split):
    """Yield (id, gpt_output) for one split, from the shard or the directory."""
    shard = paths.teacher_labels_shard(split)
    if shard.is_file():
        for row in load_jsonl(shard):
            yield row["id"], row["gpt_output"]
        return

    directory = paths.teacher_labels(split)
    if not directory.is_dir():
        return
    for path in sorted(directory.glob("*.txt")):
        _, output = split_teacher_label(read_text(path))
        if output is None:
            log.warning("%s: no %s marker", path.name, TEACHER_OUTPUT_MARKER)
            continue
        yield path.stem, output


def iter_dims_user(paths):
    """Yield DGA user-pass records, from the shard or the directory."""
    if paths.dims_user_shard.is_file():
        yield from load_jsonl(paths.dims_user_shard)
        return
    if not paths.dims_user.is_dir():
        return
    for path in sorted(paths.dims_user.glob("*.json")):
        record = json.loads(read_text(path))
        record.setdefault("id", path.stem)
        yield record


def parse_system_dims(text, uid):
    """Parse one system-pass generation, or None if it is malformed.

    ~0.5% of generations are not a well-formed object; those ids are dropped.
    """
    try:
        payload = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        payload = None

    if not isinstance(payload, dict) or not isinstance(
        payload.get("system_explicit_dimensions"), list
    ):
        log.warning("%s: malformed system dimensions; skipping", uid)
        return None

    return {"id": uid, "system_explicit_dimensions": payload["system_explicit_dimensions"]}


def load_dims_system(paths):
    """Return {id: system record}. Small enough to hold in memory."""
    if paths.dims_system_shard.is_file():
        return {row["id"]: row for row in load_jsonl(paths.dims_system_shard)}
    if not paths.dims_system.is_dir():
        return {}

    out = {}
    for path in sorted(paths.dims_system.glob("*.txt")):
        record = parse_system_dims(read_text(path), path.stem)
        if record is not None:
            out[path.stem] = record
    return out


def load_config(path):
    cfg = yaml.safe_load(read_text(path))
    cfg["paths"] = Paths(root=Path(cfg["data_root"]), domain=cfg["domain"])
    cfg["prompt_dir"] = Path(cfg.get("prompt_dir", "prompts"))
    return cfg


def prompt(cfg, name):
    """Load a domain prompt by stem, e.g. prompt(cfg, 'rga_rewrite')."""
    if name == "judge":
        return read_text(cfg["prompt_dir"] / "judge.txt")
    return read_text(cfg["prompt_dir"] / cfg["domain"] / f"{name}.txt")


def render(template, **slots):
    """Fill ``{slot}`` placeholders by substring replacement.

    Not ``str.format``: the prompts embed literal JSON schemas full of braces,
    which format() would try to interpret.
    """
    for key, value in slots.items():
        template = template.replace("{" + key + "}", str(value))
    return template
