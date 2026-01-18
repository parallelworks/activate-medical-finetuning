from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, Iterable, List, Set, Tuple

import yaml


WORKFLOW_PATH = Path(__file__).resolve().parents[1] / "workflow.yaml"

TEMPLATE_RE = re.compile(r"\${{(.*?)}}", re.DOTALL)
INPUT_REF_RE = re.compile(r"\binputs\.([A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z0-9_]+)*)")
NEEDS_REF_RE = re.compile(r"\bneeds\.([A-Za-z0-9_-]+)\b")


class NoBoolSafeLoader(yaml.SafeLoader):
    """Avoid YAML 1.1 bool coercion (e.g., 'on', 'off') for workflow keys."""


for ch, resolvers in list(NoBoolSafeLoader.yaml_implicit_resolvers.items()):
    NoBoolSafeLoader.yaml_implicit_resolvers[ch] = [
        (tag, regexp) for tag, regexp in resolvers if tag != "tag:yaml.org,2002:bool"
    ]

NoBoolSafeLoader.add_implicit_resolver(
    "tag:yaml.org,2002:bool",
    re.compile(r"^(?:true|false)$", re.IGNORECASE),
    list("tTfF"),
)


def load_workflow() -> Dict:
    with WORKFLOW_PATH.open("r", encoding="utf-8") as handle:
        return yaml.load(handle, Loader=NoBoolSafeLoader)


def iter_strings(node) -> Iterable[str]:
    if isinstance(node, dict):
        for value in node.values():
            yield from iter_strings(value)
    elif isinstance(node, list):
        for value in node:
            yield from iter_strings(value)
    elif isinstance(node, str):
        yield node


def collect_inputs(inputs_node: Dict, prefix: str = "") -> Dict[str, Dict]:
    collected: Dict[str, Dict] = {}
    for name, spec in inputs_node.items():
        path = f"{prefix}.{name}" if prefix else name
        collected[path] = spec
        if isinstance(spec, dict) and spec.get("type") == "group":
            items = spec.get("items") or {}
            collected.update(collect_inputs(items, path))
    return collected


def collect_template_refs(workflow: Dict) -> Tuple[Set[str], Set[str]]:
    inputs_refs: Set[str] = set()
    needs_refs: Set[str] = set()
    for text in iter_strings(workflow):
        for expr in TEMPLATE_RE.findall(text):
            inputs_refs.update(match.group(1) for match in INPUT_REF_RE.finditer(expr))
            needs_refs.update(match.group(1) for match in NEEDS_REF_RE.finditer(expr))
    return inputs_refs, needs_refs


def normalize_needs(value) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]
