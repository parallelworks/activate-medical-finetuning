from __future__ import annotations

from typing import Dict, List, Set, Tuple

from workflow_utils import (
    collect_inputs,
    collect_template_refs,
    load_workflow,
)


ALLOWED_RESOURCE_ATTRS = {
    "cluster": {"schedulerType", "ip", "id", "name"},
}


def _inputs(workflow: Dict) -> Dict:
    return workflow["on"]["execute"]["inputs"]


def _jobs(workflow: Dict) -> Dict:
    jobs = workflow.get("jobs")
    assert isinstance(jobs, dict), "workflow.jobs must be a mapping"
    return jobs


def _validate_input_ref(path: str, input_paths: Set[str]) -> Tuple[bool, str]:
    if path in input_paths:
        return True, ""
    parts = path.split(".")
    if not parts:
        return False, path
    base = parts[0]
    if base in ALLOWED_RESOURCE_ATTRS and len(parts) == 2:
        if parts[1] in ALLOWED_RESOURCE_ATTRS[base]:
            return True, ""
    return False, path


def test_template_input_references_are_valid() -> None:
    workflow = load_workflow()
    inputs = _inputs(workflow)
    input_paths = set(collect_inputs(inputs).keys())
    input_refs, _ = collect_template_refs(workflow)

    invalid: List[str] = []
    for ref in sorted(input_refs):
        ok, _ = _validate_input_ref(ref, input_paths)
        if not ok:
            invalid.append(ref)

    assert not invalid, f"Unknown inputs references: {invalid}"


def test_template_needs_references_are_valid() -> None:
    workflow = load_workflow()
    jobs = _jobs(workflow)
    job_names = set(jobs.keys())
    _, needs_refs = collect_template_refs(workflow)
    invalid = sorted(ref for ref in needs_refs if ref not in job_names)
    assert not invalid, f"Unknown needs references: {invalid}"
