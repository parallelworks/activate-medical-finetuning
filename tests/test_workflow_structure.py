from __future__ import annotations

from typing import Dict, List, Set

from workflow_utils import collect_inputs, load_workflow, normalize_needs


def _jobs(workflow: Dict) -> Dict:
    jobs = workflow.get("jobs")
    assert isinstance(jobs, dict), "workflow.jobs must be a mapping"
    return jobs


def test_top_level_sections_present() -> None:
    workflow = load_workflow()
    for key in ("permissions", "sessions", "on", "jobs"):
        assert key in workflow, f"Missing top-level key: {key}"


def test_permissions_allow_all() -> None:
    workflow = load_workflow()
    permissions = workflow.get("permissions")
    assert isinstance(permissions, list), "permissions must be a list"
    assert "*" in permissions, "permissions must include '*'"


def test_session_named_session_exists() -> None:
    workflow = load_workflow()
    sessions = workflow.get("sessions")
    assert isinstance(sessions, dict), "sessions must be a mapping"
    assert "session" in sessions, "sessions.session must be defined"


def test_required_input_groups_exist() -> None:
    workflow = load_workflow()
    inputs = workflow["on"]["execute"]["inputs"]
    input_paths = collect_inputs(inputs)
    required_groups = {
        "cluster",
        "scheduler",
        "model_selection",
        "infrastructure",
        "dataset_config",
        "training_hyperparameters",
        "lora_config",
        "tensorboard_config",
        "output_publishing",
    }
    missing = sorted(group for group in required_groups if group not in input_paths)
    assert not missing, f"Missing required input groups: {missing}"


def test_jobs_have_steps_and_actions() -> None:
    jobs = _jobs(load_workflow())
    for job_name, job in jobs.items():
        steps = job.get("steps", [])
        assert steps, f"Job {job_name} has no steps"
        for index, step in enumerate(steps):
            assert "name" in step, f"{job_name} step {index} missing name"
            assert (
                "run" in step or "uses" in step
            ), f"{job_name} step {index} must have run or uses"


def test_job_needs_reference_valid_jobs() -> None:
    workflow = load_workflow()
    jobs = _jobs(workflow)
    job_names = set(jobs.keys())
    missing: List[str] = []
    for job_name, job in jobs.items():
        for needed in normalize_needs(job.get("needs")):
            if needed not in job_names:
                missing.append(f"{job_name} -> {needed}")
    assert not missing, f"Invalid job needs references: {missing}"


def test_job_dependency_graph_acyclic() -> None:
    jobs = _jobs(load_workflow())
    job_names: Set[str] = set(jobs.keys())
    edges = {name: set(normalize_needs(job.get("needs"))) for name, job in jobs.items()}
    in_degree = {name: 0 for name in job_names}
    for name in job_names:
        for needed in edges[name]:
            if needed in in_degree:
                in_degree[name] += 1

    queue = [name for name, degree in in_degree.items() if degree == 0]
    visited = 0
    while queue:
        current = queue.pop()
        visited += 1
        for successor, needs in edges.items():
            if current in needs:
                in_degree[successor] -= 1
                if in_degree[successor] == 0:
                    queue.append(successor)

    assert visited == len(job_names), "Detected a cycle in job dependencies"
