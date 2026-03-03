"""Simple hierarchical TODO helper.

Stores a lightweight JSON file (default: .todo_hierarchy.json) with tasks
and subtasks. Provides helpers to add tasks/subtasks and flatten the
hierarchy into the format expected by the `manage_todo_list` tool.

This module avoids external deps and is intended as a small convenience
layer for maintaining human-friendly sub-tasks in the repo.
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional


DEFAULT_PATH = ".todo_hierarchy.json"


def _load(path: str = DEFAULT_PATH) -> Dict[str, Any]:
    if not os.path.exists(path):
        return {"tasks": []}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _save(state: Dict[str, Any], path: str = DEFAULT_PATH) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)


def _next_id(state: Dict[str, Any]) -> int:
    ids = [t.get("id", 0) for t in state.get("tasks", [])]
    return max(ids, default=0) + 1


def add_task(title: str, status: str = "not-started", path: str = DEFAULT_PATH) -> int:
    state = _load(path)
    task_id = _next_id(state)
    state.setdefault("tasks", []).append({"id": task_id, "title": title, "status": status, "subtasks": []})
    _save(state, path)
    return task_id


def add_subtask(parent_id: int, title: str, status: str = "not-started", path: str = DEFAULT_PATH) -> Optional[int]:
    state = _load(path)
    for task in state.get("tasks", []):
        if task.get("id") == parent_id:
            subtasks = task.setdefault("subtasks", [])
            # subtask id is local to parent; we don't need a global id here
            sub_id = len(subtasks) + 1
            subtasks.append({"id": sub_id, "title": title, "status": status})
            _save(state, path)
            return sub_id
    return None


def list_hierarchy(path: str = DEFAULT_PATH) -> Dict[str, Any]:
    return _load(path)


def flatten_for_manage_tool(path: str = DEFAULT_PATH) -> List[Dict[str, Any]]:
    """Return a flat list of todos as required by manage_todo_list.

    Each item is a dict with keys: id (int), title (str), status (one of
    'not-started', 'in-progress', 'completed'). Subtasks are assigned new
    sequential integer ids and their titles are prefixed to indicate
    hierarchy (e.g. "↳ Subtask title").
    """
    state = _load(path)
    out: List[Dict[str, Any]] = []
    next_id = 1
    for task in state.get("tasks", []):
        title = task.get("title", "Untitled")
        status = task.get("status", "not-started")
        out.append({"id": next_id, "title": title, "status": status})
        next_id += 1
        for sub in task.get("subtasks", []):
            stitle = f"↳ {sub.get('title', 'Untitled')}"
            sstatus = sub.get("status", "not-started")
            out.append({"id": next_id, "title": stitle, "status": sstatus})
            next_id += 1
    return out


if __name__ == "__main__":
    print(json.dumps(_load(), ensure_ascii=False, indent=2))
