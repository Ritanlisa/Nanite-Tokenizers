"""Small CLI to manipulate hierarchical todos stored at .todo_hierarchy.json.

Usage examples:
  python scripts/todo_cli.py add "Implement feature X"
  python scripts/todo_cli.py add-sub 1 "Write tests for X"
  python scripts/todo_cli.py show
  python scripts/todo_cli.py export

`export` prints a flattened JSON list that can be directly passed to the
`manage_todo_list` tool in the assistant workflow.
"""
from __future__ import annotations

import json
import sys
from typing import List

from agent import todo_helper


def usage() -> None:
    print(__doc__)


def cmd_add(args: List[str]) -> None:
    if not args:
        print("Missing title")
        return
    title = args[0]
    tid = todo_helper.add_task(title)
    print(f"Added task {tid}: {title}")


def cmd_add_sub(args: List[str]) -> None:
    if len(args) < 2:
        print("Usage: add-sub <parent_id> <title>")
        return
    parent = int(args[0])
    title = args[1]
    sid = todo_helper.add_subtask(parent, title)
    if sid is None:
        print(f"Parent {parent} not found")
    else:
        print(f"Added subtask {sid} under {parent}: {title}")


def cmd_show(_: List[str]) -> None:
    print(json.dumps(todo_helper.list_hierarchy(), ensure_ascii=False, indent=2))


def cmd_export(_: List[str]) -> None:
    flat = todo_helper.flatten_for_manage_tool()
    print(json.dumps(flat, ensure_ascii=False, indent=2))


def main(argv: List[str]) -> None:
    if not argv or argv[0] in {"-h", "--help"}:
        usage()
        return
    cmd = argv[0]
    args = argv[1:]
    if cmd == "add":
        cmd_add(args)
    elif cmd == "add-sub":
        cmd_add_sub(args)
    elif cmd == "show":
        cmd_show(args)
    elif cmd == "export":
        cmd_export(args)
    else:
        usage()


if __name__ == "__main__":
    main(sys.argv[1:])
