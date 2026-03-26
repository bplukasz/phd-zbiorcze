#!/usr/bin/env python3
"""Simple JSON-backed queue manager for e001-02 experiments with tsp."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import shlex
import subprocess
import tempfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_QUEUE_PATH = ROOT / "queues" / "e001_02_queue.json"
RUN_SCRIPT = ROOT / "scripts" / "run_e001_02.sh"


class QueueError(RuntimeError):
    """Raised when queue file content is invalid."""


def now_iso() -> str:
    return dt.datetime.now().replace(microsecond=0).isoformat()


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def default_queue() -> dict[str, Any]:
    return {"version": 1, "tasks": []}


def normalize_overrides(raw: Any) -> list[str]:
    if raw is None:
        return []
    if isinstance(raw, list):
        return [str(x) for x in raw if str(x).strip()]
    if isinstance(raw, str):
        return [x for x in shlex.split(raw) if x.strip()]
    raise QueueError("Task field 'overrides' must be string or list of strings")


def normalize_task(raw: dict[str, Any], index: int) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise QueueError(f"Task at index {index} is not an object")

    task = dict(raw)
    task.setdefault("id", f"task-{index + 1:04d}")
    task.setdefault("profile", "")
    task.setdefault("data_dir", "")
    task["overrides"] = normalize_overrides(task.get("overrides", []))

    status = str(task.get("status", "pending")).lower()
    if status not in {"pending", "queued", "running", "success", "failed"}:
        status = "pending"
    task["status"] = status

    tsp_id = task.get("tsp_id")
    if tsp_id in (None, "", "null"):
        task["tsp_id"] = None
    else:
        try:
            task["tsp_id"] = int(str(tsp_id))
        except ValueError as exc:
            raise QueueError(f"Invalid tsp_id value: {tsp_id!r}") from exc

    task.setdefault("created_at", now_iso())
    task.setdefault("submitted_at", None)
    task.setdefault("finished_at", None)
    task.setdefault("notes", "")

    return task


def load_queue(path: Path) -> dict[str, Any]:
    if not path.exists():
        return default_queue()

    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)

    if not isinstance(data, dict):
        raise QueueError("Queue file root must be an object")
    tasks = data.get("tasks", [])
    if not isinstance(tasks, list):
        raise QueueError("Queue file field 'tasks' must be an array")

    normalized = [normalize_task(task, i) for i, task in enumerate(tasks)]
    return {"version": int(data.get("version", 1)), "tasks": normalized}


def save_queue(path: Path, data: dict[str, Any]) -> None:
    ensure_parent(path)
    with tempfile.NamedTemporaryFile("w", delete=False, dir=str(path.parent), encoding="utf-8") as tmp:
        json.dump(data, tmp, indent=2)
        tmp.write("\n")
        tmp_path = Path(tmp.name)
    os.replace(tmp_path, path)


def next_task_id(tasks: list[dict[str, Any]]) -> str:
    prefix = dt.datetime.now().strftime("exp-%Y%m%d-%H%M%S")
    existing = {str(task.get("id", "")) for task in tasks}
    seq = 1
    while True:
        candidate = f"{prefix}-{seq:03d}"
        if candidate not in existing:
            return candidate
        seq += 1


def run_cmd(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, check=True, capture_output=True, text=True)


def parse_tsp_submit_id(stdout: str) -> int:
    matches = re.findall(r"\b\d+\b", stdout)
    if not matches:
        raise QueueError(f"Cannot parse tsp task id from output: {stdout!r}")
    return int(matches[-1])


def tsp_set_slots(slots: int) -> None:
    run_cmd(["tsp", "-S", str(slots)])


def build_tsp_command(task: dict[str, Any]) -> list[str]:
    data_dir = str(task.get("data_dir") or "")
    cmd = [
        "tsp",
        "bash",
        str(RUN_SCRIPT),
        str(task["profile"]),
        data_dir,
    ]
    cmd.extend(task.get("overrides", []))
    return cmd


def parse_tsp_list() -> dict[int, dict[str, Any]]:
    proc = run_cmd(["tsp", "-l"])
    states: dict[int, dict[str, Any]] = {}
    for line in proc.stdout.splitlines():
        line = line.strip()
        if not line or not re.match(r"^\d+\s+", line):
            continue
        parts = re.split(r"\s+", line, maxsplit=6)
        if len(parts) < 2:
            continue
        tsp_id = int(parts[0])
        state = parts[1].lower()
        e_level = None
        if len(parts) > 3 and re.match(r"^-?\d+$", parts[3]):
            e_level = int(parts[3])
        states[tsp_id] = {"state": state, "e_level": e_level}
    return states


def cmd_init(args: argparse.Namespace) -> int:
    path = Path(args.queue_file)
    if path.exists() and not args.force:
        print(f"Queue already exists: {path}")
        return 0
    save_queue(path, default_queue())
    print(f"Created queue file: {path}")
    return 0


def cmd_add(args: argparse.Namespace) -> int:
    path = Path(args.queue_file)
    queue = load_queue(path)

    task = {
        "id": next_task_id(queue["tasks"]),
        "profile": args.profile,
        "data_dir": args.data_dir or "",
        "overrides": normalize_overrides(args.override),
        "status": "pending",
        "tsp_id": None,
        "created_at": now_iso(),
        "submitted_at": None,
        "finished_at": None,
        "notes": args.notes or "",
    }
    queue["tasks"].append(task)
    save_queue(path, queue)

    print(f"Added task {task['id']} ({task['profile']})")
    return 0


def cmd_enqueue(args: argparse.Namespace) -> int:
    path = Path(args.queue_file)
    queue = load_queue(path)

    if args.slots is not None:
        tsp_set_slots(args.slots)

    submitted = 0
    for task in queue["tasks"]:
        if task.get("status") != "pending" or task.get("tsp_id") is not None:
            continue

        cmd = build_tsp_command(task)
        if args.dry_run:
            print("DRY-RUN:", shlex.join(cmd))
            continue

        proc = run_cmd(cmd)
        tsp_id = parse_tsp_submit_id(proc.stdout)

        task["tsp_id"] = tsp_id
        task["status"] = "queued"
        task["submitted_at"] = now_iso()
        submitted += 1
        print(f"Submitted {task['id']} -> tsp#{tsp_id}")

    save_queue(path, queue)
    print(f"Submitted tasks: {submitted}")
    return 0


def cmd_sync(args: argparse.Namespace) -> int:
    path = Path(args.queue_file)
    queue = load_queue(path)
    tsp_states = parse_tsp_list()

    changed = 0
    for task in queue["tasks"]:
        tsp_id = task.get("tsp_id")
        if tsp_id is None:
            continue
        tsp_entry = tsp_states.get(int(tsp_id))
        if tsp_entry is None:
            continue

        state = tsp_entry["state"]
        old = task["status"]
        if state in {"queued", "hold"}:
            task["status"] = "queued"
        elif state == "running":
            task["status"] = "running"
        elif state == "finished":
            if tsp_entry.get("e_level") == 0:
                task["status"] = "success"
            else:
                task["status"] = "failed"
            if task.get("finished_at") is None:
                task["finished_at"] = now_iso()

        if task["status"] != old:
            changed += 1

    save_queue(path, queue)

    if args.show:
        print_status_table(queue)
    else:
        print(f"Synced queue. Updated tasks: {changed}")
    return 0


def cmd_retry_failed(args: argparse.Namespace) -> int:
    path = Path(args.queue_file)
    queue = load_queue(path)
    reset = 0
    for task in queue["tasks"]:
        if task.get("status") == "failed":
            task["status"] = "pending"
            task["tsp_id"] = None
            task["submitted_at"] = None
            task["finished_at"] = None
            reset += 1

    save_queue(path, queue)
    print(f"Marked failed tasks as pending: {reset}")
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    path = Path(args.queue_file)
    queue = load_queue(path)
    print_status_table(queue)
    return 0


def print_status_table(queue: dict[str, Any]) -> None:
    tasks = queue["tasks"]
    if not tasks:
        print("Queue is empty.")
        return

    header = f"{'id':28} {'profile':30} {'status':8} {'tsp_id':6} {'submitted_at':19}"
    print(header)
    print("-" * len(header))
    for task in tasks:
        tsp_id = "-" if task.get("tsp_id") is None else str(task["tsp_id"])
        submitted = (task.get("submitted_at") or "-")[:19]
        print(
            f"{str(task.get('id', '-'))[:28]:28} "
            f"{str(task.get('profile', '-'))[:30]:30} "
            f"{str(task.get('status', '-'))[:8]:8} "
            f"{tsp_id[:6]:6} "
            f"{submitted:19}"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--queue-file",
        default=str(DEFAULT_QUEUE_PATH),
        help=f"Path to queue JSON file (default: {DEFAULT_QUEUE_PATH})",
    )

    sub = parser.add_subparsers(dest="command", required=True)

    p_init = sub.add_parser("init", help="Create queue file if missing")
    p_init.add_argument("--force", action="store_true", help="Overwrite queue file if it exists")
    p_init.set_defaults(func=cmd_init)

    p_add = sub.add_parser("add", help="Add a pending experiment task")
    p_add.add_argument("--profile", required=True, help="e001-02 profile name")
    p_add.add_argument("--data-dir", default="", help="Optional DATA_DIR value")
    p_add.add_argument(
        "--override",
        default="",
        help="Optional overrides string, e.g. 'steps=5000 batch_size=64'",
    )
    p_add.add_argument("--notes", default="", help="Optional free-text note")
    p_add.set_defaults(func=cmd_add)

    p_enqueue = sub.add_parser("enqueue", help="Submit pending tasks to tsp")
    p_enqueue.add_argument("--slots", type=int, default=1, help="Set tsp worker slots before submit")
    p_enqueue.add_argument("--dry-run", action="store_true", help="Only print commands, do not submit")
    p_enqueue.set_defaults(func=cmd_enqueue)

    p_sync = sub.add_parser("sync", help="Sync task statuses from 'tsp -l'")
    p_sync.add_argument("--show", action="store_true", help="Print status table after sync")
    p_sync.set_defaults(func=cmd_sync)

    p_status = sub.add_parser("status", help="Show queue table")
    p_status.set_defaults(func=cmd_status)

    p_retry = sub.add_parser("retry-failed", help="Move failed tasks back to pending")
    p_retry.set_defaults(func=cmd_retry_failed)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())



