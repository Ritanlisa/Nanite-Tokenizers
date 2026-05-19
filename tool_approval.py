from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Optional

logger = logging.getLogger(__name__)

_approval_events: dict[str, asyncio.Event] = {}
_approval_data: dict[str, dict[str, Any]] = {}
_session_pending_call: dict[str, str] = {}

_stop_events: dict[str, asyncio.Event] = {}

_running_tasks: dict[str, asyncio.Task] = {}

AUTO_APPROVE_SESSIONS: dict[str, bool] = {}

TOOLS_REQUIRING_APPROVAL = {
    "skill_shell",
}


def is_auto_approve(session_id: str) -> bool:
    val = AUTO_APPROVE_SESSIONS.get(session_id)
    return bool(val)


def set_auto_approve(session_id: str, enabled: bool) -> None:
    AUTO_APPROVE_SESSIONS[session_id] = enabled


def register_running_task(session_id: str, task: asyncio.Task) -> None:
    _running_tasks[session_id] = task


def unregister_running_task(session_id: str) -> None:
    _running_tasks.pop(session_id, None)
    _stop_events.pop(session_id, None)


async def request_stop_generation(session_id: str) -> bool:
    task = _running_tasks.get(session_id)
    if task and not task.done():
        task.cancel()
        unregister_running_task(session_id)
        return True
    return False


async def request_tool_approval(
    session_id: str,
    call_id: str,
    tool_name: str,
    tool_args: dict[str, Any],
    timeout: float = 300.0,
) -> bool:
    if tool_name not in TOOLS_REQUIRING_APPROVAL:
        return True

    if is_auto_approve(session_id):
        logger.info(
            "Auto-approve enabled for session=%s tool=%s", session_id, tool_name
        )
        return True

    event = asyncio.Event()
    _approval_events[call_id] = event
    _approval_data[call_id] = {
        "session_id": session_id,
        "tool_name": tool_name,
        "tool_args": dict(tool_args),
        "created_at": time.time(),
        "approved": False,
    }
    _session_pending_call[session_id] = call_id
    logger.info(
        "Awaiting tool approval: session=%s call_id=%s tool=%s",
        session_id,
        call_id,
        tool_name,
    )
    try:
        await asyncio.wait_for(event, timeout=timeout)
        result = _approval_data.get(call_id, {}).get("approved", False)
        logger.info(
            "Tool approval result: session=%s call_id=%s tool=%s approved=%s",
            session_id,
            call_id,
            tool_name,
            result,
        )
        return bool(result)
    except asyncio.TimeoutError:
        logger.warning(
            "Tool approval timed out: session=%s call_id=%s tool=%s",
            session_id,
            call_id,
            tool_name,
        )
        return False
    finally:
        _approval_events.pop(call_id, None)
        _approval_data.pop(call_id, None)
        if _session_pending_call.get(session_id) == call_id:
            _session_pending_call.pop(session_id, None)


def resolve_tool_approval(call_id: str, approved: bool) -> bool:
    event = _approval_events.get(call_id)
    if event is None:
        logger.warning("No pending approval found for call_id=%s", call_id)
        return False
    if call_id in _approval_data:
        _approval_data[call_id]["approved"] = approved
    event.set()
    return True


def get_pending_approval_info(
    call_id: str,
) -> Optional[dict[str, Any]]:
    data = _approval_data.get(call_id)
    if data is None:
        return None
    return {
        "call_id": call_id,
        "session_id": data["session_id"],
        "tool_name": data["tool_name"],
        "tool_args": data["tool_args"],
        "created_at": data["created_at"],
    }


def get_pending_approval_for_session(
    session_id: str,
) -> Optional[dict[str, Any]]:
    call_id = _session_pending_call.get(session_id)
    if not call_id:
        return None
    return get_pending_approval_info(call_id)
