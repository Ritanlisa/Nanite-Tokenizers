from __future__ import annotations

import asyncio
import json
import logging
import subprocess
from typing import Any

from pydantic import BaseModel, Field

import config
from tool_usage import end_current_tool_call, start_current_tool_call

from ._common import InputSugarTool, _bi, _safe_path, _t, _workspace_root

logger = logging.getLogger(__name__)

class ShellInput(BaseModel):
    command: str = Field(description=_bi("要执行的 Shell 命令", "Shell command to execute"))
    timeout: int = Field(default=20, ge=1, le=120, description=_bi("执行超时时间（秒）", "Execution timeout in seconds"))

class ShellTool(InputSugarTool):
    name: str = "skill_shell"
    description: str = _bi("在工作区执行 Shell 命令，并返回标准输出/标准错误。", "Execute shell commands in the workspace and return stdout/stderr.")
    args_schema: Any = ShellInput

    async def _arun(self, command: str, timeout: int = 20) -> str:
        call_id = start_current_tool_call(self.name, {"command": command, "timeout": timeout})
        output_text = ""
        try:
            if not config.settings.ENABLE_SHELL_SKILL:
                output_text = _t("Shell 工具已被配置禁用。", "Shell tool is disabled by configuration.")
                return output_text

            command = command.strip()
            if not command:
                output_text = _t("缺少命令参数。", "Missing command argument.")
                return output_text

            blocked_tokens = [
                "rm -rf /", "shutdown", "reboot", "mkfs", "dd if=", ":(){:|:&};:"
            ]
            lower_command = command.lower()
            if any(token in lower_command for token in blocked_tokens):
                output_text = _t("命令被安全策略拒绝。", "Command rejected by safety policy.")
                return output_text

            def _run_command():
                return subprocess.run(
                    command,
                    shell=True,
                    cwd=str(_workspace_root()),
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                )

            proc = await asyncio.to_thread(_run_command)
            stdout = (proc.stdout or "").strip()
            stderr = (proc.stderr or "").strip()
            payload = {
                "exit_code": proc.returncode,
                "stdout": stdout[: config.settings.SKILL_OUTPUT_MAX_CHARS],
                "stderr": stderr[: config.settings.SKILL_OUTPUT_MAX_CHARS],
            }
            output_text = json.dumps(payload, ensure_ascii=False)
            return output_text
        except subprocess.TimeoutExpired:
            output_text = _t(f"命令执行超时（{timeout}秒）。", f"Command timed out ({timeout}s).")
            return output_text
        except Exception as exc:
            logger.exception("shell tool failed")
            if config.settings.ENV == "prod":
                output_text = _t("Shell 工具执行失败。", "Shell tool execution failed.")
            else:
                output_text = _t(
                    f"Shell 工具执行失败: {type(exc).__name__}",
                    f"Shell tool execution failed: {type(exc).__name__}",
                )
            return output_text
        finally:
            end_current_tool_call(call_id, output_text)

    def _run(self, **kwargs) -> str:
        raise NotImplementedError("Use async call")

class FileIOInput(BaseModel):
    action: str = Field(description=_bi("操作类型：read、write、append、list、mkdir", "Action type: read, write, append, list, mkdir"))
    path: str = Field(description=_bi("相对于工作区根目录的路径", "Path relative to workspace root"))
    content: str = Field(default="", description=_bi("write/append 时要写入的内容", "Content to write/append"))

class FileIOTool(InputSugarTool):
    name: str = "skill_file_io"
    description: str = _bi("在工作区根目录下读/写/追加文件，并列出/创建目录。", "Read/write/append files and list/create directories under workspace root.")
    args_schema: Any = FileIOInput

    async def _arun(self, action: str, path: str, content: str = "") -> str:
        call_id = start_current_tool_call(self.name, {"action": action, "path": path, "content": content})
        output_text = ""
        try:
            if not config.settings.ENABLE_FILE_IO_SKILL:
                output_text = _t("文件 IO 工具已被配置禁用。", "File IO tool is disabled by configuration.")
                return output_text

            action = action.strip().lower()
            path = path.strip()
            if not action or not path:
                output_text = _t("缺少操作类型或路径参数。", "Missing action or path argument.")
                return output_text

            target = _safe_path(path)

            if action == "read":
                if not target.exists() or not target.is_file():
                    output_text = _t("文件不存在。", "File does not exist.")
                    return output_text
                if target.stat().st_size > config.settings.SKILL_MAX_FILE_BYTES:
                    output_text = _t("文件过大，无法通过工具读取。", "File is too large to read via tool.")
                    return output_text
                text = await asyncio.to_thread(target.read_text, "utf-8")
                output_text = text[: config.settings.SKILL_OUTPUT_MAX_CHARS]
                return output_text

            if action == "write":
                await asyncio.to_thread(target.parent.mkdir, parents=True, exist_ok=True)
                await asyncio.to_thread(target.write_text, content, "utf-8")
                output_text = _t(
                    f"已写入 {len(content)} 个字符到 {target.relative_to(_workspace_root())}",
                    f"Wrote {len(content)} characters to {target.relative_to(_workspace_root())}",
                )
                return output_text

            if action == "append":
                await asyncio.to_thread(target.parent.mkdir, parents=True, exist_ok=True)
                with target.open("a", encoding="utf-8") as handle:
                    handle.write(content)
                output_text = _t(
                    f"已追加 {len(content)} 个字符到 {target.relative_to(_workspace_root())}",
                    f"Appended {len(content)} characters to {target.relative_to(_workspace_root())}",
                )
                return output_text

            if action == "list":
                if target.exists() and target.is_file():
                    output_text = target.name
                    return output_text
                if not target.exists():
                    output_text = _t("路径不存在。", "Path does not exist.")
                    return output_text
                names = sorted(item.name + ("/" if item.is_dir() else "") for item in target.iterdir())
                output_text = "\n".join(names[:200])
                return output_text

            if action == "mkdir":
                await asyncio.to_thread(target.mkdir, parents=True, exist_ok=True)
                output_text = _t(
                    f"目录已就绪：{target.relative_to(_workspace_root())}",
                    f"Directory is ready: {target.relative_to(_workspace_root())}",
                )
                return output_text

            output_text = _t("不支持的操作类型。", "Unsupported action type.")
            return output_text
        except ValueError as exc:
            output_text = str(exc)
            return output_text
        except Exception as exc:
            logger.exception("file io tool failed")
            if config.settings.ENV == "prod":
                output_text = _t("文件 IO 工具执行失败。", "File IO tool execution failed.")
            else:
                output_text = _t(
                    f"文件 IO 工具执行失败: {type(exc).__name__}",
                    f"File IO tool execution failed: {type(exc).__name__}",
                )
            return output_text
        finally:
            end_current_tool_call(call_id, output_text)

    def _run(self, **kwargs) -> str:
        raise NotImplementedError("Use async call")
