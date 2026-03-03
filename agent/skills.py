from __future__ import annotations

import asyncio
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field
import yaml

import config

logger = logging.getLogger(__name__)


@dataclass
class AgentSkill:
    name: str
    description: str
    args_schema: type[BaseModel]

    async def run(self, **kwargs: Any) -> str:
        raise NotImplementedError


def _workspace_root() -> Path:
    return Path(os.getcwd()).resolve()


# ------------------------------------------------------------------------------
# 新增：支持 Anthropic Agent Skill 格式的元工具
# ------------------------------------------------------------------------------


def _get_skills_dir() -> Path:
    """返回技能根目录（默认 ./.agents/skills）。"""
    skills_dir = getattr(config.settings, "AGENT_SKILLS_DIR", "./.agents/skills")
    return _workspace_root() / skills_dir


class ListAgentSkillsInput(BaseModel):
    """无需参数"""
    pass


class ListAgentSkills(AgentSkill):
    """列出所有已安装技能的 name 和 description（仅读取 SKILL.md 的 frontmatter）。"""

    def __init__(self) -> None:
        super().__init__(
            name="list_agent_skills",
            description="List all available agent skills with their names and descriptions.",
            args_schema=ListAgentSkillsInput,
        )

    async def run(self, **kwargs: Any) -> str:
        skills_dir = _get_skills_dir()
        if not skills_dir.exists() or not skills_dir.is_dir():
            return json.dumps([], ensure_ascii=False)

        skills = []
        for item in skills_dir.iterdir():
            if not item.is_dir():
                continue
            skill_md = item / "SKILL.md"
            if not skill_md.exists():
                continue

            try:
                content = await asyncio.to_thread(skill_md.read_text, encoding="utf-8")
                name = item.name  # 默认使用目录名
                description = "No description provided."

                # 解析 YAML frontmatter (格式：---\n...\n---)
                if content.startswith("---"):
                    parts = content.split("---", 2)
                    if len(parts) >= 3:
                        frontmatter = parts[1].strip()
                        try:
                            metadata = yaml.safe_load(frontmatter)
                            if isinstance(metadata, dict):
                                name = metadata.get("name", "").strip() or name
                                description = metadata.get("description", "").strip() or description
                        except Exception as e:
                            logger.warning(f"Failed to parse YAML in {skill_md}: {e}")

                skills.append({"name": name, "description": description})
            except Exception as e:
                logger.warning(f"Error reading {skill_md}: {e}")

        return json.dumps(skills, ensure_ascii=False)


class ReadAgentSkillDetailsInput(BaseModel):
    skill_name: str = Field(description="The name of the skill to read details for.")


class ReadAgentSkillDetails(AgentSkill):
    """读取指定技能的完整 SKILL.md 内容，并列出该目录下的其他文件。"""

    def __init__(self) -> None:
        super().__init__(
            name="read_agent_skill_details",
            description="Read the full SKILL.md content and list associated files for a given skill.",
            args_schema=ReadAgentSkillDetailsInput,
        )

    async def run(self, **kwargs: Any) -> str:
        skill_name = str(kwargs.get("skill_name", "")).strip()
        if not skill_name:
            return "Missing skill_name."

        skills_dir = _get_skills_dir()
        if not skills_dir.exists():
            return "Skills directory not found."

        # 不区分大小写匹配技能目录
        target_dir = None
        for item in skills_dir.iterdir():
            if item.is_dir() and item.name.lower() == skill_name.lower():
                target_dir = item
                break

        if not target_dir:
            return f"Skill '{skill_name}' not found."

        skill_md = target_dir / "SKILL.md"
        if not skill_md.exists():
            return f"Skill directory exists but no SKILL.md found."

        try:
            content = await asyncio.to_thread(skill_md.read_text, encoding="utf-8")
        except Exception as e:
            return f"Error reading SKILL.md: {e}"

        # 列出目录下除 SKILL.md 以外的所有文件
        other_files = []
        for f in target_dir.iterdir():
            if f.is_file() and f.name != "SKILL.md":
                other_files.append(f.name)

        result = {
            "skill_name": skill_name,
            "content": content,
            "other_files": other_files,
        }
        return json.dumps(result, ensure_ascii=False)


# ------------------------------------------------------------------------------
# 公开接口：返回所有可用技能（内置基础工具 + 元工具）
# ------------------------------------------------------------------------------

def get_agent_skills() -> list[AgentSkill]:
    if not config.settings.ENABLE_AGENT_SKILLS:
        return []

    return [
        ListAgentSkills(),
        ReadAgentSkillDetails(),
    ]