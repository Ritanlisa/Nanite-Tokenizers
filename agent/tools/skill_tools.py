from __future__ import annotations

import logging
from typing import Any

from langchain_core.tools import BaseTool

from agent.skills import AgentSkill, get_agent_skills
from tool_usage import end_current_tool_call, start_current_tool_call

from ._common import InputSugarTool

logger = logging.getLogger(__name__)

class SkillBridgeTool(InputSugarTool):
    skill: AgentSkill
    name: str = ""
    description: str = ""
    args_schema: Any = None

    def __init__(self, skill: AgentSkill, **kwargs: Any):
        init_data = {
            "name": skill.name,
            "description": skill.description,
            "args_schema": skill.args_schema,
            "skill": skill,
        }
        init_data.update(kwargs)
        super().__init__(**init_data)

    async def _arun(self, **kwargs: Any) -> str:
        
        call_id = start_current_tool_call(self.name, kwargs)
        output_text = ""
        try:
            output_text = await self.skill.run(**kwargs)
            return output_text
        finally:
            end_current_tool_call(call_id, output_text)

    def _run(self, **kwargs: Any) -> str:
        raise NotImplementedError("Use async call")

def _build_skill_tools() -> list[BaseTool]:
    return [SkillBridgeTool(skill=skill) for skill in get_agent_skills()]
