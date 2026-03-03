#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 修复了ChatOpenAI类无法处理模型推理过程(reasoning)的参数的问题，添加了对reasoning参数的支持，使得在流式输出token时能够正确处理模型推理过程的信息，并且在生成消息块时将reasoning参数包含在additional_kwargs中，以便在后续的处理过程中使用。
from typing import Any, Mapping, Optional, cast
from langchain_openai import ChatOpenAI
from langchain_openai.chat_models.base import _create_usage_metadata
from langchain_core.outputs import ChatGenerationChunk
from langchain_core.messages.tool import tool_call_chunk
from langchain_core.messages.ai import UsageMetadata
from langchain_core.messages import AIMessageChunk, BaseMessageChunk, ChatMessageChunk, FunctionMessageChunk, HumanMessageChunk, SystemMessageChunk, ToolMessageChunk

def _convert_delta_to_message_chunk(
    _dict: Mapping[str, Any], default_class: type[BaseMessageChunk]
) -> BaseMessageChunk:
    """Convert to a LangChain message chunk."""
    id_ = _dict.get("id")
    role = cast(str, _dict.get("role"))
    content = cast(str, _dict.get("content") or "")
    additional_kwargs: dict = {}
    if _dict.get("function_call"):
        function_call = dict(_dict["function_call"])
        if "name" in function_call and function_call["name"] is None:
            function_call["name"] = ""
        additional_kwargs["function_call"] = function_call
    if _dict.get("reasoning"): # 添加了对模型推理过程的支持，将reasoning参数添加到additional_kwargs中，以便在生成消息块时使用
        additional_kwargs["reasoning"] = _dict["reasoning"]
    tool_call_chunks = []
    if raw_tool_calls := _dict.get("tool_calls"):
        try:
            tool_call_chunks = [
                tool_call_chunk(
                    name=rtc["function"].get("name"),
                    args=rtc["function"].get("arguments"),
                    id=rtc.get("id"),
                    index=rtc["index"],
                )
                for rtc in raw_tool_calls
            ]
        except KeyError:
            pass

    if role == "user" or default_class == HumanMessageChunk:
        return HumanMessageChunk(content=content, id=id_)
    if role == "assistant" or default_class == AIMessageChunk:
        return AIMessageChunk(
            content=content,
            additional_kwargs=additional_kwargs,
            id=id_,
            tool_call_chunks=tool_call_chunks,  # type: ignore[arg-type]
        )
    if role in ("system", "developer") or default_class == SystemMessageChunk:
        if role == "developer":
            additional_kwargs = {"__openai_role__": "developer"}
        else:
            additional_kwargs = {}
        return SystemMessageChunk(
            content=content, id=id_, additional_kwargs=additional_kwargs
        )
    if role == "function" or default_class == FunctionMessageChunk:
        return FunctionMessageChunk(content=content, name=_dict["name"], id=id_)
    if role == "tool" or default_class == ToolMessageChunk:
        return ToolMessageChunk(
            content=content, tool_call_id=_dict["tool_call_id"], id=id_
        )
    if role or default_class == ChatMessageChunk:
        return ChatMessageChunk(content=content, role=role, id=id_)
    return default_class(content=content, id=id_)  # type: ignore[call-arg]


class ChatOpenAIWithReasoning(ChatOpenAI):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _get_chat_model_kwargs(self, stop: Optional[list[str]] = None) -> dict:
        """Get the kwargs to pass to the chat model."""
        kwargs = super()._get_chat_model_kwargs(stop=stop) # type: ignore
        # 添加 reasoning 参数
        kwargs["reasoning"] = True
        return kwargs
    
    def _convert_chunk_to_generation_chunk(
        self,
        chunk: dict,
        default_chunk_class: type,
        base_generation_info: dict | None,
    ) -> ChatGenerationChunk | None: # 只是重新使用了_convert_delta_to_message_chunk方法来处理reasoning参数，并没有改变原有的逻辑

        if chunk.get("type") == "content.delta":  # From beta.chat.completions.stream
            return None
        token_usage = chunk.get("usage")
        choices = (
            chunk.get("choices", [])
            # From beta.chat.completions.stream
            or chunk.get("chunk", {}).get("choices", [])
        )

        usage_metadata: UsageMetadata | None = (
            _create_usage_metadata(token_usage, chunk.get("service_tier"))
            if token_usage
            else None
        )
        if len(choices) == 0:
            # logprobs is implicitly None
            generation_chunk = ChatGenerationChunk(
                message=default_chunk_class(content="", usage_metadata=usage_metadata),
                generation_info=base_generation_info,
            )
            if self.output_version == "v1":
                generation_chunk.message.content = []
                generation_chunk.message.response_metadata["output_version"] = "v1"

            return generation_chunk

        choice = choices[0]
        if choice["delta"] is None:
            return None

        message_chunk = _convert_delta_to_message_chunk(
            choice["delta"], default_chunk_class
        )
        generation_info = {**base_generation_info} if base_generation_info else {}

        if finish_reason := choice.get("finish_reason"):
            generation_info["finish_reason"] = finish_reason
            if model_name := chunk.get("model"):
                generation_info["model_name"] = model_name
            if system_fingerprint := chunk.get("system_fingerprint"):
                generation_info["system_fingerprint"] = system_fingerprint
            if service_tier := chunk.get("service_tier"):
                generation_info["service_tier"] = service_tier

        logprobs = choice.get("logprobs")
        if logprobs:
            generation_info["logprobs"] = logprobs

        if usage_metadata and isinstance(message_chunk, AIMessageChunk):
            message_chunk.usage_metadata = usage_metadata

        message_chunk.response_metadata["model_provider"] = "openai"
        return ChatGenerationChunk(
            message=message_chunk, generation_info=generation_info or None
        )