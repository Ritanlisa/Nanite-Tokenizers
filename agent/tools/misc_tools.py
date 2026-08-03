from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Literal

from pydantic import BaseModel, Field
import sympy as sp

import config
from tool_usage import end_current_tool_call, start_current_tool_call

from ._common import InputSugarTool, _bi, _t, _workspace_root

logger = logging.getLogger(__name__)

class MathComputeInput(BaseModel):
    mode: Literal["eval", "simplify", "function", "solve", "matrix"] = Field(
        description=_bi("计算模式：eval/simplify/function/solve/matrix", "Computation mode: eval/simplify/function/solve/matrix")
    )
    expression: str = Field(default="", description=_bi("表达式（eval/simplify/function/matrix 时可用）", "Expression (used in eval/simplify/function/matrix modes)"))
    variables: dict[str, float] = Field(default_factory=dict, description=_bi("变量赋值，如 {\"x\": 2}", "Variable assignments, e.g. {\"x\": 2}"))
    equation: str = Field(default="", description=_bi("方程字符串，如 'x**2-4=0'（solve 模式）", "Equation string, e.g. 'x**2-4=0' (solve mode)"))
    symbol: str = Field(default="x", description=_bi("求解变量名（solve 模式）", "Variable name to solve for (solve mode)"))
    matrix_a: list[list[float]] = Field(default_factory=list, description=_bi("矩阵 A（matrix 模式）", "Matrix A (matrix mode)"))
    matrix_b: list[list[float]] = Field(default_factory=list, description=_bi("矩阵 B（matrix add/sub/mul 模式）", "Matrix B (matrix add/sub/mul modes)"))
    matrix_op: Literal["add", "sub", "mul", "det", "inv", "transpose", "rank"] = Field(
        default="mul", description=_bi("矩阵运算类型", "Matrix operation type")
    )

class MathComputeTool(InputSugarTool):
    name: str = "math_compute"
    description: str = _bi("数学计算工具：表达式计算/表达式化简/函数计算/方程求解/矩阵运算。", "Math tool: expression eval/simplify, function eval, equation solving, and matrix operations.")
    args_schema: Any = MathComputeInput

    @staticmethod
    def _to_number_if_possible(value: Any) -> Any:
        if isinstance(value, (int, float)):
            return value
        if getattr(value, "is_real", False) and getattr(value, "is_number", False):
            as_float = float(value)
            if as_float.is_integer():
                return int(as_float)
            return as_float
        return str(value)

    async def _arun(
        self,
        mode: str,
        expression: str = "",
        variables: dict[str, float] | None = None,
        equation: str = "",
        symbol: str = "x",
        matrix_a: list[list[float]] | None = None,
        matrix_b: list[list[float]] | None = None,
        matrix_op: str = "mul",
    ) -> str:
        call_payload = {
            "mode": mode,
            "expression": expression,
            "variables": variables or {},
            "equation": equation,
            "symbol": symbol,
            "matrix_op": matrix_op,
        }
        call_id = start_current_tool_call(self.name, call_payload)
        output_text = ""
        try:
            mode = (mode or "").strip().lower()
            variables = variables or {}
            local_vars = {name: sp.Float(value) for name, value in variables.items()}

            if mode == "eval":
                if not expression.strip():
                    return _t("表达式不能为空。", "Expression cannot be empty.")
                expr = sp.sympify(expression)
                result = expr.evalf(subs=local_vars)
                payload = {
                    "mode": mode,
                    "result": self._to_number_if_possible(result),
                }
                output_text = json.dumps(payload, ensure_ascii=False)
                return output_text

            if mode == "simplify":
                if not expression.strip():
                    return _t("表达式不能为空。", "Expression cannot be empty.")
                expr = sp.sympify(expression)
                simplified = sp.simplify(expr)
                payload = {
                    "mode": mode,
                    "result": str(simplified),
                }
                output_text = json.dumps(payload, ensure_ascii=False)
                return output_text

            if mode == "function":
                if not expression.strip():
                    return _t("函数表达式不能为空。", "Function expression cannot be empty.")
                expr = sp.sympify(expression)
                result = expr.evalf(subs=local_vars)
                payload = {
                    "mode": mode,
                    "result": self._to_number_if_possible(result),
                }
                output_text = json.dumps(payload, ensure_ascii=False)
                return output_text

            if mode == "solve":
                source = (equation or expression or "").strip()
                if not source:
                    return _t("方程不能为空。", "Equation cannot be empty.")
                var = sp.Symbol((symbol or "x").strip() or "x")
                if "=" in source:
                    left, right = source.split("=", 1)
                    eq = sp.Eq(sp.sympify(left), sp.sympify(right))
                    solutions = sp.solve(eq, var)
                else:
                    solutions = sp.solve(sp.sympify(source), var)
                payload = {
                    "mode": mode,
                    "symbol": str(var),
                    "solutions": [self._to_number_if_possible(item) for item in solutions],
                }
                output_text = json.dumps(payload, ensure_ascii=False)
                return output_text

            if mode == "matrix":
                matrix_a = matrix_a or []
                matrix_b = matrix_b or []
                if not matrix_a:
                    return _t("矩阵模式需要 matrix_a 参数。", "Matrix mode requires matrix_a.")
                mat_a = sp.Matrix(matrix_a)
                op = (matrix_op or "mul").strip().lower()
                result: Any
                if op == "add":
                    if not matrix_b:
                        return _t("矩阵加法需要 matrix_b 参数。", "Matrix add requires matrix_b.")
                    result = mat_a + sp.Matrix(matrix_b)
                elif op == "sub":
                    if not matrix_b:
                        return _t("矩阵减法需要 matrix_b 参数。", "Matrix subtraction requires matrix_b.")
                    result = mat_a - sp.Matrix(matrix_b)
                elif op == "mul":
                    if not matrix_b:
                        return _t("矩阵乘法需要 matrix_b 参数。", "Matrix multiplication requires matrix_b.")
                    result = mat_a * sp.Matrix(matrix_b)
                elif op == "det":
                    result = mat_a.det()
                elif op == "inv":
                    result = mat_a.inv()
                elif op == "transpose":
                    result = mat_a.T
                elif op == "rank":
                    result = mat_a.rank()
                else:
                    return _t("不支持的矩阵运算类型。", "Unsupported matrix operation.")

                if isinstance(result, sp.MatrixBase):
                    matrix_payload = [
                        [self._to_number_if_possible(cell) for cell in row]
                        for row in result.tolist()
                    ]
                    payload = {
                        "mode": mode,
                        "matrix_op": op,
                        "result": matrix_payload,
                    }
                else:
                    payload = {
                        "mode": mode,
                        "matrix_op": op,
                        "result": self._to_number_if_possible(result),
                    }
                output_text = json.dumps(payload, ensure_ascii=False)
                return output_text

            output_text = _t("不支持的计算模式。", "Unsupported computation mode.")
            return output_text
        except Exception as exc:
            logger.exception("math_compute failed")
            if config.settings.ENV == "prod":
                output_text = _t("数学计算失败。", "Math computation failed.")
            else:
                output_text = _t(
                    f"数学计算失败: {type(exc).__name__}: {str(exc)[:160]}",
                    f"Math computation failed: {type(exc).__name__}: {str(exc)[:160]}",
                )
            return output_text
        finally:
            end_current_tool_call(call_id, output_text)

    def _run(self, **kwargs) -> str:
        raise NotImplementedError("Use async call")

class AgentIdeaInput(BaseModel):
    content: str = Field(description=_bi("要记录的内容", "Content to record"))

class FeedbackTool(InputSugarTool):
    name: str = "feedback"
    description: str = _bi("反馈 Agent 自己需要的功能，追加记录到 ./agentIdeas/feedback.txt", "Record desired agent features, appended to ./agentIdeas/feedback.txt")
    args_schema: Any = AgentIdeaInput

    async def _arun(self, content: str) -> str:
        call_id = start_current_tool_call(self.name, {"content": content})
        output_text = ""
        try:
            text = (content or "").strip()
            if not text:
                output_text = _t("反馈内容不能为空。", "Feedback content cannot be empty.")
                return output_text

            target = _workspace_root() / "agentIdeas" / "feedback.txt"
            await asyncio.to_thread(target.parent.mkdir, parents=True, exist_ok=True)

            def _append() -> None:
                with target.open("a", encoding="utf-8") as handle:
                    handle.write(text + "\n")

            await asyncio.to_thread(_append)
            output_text = _t("已记录到 agentIdeas/feedback.txt", "Recorded to agentIdeas/feedback.txt")
            return output_text
        except Exception as exc:
            logger.exception("feedback tool failed")
            if config.settings.ENV == "prod":
                output_text = _t("写入反馈失败。", "Failed to write feedback.")
            else:
                output_text = _t(
                    f"写入反馈失败: {type(exc).__name__}",
                    f"Failed to write feedback: {type(exc).__name__}",
                )
            return output_text
        finally:
            end_current_tool_call(call_id, output_text)

    def _run(self, **kwargs) -> str:
        raise NotImplementedError("Use async call")


class BugTool(InputSugarTool):
    name: str = "bug"
    description: str = _bi("反馈 Agent 自己遇到的 bug，追加记录到 ./agentIdeas/bug.txt", "Record agent bugs encountered, appended to ./agentIdeas/bug.txt")
    args_schema: Any = AgentIdeaInput

    async def _arun(self, content: str) -> str:
        call_id = start_current_tool_call(self.name, {"content": content})
        output_text = ""
        try:
            text = (content or "").strip()
            if not text:
                output_text = _t("缺陷内容不能为空。", "Bug content cannot be empty.")
                return output_text

            target = _workspace_root() / "agentIdeas" / "bug.txt"
            await asyncio.to_thread(target.parent.mkdir, parents=True, exist_ok=True)

            def _append() -> None:
                with target.open("a", encoding="utf-8") as handle:
                    handle.write(text + "\n")

            await asyncio.to_thread(_append)
            output_text = _t("已记录到 agentIdeas/bug.txt", "Recorded to agentIdeas/bug.txt")
            return output_text
        except Exception as exc:
            logger.exception("bug tool failed")
            if config.settings.ENV == "prod":
                output_text = _t("写入缺陷记录失败。", "Failed to write bug record.")
            else:
                output_text = _t(
                    f"写入缺陷记录失败: {type(exc).__name__}",
                    f"Failed to write bug record: {type(exc).__name__}",
                )
            return output_text
        finally:
            end_current_tool_call(call_id, output_text)

    def _run(self, **kwargs) -> str:
        raise NotImplementedError("Use async call")
