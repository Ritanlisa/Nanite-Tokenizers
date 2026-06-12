"""
SysML 2.0 抽象语法模型及文本序列化。
符合 KerML/SysML 2.0 文本记法规范（子集）。
"""

from __future__ import annotations
from typing import Optional, List, Union
from enum import Enum


class VisibilityKind(str, Enum):
    PUBLIC = "public"
    PRIVATE = "private"
    PROTECTED = "protected"


class DirectionKind(str, Enum):
    IN = "in"
    OUT = "out"
    INOUT = "inout"


class SysMLElement:
    """所有模型元素的基类"""
    def __init__(self, name: Optional[str] = None, short_name: Optional[str] = None):
        self.name = name
        self.short_name = short_name
        self.owner: Optional[SysMLElement] = None
        self._id = id(self)  # 简单模拟唯一标识

    @property
    def qualified_name(self) -> str:
        if self.owner is None or not hasattr(self.owner, 'qualified_name'):
            return self.name or ''
        parent_qn = self.owner.qualified_name
        return f"{parent_qn}::{self.name}" if parent_qn else (self.name or '')

    @staticmethod
    def _is_std_identifier(name: str) -> bool:
        """SysML v2 / KerML 标准标识符: [a-zA-Z_][a-zA-Z0-9_]*"""
        import re
        return bool(re.fullmatch(r'[a-zA-Z_][a-zA-Z0-9_]*', name))

    @staticmethod
    def _needs_quoting(name: str) -> bool:
        """任何非标准 Latin 标识符都需单引号引用"""
        return not SysMLElement._is_std_identifier(name)

    @staticmethod
    def _quote(name: str) -> str:
        """必要时给名称加引号"""
        if SysMLElement._needs_quoting(name):
            return f"'{name}'"
        return name

    def _name_part(self) -> str:
        """返回名称部分文本（含短名）。若短名与名称相同/包含则仅返回名称。"""
        if self.short_name and self.name:
            s = str(self.short_name)
            n = str(self.name)
            if s.strip() == n.strip() or s in n:
                return self._quote(n)
            return f"<{self._quote(s)}> {self._quote(n)}"
        elif self.short_name:
            return f"<{self._quote(self.short_name)}>"
        elif self.name:
            return self._quote(self.name)
        return ""

    def to_text(self, indent: int = 0) -> str:
        raise NotImplementedError


class Doc(SysMLElement):
    """Doc statement — documentation text attached to an element"""
    def __init__(self, text: str):
        super().__init__(name=None)
        self.text = text

    def to_text(self, indent: int = 0) -> str:
        prefix = "    " * indent
        return f'{prefix}doc "{self.text}";'


class Namespace(SysMLElement):
    """命名空间基类（包、定义、使用）"""
    def __init__(self, name: Optional[str] = None, short_name: Optional[str] = None):
        super().__init__(name, short_name)
        self.members: List[Union[Definition, Usage, Alias, Import, "Package", Doc]] = []

    def add_member(self, member: Union[Definition, Usage, Alias, Import, "Package", Doc]):
        member.owner = self
        self.members.append(member)
        return member

    def members_to_text(self, indent: int) -> str:
        lines = []
        for m in self.members:
            lines.append(m.to_text(indent))
        return "\n".join(lines)


class Package(Namespace):
    def __init__(self, name: str, short_name: Optional[str] = None):
        super().__init__(name, short_name)

    def to_text(self, indent: int = 0) -> str:
        prefix = "    " * indent
        name_part = self._name_part()
        body = self.members_to_text(indent + 1)
        if body:
            return f"{prefix}package {name_part} {{\n{body}\n{prefix}}}"
        else:
            return f"{prefix}package {name_part};"


class Multiplicity:
    def __init__(self, lower: Optional[str] = None, upper: Optional[str] = None,
                 ordered: bool = False, unique: bool = True):
        self.lower = lower
        self.upper = upper
        self.ordered = ordered
        self.unique = unique

    def to_text(self) -> str:
        if self.lower is None and self.upper is None:
            return ""
        lower_str = self.lower if self.lower is not None else ""
        upper_str = self.upper if self.upper is not None else ""
        if lower_str == upper_str:
            mult = f"[{lower_str}]" if lower_str else ""
        else:
            mult = f"[{lower_str}..{upper_str}]"
        if self.ordered:
            mult += " ordered"
        if not self.unique:
            mult += " nonunique"
        return mult


class Feature(SysMLElement):
    """特征基类（定义的特征或使用的特征）"""
    def __init__(self, name: Optional[str] = None, short_name: Optional[str] = None,
                 direction: Optional[DirectionKind] = None,
                 multiplicity: Optional[Multiplicity] = None,
                 is_derived: bool = False, is_abstract: bool = False,
                 is_constant: bool = False, is_reference: bool = False):
        super().__init__(name, short_name)
        self.direction = direction
        self.multiplicity = multiplicity
        self.is_derived = is_derived
        self.is_abstract = is_abstract
        self.is_constant = is_constant
        self.is_reference = is_reference
        self.type_refs: List[str] = []  # 定义类型引用
        self.subsetted: List[str] = []
        self.redefined: List[str] = []
        self.value_expr: Optional[str] = None  # 绑定值

    def feature_prefix(self) -> str:
        parts = []
        if self.direction:
            parts.append(self.direction.value)
        if self.is_derived:
            parts.append("derived")
        if self.is_abstract:
            parts.append("abstract")
        if self.is_constant:
            parts.append("constant")
        if self.is_reference:
            parts.append("ref")
        return " ".join(parts) + (" " if parts else "")

    def specialization_part(self) -> str:
        specs = []
        if self.type_refs:
            specs.append(": " + ", ".join(self.type_refs))
        if self.subsetted:
            specs.append("subsets " + ", ".join(self.subsetted))
        if self.redefined:
            specs.append("redefines " + ", ".join(self.redefined))
        return " ".join(specs)


class Definition(Namespace):
    """定义基类"""
    def __init__(self, name: Optional[str] = None, short_name: Optional[str] = None,
                 is_abstract: bool = False, is_variation: bool = False,
                 supertypes: List[str] = []):
        super().__init__(name, short_name)
        self.is_abstract = is_abstract
        self.is_variation = is_variation
        self.supertypes = supertypes or []

    def def_prefix(self) -> str:
        prefix = ""
        if self.is_abstract:
            prefix += "abstract "
        if self.is_variation:
            prefix += "variation "
        return prefix

    def def_kind(self) -> str:
        raise NotImplementedError

    def to_text(self, indent: int = 0) -> str:
        prefix = "    " * indent
        name_part = self._name_part()
        super_part = ""
        if self.supertypes:
            super_part = " :> " + ", ".join(self.supertypes)
        header = f"{prefix}{self.def_prefix()}{self.def_kind()} {name_part}{super_part}"
        body = self.members_to_text(indent + 1)
        if body:
            return f"{header} {{\n{body}\n{prefix}}}"
        else:
            return f"{header};"


class Usage(Namespace):
    """使用基类"""
    def __init__(self, name: Optional[str] = None, short_name: Optional[str] = None,
                 direction: Optional[DirectionKind] = None,
                 multiplicity: Optional[Multiplicity] = None,
                 is_derived: bool = False, is_abstract: bool = False,
                 is_constant: bool = False, is_reference: bool = False,
                 type_refs: List[str] = [], subsetted: List[str] = [], redefined: List[str] = [],
                 value_expr: Optional[str] = None):
        super().__init__(name, short_name)
        self.direction = direction
        self.multiplicity = multiplicity
        self.is_derived = is_derived
        self.is_abstract = is_abstract
        self.is_constant = is_constant
        self.is_reference = is_reference
        self.type_refs = type_refs or []
        self.subsetted = subsetted or []
        self.redefined = redefined or []
        self.value_expr = value_expr

    def feature_prefix(self) -> str:
        parts = []
        if self.direction:
            parts.append(self.direction.value)
        if self.is_derived:
            parts.append("derived")
        if self.is_abstract:
            parts.append("abstract")
        if self.is_constant:
            parts.append("constant")
        if self.is_reference:
            parts.append("ref")
        return " ".join(parts) + (" " if parts else "")

    def usage_kind(self) -> str:
        raise NotImplementedError

    def specialization_part(self) -> str:
        specs = []
        if self.type_refs:
            specs.append(": " + ", ".join(self.type_refs))
        if self.subsetted:
            specs.append("subsets " + ", ".join(self.subsetted))
        if self.redefined:
            specs.append("redefines " + ", ".join(self.redefined))
        return " ".join(specs)

    def to_text(self, indent: int = 0) -> str:
        prefix = "    " * indent
        name_part = self._name_part()
        mult = self.multiplicity.to_text() if self.multiplicity else ""
        spec = self.specialization_part()
        val = f" = {self.value_expr}" if self.value_expr else ""
        header = f"{prefix}{self.feature_prefix()}{self.usage_kind()} {name_part}{mult}{spec}{val}"
        body = self.members_to_text(indent + 1)
        if body:
            return f"{header} {{\n{body}\n{prefix}}}"
        else:
            return f"{header};"


# ---------- 具体定义类 ----------
class PartDef(Definition):
    def def_kind(self) -> str:
        return "part def"

class PartUsage(Usage):
    def usage_kind(self) -> str:
        return "part"

class AttributeDef(Definition):
    def def_kind(self) -> str:
        return "attribute def"

class AttributeUsage(Usage):
    def usage_kind(self) -> str:
        return "attribute"

class PortDef(Definition):
    def def_kind(self) -> str:
        return "port def"

class PortUsage(Usage):
    def usage_kind(self) -> str:
        return "port"

class ItemDef(Definition):
    def def_kind(self) -> str:
        return "item def"

class ItemUsage(Usage):
    def usage_kind(self) -> str:
        return "item"

class ConnectionDef(Definition):
    def def_kind(self) -> str:
        return "connection def"

class ConnectionUsage(Usage):
    def __init__(self, name: Optional[str] = None, short_name: Optional[str] = None,
                 ends: List[ConnectionEnd] = [], **kwargs):
        super().__init__(name, short_name, **kwargs)
        self.ends = ends or []

    def usage_kind(self) -> str:
        return "connection"

    def to_text(self, indent: int = 0) -> str:
        prefix = "    " * indent
        if len(self.ends) == 2:
            e1, e2 = self.ends
            if not self.name and not self.short_name:
                return f"{prefix}connect {SysMLElement._quote(e1.ref)} to {SysMLElement._quote(e2.ref)};"
            # Named connection with ends: serialize ends in body block for round-trip safety
            name_part = self._name_part()
            spec = self.specialization_part()
            header = f"{prefix}{self.usage_kind()} {name_part}{spec}"
            body_prefix = "    " * (indent + 1)
            body_line = f"connect {SysMLElement._quote(e1.ref)} to {SysMLElement._quote(e2.ref)};"
            return f"{header} {{\n{body_prefix}{body_line}\n{prefix}}}"
        else:
            return super().to_text(indent)

class ConnectionEnd:
    def __init__(self, ref: str, role: Optional[str] = None):
        self.ref = ref
        self.role = role

class InterfaceDef(Definition):
    def def_kind(self) -> str:
        return "interface def"

class InterfaceUsage(ConnectionUsage):   # 改为继承 ConnectionUsage
    def usage_kind(self) -> str:
        return "interface"

    def to_text(self, indent: int = 0) -> str:
        prefix = "    " * indent
        if len(self.ends) == 2 and (self.name or self.short_name):
            e1, e2 = self.ends
            name_part = self._name_part()
            spec = self.specialization_part()
            header = f"{prefix}{self.usage_kind()} {name_part}{spec}"
            body_prefix = "    " * (indent + 1)
            body_line = f"connect {SysMLElement._quote(e1.ref)} to {SysMLElement._quote(e2.ref)};"
            return f"{header} {{\n{body_prefix}{body_line}\n{prefix}}}"
        return ConnectionUsage.to_text(self, indent)

class AllocationDef(Definition):
    def def_kind(self) -> str:
        return "allocation def"

class AllocationUsage(ConnectionUsage):  # 改为继承 ConnectionUsage
    def usage_kind(self) -> str:
        return "allocation"

    def to_text(self, indent: int = 0) -> str:
        prefix = "    " * indent
        if len(self.ends) == 2 and (self.name or self.short_name):
            e1, e2 = self.ends
            name_part = self._name_part()
            spec = self.specialization_part()
            header = f"{prefix}{self.usage_kind()} {name_part}{spec}"
            body_prefix = "    " * (indent + 1)
            body_line = f"connect {SysMLElement._quote(e1.ref)} to {SysMLElement._quote(e2.ref)};"
            return f"{header} {{\n{body_prefix}{body_line}\n{prefix}}}"
        return ConnectionUsage.to_text(self, indent)

class ContainmentUsage(ConnectionUsage):
    """包含关系: 整体包含部分 (hierarchical decomposition)"""
    def usage_kind(self) -> str:
        return "containment"

    def to_text(self, indent: int = 0) -> str:
        prefix = "    " * indent
        if len(self.ends) == 2 and (self.name or self.short_name):
            e1, e2 = self.ends
            name_part = self._name_part()
            spec = self.specialization_part()
            header = f"{prefix}{self.usage_kind()} {name_part}{spec}"
            body_prefix = "    " * (indent + 1)
            body_line = f"connect {SysMLElement._quote(e1.ref)} to {SysMLElement._quote(e2.ref)};"
            return f"{header} {{\n{body_prefix}{body_line}\n{prefix}}}"
        return ConnectionUsage.to_text(self, indent)

class CompositionUsage(ConnectionUsage):
    """组合关系: 强整体-部分 (strong whole-part, 部分不能脱离整体)"""
    def usage_kind(self) -> str:
        return "composition"

    def to_text(self, indent: int = 0) -> str:
        prefix = "    " * indent
        if len(self.ends) == 2 and (self.name or self.short_name):
            e1, e2 = self.ends
            name_part = self._name_part()
            spec = self.specialization_part()
            header = f"{prefix}{self.usage_kind()} {name_part}{spec}"
            body_prefix = "    " * (indent + 1)
            body_line = f"connect {SysMLElement._quote(e1.ref)} to {SysMLElement._quote(e2.ref)};"
            return f"{header} {{\n{body_prefix}{body_line}\n{prefix}}}"
        return ConnectionUsage.to_text(self, indent)

class ReferenceUsage(ConnectionUsage):
    """引用关系: 弱交叉引用 (weak cross-reference)"""
    def usage_kind(self) -> str:
        return "reference"

    def to_text(self, indent: int = 0) -> str:
        prefix = "    " * indent
        if len(self.ends) == 2 and (self.name or self.short_name):
            e1, e2 = self.ends
            name_part = self._name_part()
            spec = self.specialization_part()
            header = f"{prefix}{self.usage_kind()} {name_part}{spec}"
            body_prefix = "    " * (indent + 1)
            body_line = f"connect {SysMLElement._quote(e1.ref)} to {SysMLElement._quote(e2.ref)};"
            return f"{header} {{\n{body_prefix}{body_line}\n{prefix}}}"
        return ConnectionUsage.to_text(self, indent)

class GeneralizationUsage(ConnectionUsage):
    def usage_kind(self) -> str:
        return "generalization"

    def to_text(self, indent: int = 0) -> str:
        prefix = "    " * indent
        if len(self.ends) == 2 and (self.name or self.short_name):
            e1, e2 = self.ends
            name_part = self._name_part()
            spec = self.specialization_part()
            header = f"{prefix}{self.usage_kind()} {name_part}{spec}"
            body_prefix = "    " * (indent + 1)
            body_line = f"connect {SysMLElement._quote(e1.ref)} to {SysMLElement._quote(e2.ref)};"
            return f"{header} {{\n{body_prefix}{body_line}\n{prefix}}}"
        return ConnectionUsage.to_text(self, indent)

class DependencyUsage(ConnectionUsage):
    def usage_kind(self) -> str:
        return "dependency"

    def to_text(self, indent: int = 0) -> str:
        prefix = "    " * indent
        if len(self.ends) == 2 and (self.name or self.short_name):
            e1, e2 = self.ends
            name_part = self._name_part()
            spec = self.specialization_part()
            header = f"{prefix}{self.usage_kind()} {name_part}{spec}"
            body_prefix = "    " * (indent + 1)
            body_line = f"connect {SysMLElement._quote(e1.ref)} to {SysMLElement._quote(e2.ref)};"
            return f"{header} {{\n{body_prefix}{body_line}\n{prefix}}}"
        return ConnectionUsage.to_text(self, indent)

class AbstractionUsage(ConnectionUsage):
    def usage_kind(self) -> str:
        return "abstraction"

    def to_text(self, indent: int = 0) -> str:
        prefix = "    " * indent
        if len(self.ends) == 2 and (self.name or self.short_name):
            e1, e2 = self.ends
            name_part = self._name_part()
            spec = self.specialization_part()
            header = f"{prefix}{self.usage_kind()} {name_part}{spec}"
            body_prefix = "    " * (indent + 1)
            body_line = f"connect {SysMLElement._quote(e1.ref)} to {SysMLElement._quote(e2.ref)};"
            return f"{header} {{\n{body_prefix}{body_line}\n{prefix}}}"
        return ConnectionUsage.to_text(self, indent)

class RealizationUsage(ConnectionUsage):
    def usage_kind(self) -> str:
        return "realization"

    def to_text(self, indent: int = 0) -> str:
        prefix = "    " * indent
        if len(self.ends) == 2 and (self.name or self.short_name):
            e1, e2 = self.ends
            name_part = self._name_part()
            spec = self.specialization_part()
            header = f"{prefix}{self.usage_kind()} {name_part}{spec}"
            body_prefix = "    " * (indent + 1)
            body_line = f"connect {SysMLElement._quote(e1.ref)} to {SysMLElement._quote(e2.ref)};"
            return f"{header} {{\n{body_prefix}{body_line}\n{prefix}}}"
        return ConnectionUsage.to_text(self, indent)

class DeriveUsage(ConnectionUsage):
    def usage_kind(self) -> str:
        return "derive"

    def to_text(self, indent: int = 0) -> str:
        prefix = "    " * indent
        if len(self.ends) == 2 and (self.name or self.short_name):
            e1, e2 = self.ends
            name_part = self._name_part()
            spec = self.specialization_part()
            header = f"{prefix}{self.usage_kind()} {name_part}{spec}"
            body_prefix = "    " * (indent + 1)
            body_line = f"connect {SysMLElement._quote(e1.ref)} to {SysMLElement._quote(e2.ref)};"
            return f"{header} {{\n{body_prefix}{body_line}\n{prefix}}}"
        return ConnectionUsage.to_text(self, indent)

class TraceUsage(ConnectionUsage):
    def usage_kind(self) -> str:
        return "trace"

    def to_text(self, indent: int = 0) -> str:
        prefix = "    " * indent
        if len(self.ends) == 2 and (self.name or self.short_name):
            e1, e2 = self.ends
            name_part = self._name_part()
            spec = self.specialization_part()
            header = f"{prefix}{self.usage_kind()} {name_part}{spec}"
            body_prefix = "    " * (indent + 1)
            body_line = f"connect {SysMLElement._quote(e1.ref)} to {SysMLElement._quote(e2.ref)};"
            return f"{header} {{\n{body_prefix}{body_line}\n{prefix}}}"
        return ConnectionUsage.to_text(self, indent)

class DeriveReqtUsage(ConnectionUsage):
    def usage_kind(self) -> str:
        return "derivereqt"

    def to_text(self, indent: int = 0) -> str:
        prefix = "    " * indent
        if len(self.ends) == 2 and (self.name or self.short_name):
            e1, e2 = self.ends
            name_part = self._name_part()
            spec = self.specialization_part()
            header = f"{prefix}{self.usage_kind()} {name_part}{spec}"
            body_prefix = "    " * (indent + 1)
            body_line = f"connect {SysMLElement._quote(e1.ref)} to {SysMLElement._quote(e2.ref)};"
            return f"{header} {{\n{body_prefix}{body_line}\n{prefix}}}"
        return ConnectionUsage.to_text(self, indent)

class RefineUsage(ConnectionUsage):
    def usage_kind(self) -> str:
        return "refine"

    def to_text(self, indent: int = 0) -> str:
        prefix = "    " * indent
        if len(self.ends) == 2 and (self.name or self.short_name):
            e1, e2 = self.ends
            name_part = self._name_part()
            spec = self.specialization_part()
            header = f"{prefix}{self.usage_kind()} {name_part}{spec}"
            body_prefix = "    " * (indent + 1)
            body_line = f"connect {SysMLElement._quote(e1.ref)} to {SysMLElement._quote(e2.ref)};"
            return f"{header} {{\n{body_prefix}{body_line}\n{prefix}}}"
        return ConnectionUsage.to_text(self, indent)

class SatisfyUsage(ConnectionUsage):
    def usage_kind(self) -> str:
        return "satisfy"

    def to_text(self, indent: int = 0) -> str:
        prefix = "    " * indent
        if len(self.ends) == 2 and (self.name or self.short_name):
            e1, e2 = self.ends
            name_part = self._name_part()
            spec = self.specialization_part()
            header = f"{prefix}{self.usage_kind()} {name_part}{spec}"
            body_prefix = "    " * (indent + 1)
            body_line = f"connect {SysMLElement._quote(e1.ref)} to {SysMLElement._quote(e2.ref)};"
            return f"{header} {{\n{body_prefix}{body_line}\n{prefix}}}"
        return ConnectionUsage.to_text(self, indent)

class VerifyUsage(ConnectionUsage):
    def usage_kind(self) -> str:
        return "verify"

    def to_text(self, indent: int = 0) -> str:
        prefix = "    " * indent
        if len(self.ends) == 2 and (self.name or self.short_name):
            e1, e2 = self.ends
            name_part = self._name_part()
            spec = self.specialization_part()
            header = f"{prefix}{self.usage_kind()} {name_part}{spec}"
            body_prefix = "    " * (indent + 1)
            body_line = f"connect {SysMLElement._quote(e1.ref)} to {SysMLElement._quote(e2.ref)};"
            return f"{header} {{\n{body_prefix}{body_line}\n{prefix}}}"
        return ConnectionUsage.to_text(self, indent)

class CopyUsage(ConnectionUsage):
    def usage_kind(self) -> str:
        return "copy"

    def to_text(self, indent: int = 0) -> str:
        prefix = "    " * indent
        if len(self.ends) == 2 and (self.name or self.short_name):
            e1, e2 = self.ends
            name_part = self._name_part()
            spec = self.specialization_part()
            header = f"{prefix}{self.usage_kind()} {name_part}{spec}"
            body_prefix = "    " * (indent + 1)
            body_line = f"connect {SysMLElement._quote(e1.ref)} to {SysMLElement._quote(e2.ref)};"
            return f"{header} {{\n{body_prefix}{body_line}\n{prefix}}}"
        return ConnectionUsage.to_text(self, indent)

class UseCaseAssociationUsage(ConnectionUsage):
    def usage_kind(self) -> str:
        return "usecaseassociation"

    def to_text(self, indent: int = 0) -> str:
        prefix = "    " * indent
        if len(self.ends) == 2 and (self.name or self.short_name):
            e1, e2 = self.ends
            name_part = self._name_part()
            spec = self.specialization_part()
            header = f"{prefix}{self.usage_kind()} {name_part}{spec}"
            body_prefix = "    " * (indent + 1)
            body_line = f"connect {SysMLElement._quote(e1.ref)} to {SysMLElement._quote(e2.ref)};"
            return f"{header} {{\n{body_prefix}{body_line}\n{prefix}}}"
        return ConnectionUsage.to_text(self, indent)

class UseCaseIncludeUsage(ConnectionUsage):
    def usage_kind(self) -> str:
        return "usecaseinclude"

    def to_text(self, indent: int = 0) -> str:
        prefix = "    " * indent
        if len(self.ends) == 2 and (self.name or self.short_name):
            e1, e2 = self.ends
            name_part = self._name_part()
            spec = self.specialization_part()
            header = f"{prefix}{self.usage_kind()} {name_part}{spec}"
            body_prefix = "    " * (indent + 1)
            body_line = f"connect {SysMLElement._quote(e1.ref)} to {SysMLElement._quote(e2.ref)};"
            return f"{header} {{\n{body_prefix}{body_line}\n{prefix}}}"
        return ConnectionUsage.to_text(self, indent)

class UseCaseExtendUsage(ConnectionUsage):
    def usage_kind(self) -> str:
        return "usecaseextend"

    def to_text(self, indent: int = 0) -> str:
        prefix = "    " * indent
        if len(self.ends) == 2 and (self.name or self.short_name):
            e1, e2 = self.ends
            name_part = self._name_part()
            spec = self.specialization_part()
            header = f"{prefix}{self.usage_kind()} {name_part}{spec}"
            body_prefix = "    " * (indent + 1)
            body_line = f"connect {SysMLElement._quote(e1.ref)} to {SysMLElement._quote(e2.ref)};"
            return f"{header} {{\n{body_prefix}{body_line}\n{prefix}}}"
        return ConnectionUsage.to_text(self, indent)

class CommandDef(Definition):
    """运维命令定义"""
    def __init__(self, name: Optional[str] = None, short_name: Optional[str] = None,
                 is_abstract: bool = False, is_variation: bool = False,
                 supertypes: List[str] = [],
                 command_text: str = "",
                 target_device: Optional[str] = None,
                 invocation: Optional[str] = None,
                 source_section: Optional[str] = None):
        super().__init__(name, short_name, is_abstract, is_variation, supertypes)
        self.command_text = command_text
        self.target_device = target_device
        self.invocation = invocation
        self.source_section = source_section

    def def_kind(self) -> str:
        return "command def"

    def to_text(self, indent: int = 0) -> str:
        prefix = "    " * indent
        name_part = self._name_part()
        header = f"{prefix}{self.def_prefix()}{self.def_kind()} {name_part}"
        body_lines = []
        if self.command_text:
            body_lines.append(f"{prefix}    doc /* command_text = {self.command_text} */;")
        if self.target_device:
            body_lines.append(f"{prefix}    doc /* target_device = {self.target_device} */;")
        if self.invocation:
            body_lines.append(f"{prefix}    doc /* invocation = {self.invocation} */;")
        if self.source_section:
            body_lines.append(f"{prefix}    doc /* source_section = {self.source_section} */;")
        body = self.members_to_text(indent + 1)
        if body:
            body_lines.append(body)
        if body_lines:
            return f"{header} {{\n" + "\n".join(body_lines) + f"\n{prefix}}}"
        else:
            return f"{header};"


class RequirementDef(Definition):
    def def_kind(self) -> str:
        return "requirement def"

class RequirementUsage(Usage):
    def usage_kind(self) -> str:
        return "requirement"


# ---------- 导入与别名 ----------
class Import(SysMLElement):
    def __init__(self, imported_path: str, visibility: VisibilityKind = VisibilityKind.PRIVATE,
                 is_recursive: bool = False, is_all: bool = False):
        super().__init__()
        self.imported_path = imported_path
        self.visibility = visibility
        self.is_recursive = is_recursive
        self.is_all = is_all

    def to_text(self, indent: int = 0) -> str:
        prefix = "    " * indent
        vis = self.visibility.value if self.visibility != VisibilityKind.PRIVATE else ""
        rec = "::**" if self.is_recursive else "::*" if self.is_all else ""
        return f"{prefix}{vis} import {self.imported_path}{rec};"


class Alias(SysMLElement):
    def __init__(self, alias_name: str, target_path: str, visibility: VisibilityKind = VisibilityKind.PUBLIC):
        super().__init__(alias_name)
        self.target_path = target_path
        self.visibility = visibility

    def to_text(self, indent: int = 0) -> str:
        prefix = "    " * indent
        vis = self.visibility.value if self.visibility != VisibilityKind.PUBLIC else ""
        return f"{prefix}{vis} alias {self.name} for {self.target_path};"