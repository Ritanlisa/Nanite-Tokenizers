"""
SysML 2.0 文本解析器 (Lark 实现)
"""

from typing import cast

from lark import Lark, Transformer, v_args
from sysml_model import *

# 语法定义（SysML 2.0 子集）
SYML_GRAMMAR = r"""
    start: (package | definition | usage | import_stmt | alias_stmt)*

    package: "package" name=IDENTIFIER ("{" members+=member* "}" | ";")
    member: definition | usage | import_stmt | alias_stmt

    definition: def_prefix def_kind name=IDENTIFIER supertypes? ("{" members+=member* "}" | ";")
    def_prefix: (ABSTRACT? VARIATION? | ABSTRACT | VARIATION)?
    def_kind: "part" "def" -> part_def
            | "attribute" "def" -> attribute_def
            | "port" "def" -> port_def
            | "item" "def" -> item_def
            | "connection" "def" -> connection_def
            | "interface" "def" -> interface_def
            | "allocation" "def" -> allocation_def
            | "requirement" "def" -> requirement_def

    usage: usage_prefix usage_kind name=IDENTIFIER multiplicity? specialization? value? ("{" members+=member* "}" | ";")
    usage_prefix: (direction? DERIVED? ABSTRACT? CONSTANT? REF?) | (direction | DERIVED | ABSTRACT | CONSTANT | REF)*
    usage_kind: "part" -> part_usage
              | "attribute" -> attribute_usage
              | "port" -> port_usage
              | "item" -> item_usage
              | "connection" -> connection_usage
              | "interface" -> interface_usage
              | "allocation" -> allocation_usage
              | "requirement" -> requirement_usage

    direction: "in" | "out" | "inout"
    multiplicity: "[" lower=NUMBER? ".." upper=NUMBER? "]" (ORDERED? NONUNIQUE?)?
    ORDERED: "ordered"
    NONUNIQUE: "nonunique"

    specialization: (":" type_refs | "subsets" subset_refs | "redefines" redef_refs)+
    type_refs: IDENTIFIER ("," IDENTIFIER)*
    subset_refs: IDENTIFIER ("," IDENTIFIER)*
    redef_refs: IDENTIFIER ("," IDENTIFIER)*

    value: "=" expr

    supertypes: ":>" IDENTIFIER ("," IDENTIFIER)*

    import_stmt: visibility? "import" imported_path ("::**" | "::*")? ";"
    alias_stmt: visibility? "alias" alias_name=IDENTIFIER "for" target_path=IDENTIFIER ";"
    visibility: "public" | "private" | "protected"

    // 连接简写
    connection_usage: "connect" src=IDENTIFIER "to" tgt=IDENTIFIER ";"

    expr: /[^;\n]+/

    ABSTRACT: "abstract"
    VARIATION: "variation"
    DERIVED: "derived"
    CONSTANT: "constant"
    REF: "ref"

    IDENTIFIER: /[a-zA-Z_][a-zA-Z0-9_]*/ | "'" /[^']+/ "'"
    NUMBER: /\d+/

    %import common.WS
    %ignore WS
    %ignore /\/\*.*?\*\//
    %ignore /\/\/.*/
"""

@v_args(inline=True)
class SysMLTransformer(Transformer):
    def __init__(self):
        super().__init__()
        self.current_package = None

    def start(self, *items):
        return list(items)

    def package(self, name, *members):
        pkg = Package(str(name))
        for m in members:
            if isinstance(m, list):
                for item in m:
                    pkg.add_member(item)
            else:
                pkg.add_member(m)
        return pkg

    def member(self, item):
        return item

    def part_def(self, name, supertypes=None):
        defn = PartDef(str(name))
        if supertypes:
            defn.supertypes = [str(t) for t in supertypes]
        return defn

    def attribute_def(self, name, supertypes=None):
        defn = AttributeDef(str(name))
        if supertypes:
            defn.supertypes = [str(t) for t in supertypes]
        return defn

    def port_def(self, name, supertypes=None):
        defn = PortDef(str(name))
        if supertypes:
            defn.supertypes = [str(t) for t in supertypes]
        return defn

    def item_def(self, name, supertypes=None):
        defn = ItemDef(str(name))
        if supertypes:
            defn.supertypes = [str(t) for t in supertypes]
        return defn

    def connection_def(self, name, supertypes=None):
        defn = ConnectionDef(str(name))
        if supertypes:
            defn.supertypes = [str(t) for t in supertypes]
        return defn

    def interface_def(self, name, supertypes=None):
        defn = InterfaceDef(str(name))
        if supertypes:
            defn.supertypes = [str(t) for t in supertypes]
        return defn

    def allocation_def(self, name, supertypes=None):
        defn = AllocationDef(str(name))
        if supertypes:
            defn.supertypes = [str(t) for t in supertypes]
        return defn

    def requirement_def(self, name, supertypes=None):
        defn = RequirementDef(str(name))
        if supertypes:
            defn.supertypes = [str(t) for t in supertypes]
        return defn

    def part_usage(self, name, multiplicity=None, specialization=None, value=None):
        usage = PartUsage(str(name))
        self._apply_usage_props(usage, multiplicity, specialization, value)
        return usage

    def attribute_usage(self, name, multiplicity=None, specialization=None, value=None):
        usage = AttributeUsage(str(name))
        self._apply_usage_props(usage, multiplicity, specialization, value)
        return usage

    def port_usage(self, name, multiplicity=None, specialization=None, value=None):
        usage = PortUsage(str(name))
        self._apply_usage_props(usage, multiplicity, specialization, value)
        return usage

    def item_usage(self, name, multiplicity=None, specialization=None, value=None):
        usage = ItemUsage(str(name))
        self._apply_usage_props(usage, multiplicity, specialization, value)
        return usage

    def connection_usage(self, name=None, multiplicity=None, specialization=None, value=None):
        # 连接简写处理在 connect 规则
        pass

    def connect(self, src, tgt):
        usage = ConnectionUsage()
        usage.ends = [ConnectionEnd(str(src)), ConnectionEnd(str(tgt))]
        return usage

    def _apply_usage_props(self, usage, multiplicity, specialization, value):
        if multiplicity:
            usage.multiplicity = multiplicity
        if specialization:
            if isinstance(specialization, dict):
                usage.type_refs = specialization.get('type', [])
                usage.subsetted = specialization.get('subsets', [])
                usage.redefined = specialization.get('redefines', [])
        if value:
            usage.value_expr = str(value)

    def multiplicity(self, lower=None, upper=None, ordered=False, nonunique=False):
        lower = str(lower) if lower else None
        upper = str(upper) if upper else None
        return Multiplicity(lower, upper, ordered=bool(ordered), unique=not nonunique)

    def specialization(self, *parts):
        result = {'type': [], 'subsets': [], 'redefines': []}
        i = 0
        while i < len(parts):
            if parts[i] == ':':
                result['type'] = [str(t) for t in parts[i+1]]
                i += 2
            elif parts[i] == 'subsets':
                result['subsets'] = [str(t) for t in parts[i+1]]
                i += 2
            elif parts[i] == 'redefines':
                result['redefines'] = [str(t) for t in parts[i+1]]
                i += 2
            else:
                i += 1
        return result

    def type_refs(self, *refs):
        return [str(r) for r in refs]

    def subset_refs(self, *refs):
        return [str(r) for r in refs]

    def redef_refs(self, *refs):
        return [str(r) for r in refs]

    def supertypes(self, *types):
        return [str(t) for t in types]

    def import_stmt(self, visibility=None, imported_path=None, recursive=None):
        vis = VisibilityKind(visibility) if visibility else VisibilityKind.PRIVATE
        is_rec = recursive == "::**"
        is_all = recursive == "::*"
        return Import(str(imported_path), vis, is_rec, is_all)

    def alias_stmt(self, visibility=None, alias_name=None, target_path=None):
        vis = VisibilityKind(visibility) if visibility else VisibilityKind.PUBLIC
        return Alias(str(alias_name), str(target_path), vis)

    def visibility(self, token):
        return str(token)

    def expr(self, token):
        return str(token).strip()

    def IDENTIFIER(self, token):
        return str(token).strip("'")


def parse_sysml_text(text: str) -> List[SysMLElement]:
    parser = Lark(SYML_GRAMMAR, parser='lalr', transformer=SysMLTransformer())
    result = parser.parse(text)
    # transformer 的 start 方法返回 List[SysMLElement]
    return cast(List[SysMLElement], result)