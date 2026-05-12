"""
SysML 2.0 文本解析器 (Lark 1.x 兼容实现)
使用子规则替代内联命名捕获，Earley 解析器 + 后置 Transformer。
"""
from typing import cast
from lark import Lark, Transformer, v_args
from .sysml_model import *

SYML_GRAMMAR = r"""
    start: (package | definition | usage | import_stmt | alias_stmt | connect_usage)*

    package: "package" IDENTIFIER package_body
    package_body: "{" member* "}" | ";"
    member: definition | usage | import_stmt | alias_stmt | connect_usage | package | doc_stmt
    doc_stmt: "doc" (STRING | IDENTIFIER) ";"

    definition: def_prefix def_kind IDENTIFIER supertypes_opt? definition_body
    def_prefix: (ABSTRACT? VARIATION? | ABSTRACT | VARIATION)?
    def_kind: "part" "def" -> part_def
            | "attribute" "def" -> attribute_def
            | "port" "def" -> port_def
            | "item" "def" -> item_def
            | "connection" "def" -> connection_def
            | "interface" "def" -> interface_def
            | "allocation" "def" -> allocation_def
            | "requirement" "def" -> requirement_def
    definition_body: "{" member* "}" | ";"

    usage: usage_prefix usage_kind IDENTIFIER multiplicity_opt? specialization_opt? value_opt? usage_body
    usage_prefix: (direction | DERIVED | ABSTRACT | CONSTANT | REF)*
    usage_kind: "part" -> part_usage
              | "attribute" -> attribute_usage
              | "port" -> port_usage
              | "item" -> item_usage
              | "connection" -> connection_usage
              | "interface" -> interface_usage
              | "allocation" -> allocation_usage
              | "requirement" -> requirement_usage
    usage_body: "{" member* "}" | ";"

    direction: "in" | "out" | "inout"
    direction_opt: ["in" | "out" | "inout"]
    multiplicity: "[" NUMBER? ".." NUMBER? "]" (ORDERED? NONUNIQUE?)?
    multiplicity_opt: [multiplicity]
    ORDERED: "ordered"
    NONUNIQUE: "nonunique"

    specialization: (":" type_refs | "subsets" subset_refs | "redefines" redef_refs)+
    specialization_opt: [specialization]
    type_refs: IDENTIFIER ("," IDENTIFIER)*
    subset_refs: IDENTIFIER ("," IDENTIFIER)*
    redef_refs: IDENTIFIER ("," IDENTIFIER)*

    value: "=" expr
    value_opt: [value]

    supertypes: ":>" IDENTIFIER ("," IDENTIFIER)*
    supertypes_opt: [supertypes]

    import_stmt: visibility_opt? "import" IDENTIFIER ("::**" | "::*")? ";"
    alias_stmt: visibility_opt? "alias" IDENTIFIER "for" IDENTIFIER ";"
    visibility_opt: ["public" | "private" | "protected"]

    // 连接简写（可带 "connection" 前缀）
    connect_usage: ["connection"] "connect" IDENTIFIER "to" IDENTIFIER ";"

    expr: /[^;\n]+/

    ABSTRACT: "abstract"
    VARIATION: "variation"
    DERIVED: "derived"
    CONSTANT: "constant"
    REF: "ref"

    IDENTIFIER: /[a-zA-Z_\u4e00-\u9fff\u3400-\u4dbf][-a-zA-Z0-9_.\u4e00-\u9fff\u3400-\u4dbf]*/ | "'" /[^']+/ "'"
    STRING: /"[^"]*"/
    NUMBER: /\d+/

    %import common.WS
    %ignore WS
    %ignore /\/\*[\s\S]*?\*\//
    %ignore /\/\/.*/
"""

@v_args(inline=True)
class SysMLTransformer(Transformer):
    def start(self, *items):
        return list(items)

    def package(self, name_token, body):
        pkg = Package(self._id_str(name_token))
        members = body[0] if isinstance(body, tuple) else body
        for m in (members or []):
            pkg.add_member(m)
        return pkg

    def package_body(self, *items):
        members = []
        for item in items:
            if isinstance(item, list):
                members.extend(item)
            elif item is not None and not isinstance(item, str):
                members.append(item)
        return members

    def member(self, item):
        return item

    def definition(self, prefix, kind, name_token, supertypes, body):
        defn = kind
        defn.name = str(name_token).strip("'")
        if supertypes:
            defn.supertypes = [str(t) for t in supertypes]
        members = body if isinstance(body, list) else (body[0] if isinstance(body, tuple) and len(body) > 0 else [])
        for m in (members if isinstance(members, list) else []):
            defn.add_member(m)
        return defn

    def definition_body(self, *items):
        members = []
        for item in items:
            if isinstance(item, list):
                members.extend(item)
            elif item is not None and not isinstance(item, str):
                members.append(item)
        return members

    def def_prefix(self, *tokens):
        return None

    # ── def_kind 方法 ──
    def part_def(self):
        return PartDef()

    def attribute_def(self):
        return AttributeDef()

    def port_def(self):
        return PortDef()

    def item_def(self):
        return ItemDef()

    def connection_def(self):
        return ConnectionDef()

    def interface_def(self):
        return InterfaceDef()

    def allocation_def(self):
        return AllocationDef()

    def requirement_def(self):
        return RequirementDef()

    # ── usage ──
    def usage(self, prefix, kind, name_token, multiplicity, specialization, value, body):
        usage_obj = kind
        usage_obj.name = str(name_token).strip("'")
        if prefix:
            usage_obj.is_reference = "ref" in prefix
            usage_obj.is_abstract = "abstract" in prefix
            usage_obj.is_derived = "derived" in prefix
            usage_obj.is_constant = "constant" in prefix
        self._apply_usage_props(usage_obj, multiplicity, specialization, value)
        members = body if isinstance(body, list) else (body[0] if isinstance(body, tuple) else [])
        for m in (members or []):
            if isinstance(m, tuple):
                for sub in m:
                    usage_obj.add_member(sub)
            elif m is not None:
                usage_obj.add_member(m)
        return usage_obj

    def usage_body(self, *items):
        members = []
        for item in items:
            if isinstance(item, list):
                members.extend(item)
            elif item is not None and not isinstance(item, str):
                members.append(item)
        return members

    def usage_prefix(self, *tokens):
        flags = set()
        for t in tokens:
            if isinstance(t, str):
                flags.add(t.lower())
        return flags if flags else None

    # ── usage_kind 方法 ──
    def part_usage(self):
        return PartUsage()

    def attribute_usage(self):
        return AttributeUsage()

    def port_usage(self):
        return PortUsage()

    def item_usage(self):
        return ItemUsage()

    def connection_usage(self):
        return ConnectionUsage()

    def interface_usage(self):
        return InterfaceUsage()

    def allocation_usage(self):
        return AllocationUsage()

    def requirement_usage(self):
        return RequirementUsage()

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

    # ── 连接简写 ──
    def connect_usage(self, *args):
        """处理 connection connect 'A' to 'B' 或 connect 'A' to 'B'"""
        src = None
        tgt = None
        for a in args:
            s = str(a).strip("'") if a is not None else ""
            if s in ("connection", "connect", "to"):
                continue
            if src is None:
                src = s
            elif tgt is None:
                tgt = s
        usage = ConnectionUsage()
        usage.ends = [ConnectionEnd(str(src or "")), ConnectionEnd(str(tgt or ""))]
        return usage

    # ── 辅助规则 ──
    def direction(self, token):
        return str(token)

    def direction_opt(self, token=None):
        return str(token) if token else None

    def multiplicity(self, lower=None, upper=None, ordered=False, nonunique=False):
        lower = str(lower) if lower is not None else None
        upper = str(upper) if upper is not None else None
        return Multiplicity(lower, upper, ordered=bool(ordered), unique=not nonunique)

    def multiplicity_opt(self, mult=None):
        return mult

    def specialization(self, *parts):
        result = {'type': [], 'subsets': [], 'redefines': []}
        guard = None
        for p in parts:
            if isinstance(p, list):
                target = guard or 'type'
                result[target] = [str(t) for t in p]
                guard = None
            elif p == 'subsets':
                guard = 'subsets'
            elif p == 'redefines':
                guard = 'redefines'
            # ':' literals are dropped by Lark; skip them
        return result

    def specialization_opt(self, spec=None):
        return spec

    def type_refs(self, *refs):
        r = [str(x) for x in refs if not isinstance(x, str) or x != ',']
        return r

    def subset_refs(self, *refs):
        r = [str(x) for x in refs if not isinstance(x, str) or x != ',']
        return r

    def redef_refs(self, *refs):
        r = [str(x) for x in refs if not isinstance(x, str) or x != ',']
        return r

    def value(self, expr_token):
        return str(expr_token).strip()

    def value_opt(self, val=None):
        return val

    def supertypes(self, *types):
        return [str(t) for t in types if not isinstance(t, str) or t != ',']

    def supertypes_opt(self, types=None):
        return types

    def import_stmt(self, visibility, imported_path, suffix=None):
        vis = VisibilityKind(str(visibility)) if visibility else VisibilityKind.PRIVATE
        is_rec = str(suffix).strip() == "::**" if suffix else False
        is_all = str(suffix).strip() == "::*" if suffix else False
        return Import(str(imported_path).strip("'"), vis, is_rec, is_all)

    def alias_stmt(self, visibility, alias_name, target_path):
        vis = VisibilityKind(str(visibility)) if visibility else VisibilityKind.PUBLIC
        return Alias(str(alias_name).strip("'"), str(target_path).strip("'"), vis)

    def visibility_opt(self, token=None):
        return str(token) if token else None

    def expr(self, token):
        return str(token).strip()

    def doc_stmt(self, *args):
        from .sysml_model import Doc
        text = str(args[0]).strip('"') if args else ""
        return Doc(text=text)

    def IDENTIFIER(self, token):
        return str(token).strip("'")

    def NUMBER(self, token):
        return str(token)

    @staticmethod
    def _id_str(value):
        if isinstance(value, (str,)):
            return value.strip("'")
        return str(value)


def parse_sysml_text(text: str) -> List[SysMLElement]:
    parser = Lark(SYML_GRAMMAR)
    tree = parser.parse(text)
    transformer = SysMLTransformer()
    result = transformer.transform(tree)
    return cast(List[SysMLElement], result)
