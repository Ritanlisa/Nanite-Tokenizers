"""
SysML 模型管理器：加载、保存、查询、修改模型。
支持别名注册表、实体元数据和知识图谱构建。
"""

import json
import re
from pathlib import Path
from typing import Optional, List, Union, Dict, Any, TypeVar, cast
from .sysml_model import (
    SysMLElement,
    Namespace,
    Package,
    Definition,
    Usage,
    Alias,
    Import,
    ConnectionUsage,
    InterfaceUsage,
    AllocationUsage,
    ContainmentUsage,
    CompositionUsage,
    ReferenceUsage,
    GeneralizationUsage,
    DependencyUsage,
    AbstractionUsage,
    RealizationUsage,
    DeriveUsage,
    TraceUsage,
    DeriveReqtUsage,
    RefineUsage,
    SatisfyUsage,
    VerifyUsage,
    CopyUsage,
    UseCaseAssociationUsage,
    UseCaseIncludeUsage,
    UseCaseExtendUsage,
    ConnectionEnd,
    PartDef, PartUsage,
    AttributeDef, AttributeUsage,
    PortDef, PortUsage,
    ItemDef, ItemUsage,
    ConnectionDef,
    InterfaceDef,
    AllocationDef,
    CommandDef,
    RequirementDef, RequirementUsage,
)
from .sysml_parser import parse_sysml_text

T = TypeVar('T', bound=SysMLElement)

ENTITY_CLASS_MAP: Dict[str, type] = {
    "PartDef": PartDef, "PartUsage": PartUsage,
    "AttributeDef": AttributeDef, "AttributeUsage": AttributeUsage,
    "PortDef": PortDef, "PortUsage": PortUsage,
    "ItemDef": ItemDef, "ItemUsage": ItemUsage,
    "ConnectionDef": ConnectionDef, "ConnectionUsage": ConnectionUsage,
    "InterfaceDef": InterfaceDef, "InterfaceUsage": InterfaceUsage,
    "AllocationDef": AllocationDef, "AllocationUsage": AllocationUsage,
    "CommandDef": CommandDef,
    "RequirementDef": RequirementDef, "RequirementUsage": RequirementUsage,
    "Package": Package,
}

RELATION_CLASS_MAP: Dict[str, type] = {
    "connection": ConnectionUsage,
    "interface": InterfaceUsage,
    "allocation": AllocationUsage,
    "containment": ContainmentUsage,
    "composition": CompositionUsage,
    "reference": ReferenceUsage,
    "generalization": GeneralizationUsage,
    "dependency": DependencyUsage,
    "abstraction": AbstractionUsage,
    "realization": RealizationUsage,
    "derive": DeriveUsage,
    "trace": TraceUsage,
    "derivereqt": DeriveReqtUsage,
    "refine": RefineUsage,
    "satisfy": SatisfyUsage,
    "verify": VerifyUsage,
    "copy": CopyUsage,
    "usecaseassociation": UseCaseAssociationUsage,
    "usecaseinclude": UseCaseIncludeUsage,
    "usecaseextend": UseCaseExtendUsage,
}


class AliasRegistry:
    """实体名称与别名注册表，支持多策略模糊搜索和归一化"""

    def __init__(self):
        self._alias_map: Dict[str, str] = {}
        self._aliases_by_entity: Dict[str, List[str]] = {}

    def register(self, qualified_name: str, aliases: List[str]) -> None:
        if not aliases:
            return
        existing = self._aliases_by_entity.setdefault(qualified_name, [])
        seen = set(self._normalize(a) for a in existing)
        for alias in aliases:
            alias = alias.strip()
            if not alias:
                continue
            norm = self._normalize(alias)
            if norm not in seen:
                self._alias_map[norm] = qualified_name
                existing.append(alias)
                seen.add(norm)

    def remove(self, qualified_name: str) -> None:
        aliases = self._aliases_by_entity.pop(qualified_name, [])
        for alias in aliases:
            norm = self._normalize(alias)
            if self._alias_map.get(norm) == qualified_name:
                del self._alias_map[norm]

    def transfer(self, source_qn: str, target_qn: str) -> None:
        source_aliases = self._aliases_by_entity.pop(source_qn, [])
        if not source_aliases:
            return
        existing = self._aliases_by_entity.setdefault(target_qn, [])
        for alias in source_aliases:
            self._alias_map[self._normalize(alias)] = target_qn
            if alias not in existing:
                existing.append(alias)

    def lookup(self, name: str) -> Optional[str]:
        return self._alias_map.get(self._normalize(name))

    def search(self, query: str, threshold: float = 0.3,
               regex_pattern: Optional[str] = None) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        query_norm = self._normalize(query)

        compiled_regex = None
        if regex_pattern:
            try:
                compiled_regex = re.compile(regex_pattern, re.IGNORECASE)
            except re.error:
                pass

        for qn, aliases in self._aliases_by_entity.items():
            entity_name = qn.split("::")[-1] if "::" in qn else qn
            entity_norm = self._normalize(entity_name)
            score = 0.0
            match_reason = ""

            if query_norm and entity_norm:
                if entity_norm == query_norm:
                    score = 1.0
                    match_reason = "exact"
                elif query_norm in entity_norm:
                    score = 0.7 + 0.2 * (len(query_norm) / max(len(entity_norm), 1))
                    match_reason = f"substring(main:{score:.2f})"
                elif entity_norm in query_norm:
                    score = 0.6 + 0.1 * (len(entity_norm) / max(len(query_norm), 1))
                    match_reason = f"substring(query:{score:.2f})"
                else:
                    overlap = len(set(query_norm) & set(entity_norm))
                    if overlap > 0:
                        score = 0.4 + 0.2 * (overlap / max(len(set(query_norm) | set(entity_norm)), 1))
                        match_reason = f"token_overlap:{score:.2f}"

            if compiled_regex:
                if compiled_regex.search(entity_name) or any(
                    compiled_regex.search(a) for a in aliases
                ):
                    score = max(score, 0.8)
                    match_reason = (match_reason + " regex" if match_reason else "regex")

            for alias in aliases:
                alias_norm = self._normalize(alias)
                if alias_norm == query_norm:
                    score = 0.90
                    match_reason = "alias_exact"
                    break
                elif query_norm in alias_norm:
                    score = max(score, 0.65)
                    match_reason = "alias_substring"
                elif alias_norm in query_norm:
                    score = max(score, 0.55)
                    match_reason = "alias_superstring"

            if score >= threshold:
                results.append({
                    "qualified_name": qn,
                    "entity_name": entity_name,
                    "confidence": round(score, 3),
                    "match_reason": match_reason,
                    "matched_aliases": [
                        a for a in aliases
                        if query_norm in self._normalize(a)
                    ] if query_norm else [],
                })

        results.sort(key=lambda x: -x["confidence"])
        return results[:30]

    def get_aliases(self, qualified_name: str) -> List[str]:
        return list(self._aliases_by_entity.get(qualified_name, []))

    @staticmethod
    def normalize(name: str) -> str:
        return AliasRegistry._normalize(name)

    @staticmethod
    def _normalize(name: str) -> str:
        # 全角 → 半角
        result_chars: List[str] = []
        for ch in name:
            code = ord(ch)
            if 0xFF01 <= code <= 0xFF5E:
                result_chars.append(chr(code - 0xFEE0))
            elif code == 0x3000:
                result_chars.append(" ")
            else:
                result_chars.append(ch)
        name = "".join(result_chars)
        # 小写 + 去除非字母数字和中文
        name = name.lower().strip()
        name = re.sub(r"[^a-z0-9\u4e00-\u9fff\u3400-\u4dbf]", "", name)
        return name

    def to_dict(self) -> Dict[str, Any]:
        return {"aliases": self._aliases_by_entity, "version": 1}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AliasRegistry":
        reg = cls()
        for qn, aliases in data.get("aliases", {}).items():
            reg._aliases_by_entity[qn] = list(aliases)
            for alias in aliases:
                reg._alias_map[reg._normalize(alias)] = qn
        return reg


class SysMLManager:
    def __init__(self, workspace_root: Optional[Path] = None):
        self.workspace_root = workspace_root or Path.cwd()
        self.root_elements: List[SysMLElement] = []
        self.current_model_file: Optional[Path] = None
        self._entity_metadata: Dict[str, Dict[str, Any]] = {}
        self._alias_registry = AliasRegistry()

    def _meta_path(self, file_path: Path) -> Path:
        return file_path.with_suffix(".meta.json")

    def load_from_file(self, file_path: Union[str, Path]) -> None:
        path = Path(file_path)
        if not path.is_absolute():
            path = self.workspace_root / path
        with open(path, 'r', encoding='utf-8') as f:
            text = f.read()
        self.root_elements = parse_sysml_text(text)
        self.current_model_file = path
        meta_path = self._meta_path(path)
        if meta_path.exists():
            try:
                with open(meta_path, 'r', encoding='utf-8') as f:
                    meta_data = json.load(f)
                self._entity_metadata = meta_data.get("entities", {})
                self._alias_registry = AliasRegistry.from_dict(
                    meta_data.get("aliases", {"version": 1, "aliases": {}})
                )
            except (json.JSONDecodeError, KeyError) as e:
                self._entity_metadata = {}
                self._alias_registry = AliasRegistry()

    def save_to_file(self, file_path: Optional[Union[str, Path]] = None) -> None:
        if file_path is None:
            if self.current_model_file is None:
                raise ValueError("No file path specified and no current model file.")
            file_path = self.current_model_file
        else:
            file_path = Path(file_path)
            if not file_path.is_absolute():
                file_path = self.workspace_root / file_path
        text = self.to_text()
        # 原子写入：先写临时文件，再 rename 覆盖
        tmp_path = file_path.with_suffix(file_path.suffix + ".tmp")
        meta_path = self._meta_path(file_path)
        tmp_meta = meta_path.with_suffix(meta_path.suffix + ".tmp")
        try:
            with open(tmp_path, 'w', encoding='utf-8') as f:
                f.write(text)
            # Export relations by parsing connect statements from to_text()
            rels_list = []
            import re as _re
            for r in self.get_all_relations():
                txt = r.to_text() if hasattr(r, 'to_text') else str(r)
                m = _re.search(r"connect\s+'((?:[^']|'')*?)'\s+to\s+'((?:[^']|'')*?)'\s*;", txt)
                if not m:
                    m = _re.search(r"connect\s+([A-Za-z_][A-Za-z_0-9]*)\s+to\s+([A-Za-z_][A-Za-z_0-9]*)\s*;", txt)
                if m:
                    rels_list.append({
                        "source": m.group(1).strip(),
                        "target": m.group(2).strip(),
                        "type": type(r).__name__,
                        "name": r.name,
                    })
            with open(tmp_meta, 'w', encoding='utf-8') as f:
                json.dump({
                    "entities": self._entity_metadata,
                    "relations": rels_list,
                    "aliases": self._alias_registry.to_dict(),
                    "version": 3,
                }, f, ensure_ascii=False, indent=2)
            tmp_path.replace(file_path)
            tmp_meta.replace(meta_path)
        finally:
            if tmp_path.exists():
                tmp_path.unlink(missing_ok=True)
            if tmp_meta.exists():
                tmp_meta.unlink(missing_ok=True)
        self.current_model_file = file_path

    def to_text(self) -> str:
        return "\n\n".join(elem.to_text() for elem in self.root_elements)

    def add_element(self, element: SysMLElement, parent: Optional[Namespace] = None) -> None:
        if parent is None:
            self.root_elements.append(element)
        else:
            # 确保 element 是 Namespace 允许的成员类型
            if not isinstance(element, (Definition, Usage, Alias, Import, Package)):
                raise TypeError(
                    f"Cannot add element of type '{type(element).__name__}' to a namespace. "
                    "Only Definition, Usage, Alias, Import, or Package are allowed."
                )
            parent.add_member(element)

    def find_package(self, qualified_name: str) -> Optional[Package]:
        parts = qualified_name.split("::")
        # 简化搜索，只支持单层包名
        for elem in self.root_elements:
            if isinstance(elem, Package) and elem.name == parts[0]:
                return elem
        return None

    def find_definition(self, name: str) -> Optional[Definition]:
        def search_in(ns: Namespace):
            for m in ns.members:
                if isinstance(m, Definition) and m.name == name:
                    return m
                if isinstance(m, Namespace):
                    found = search_in(m)
                    if found:
                        return found
            return None

        for elem in self.root_elements:
            if isinstance(elem, Definition) and elem.name == name:
                return elem
            if isinstance(elem, Namespace):
                found = search_in(elem)
                if found:
                    return found
        return None

    def get_all_entities(self) -> List[Definition]:
        """获取所有定义（用于知识图谱节点），包括根元素中的定义"""
        entities: List[Definition] = []
        def collect(ns: Namespace):
            for m in ns.members:
                if isinstance(m, Definition):
                    entities.append(m)
                if isinstance(m, Namespace):
                    collect(m)
        for elem in self.root_elements:
            if isinstance(elem, Definition):
                entities.append(elem)
            if isinstance(elem, Namespace):
                collect(elem)
        return entities

    def get_all_relations(self) -> List[Union[ConnectionUsage, InterfaceUsage, AllocationUsage]]:
        """获取所有关系使用（连接、接口、分配），包括根元素中的关系"""
        rels: List[Union[ConnectionUsage, InterfaceUsage, AllocationUsage]] = []
        def collect(ns: Namespace):
            for m in ns.members:
                if isinstance(m, (ConnectionUsage, InterfaceUsage, AllocationUsage)):
                    rels.append(m)
                if isinstance(m, Namespace):
                    collect(m)
        for elem in self.root_elements:
            if isinstance(elem, (ConnectionUsage, InterfaceUsage, AllocationUsage)):
                rels.append(elem)
            if isinstance(elem, Namespace):
                collect(elem)
        return rels

    def find_element(self, qualified_name: Optional[str] = None,
                    name: Optional[str] = None,
                    parent_package: Optional[str] = None) -> Optional[SysMLElement]:
        if qualified_name:
            parts = qualified_name.split("::")
            if not parts:
                return None
            search_name = parts[0]
            for elem in self.root_elements:
                if getattr(elem, 'name', '') == search_name:
                    if len(parts) == 1:
                        return elem
                    if isinstance(elem, Namespace):
                        return self._find_in_namespace(elem, parts[1:])
            return None

        if name:
            search_ns = self._resolve_namespace(parent_package) if parent_package else None
            if search_ns is None:
                for elem in self.root_elements:
                    if getattr(elem, 'name', '') == name:
                        return elem
                    if isinstance(elem, Namespace):
                        found = self._find_by_name_in_ns(elem, name)
                        if found:
                            return found
            else:
                return self._find_by_name_in_ns(search_ns, name)
        return None

    def _find_in_namespace(self, ns: Namespace, path: List[str]) -> Optional[SysMLElement]:
        """递归查找限定名路径"""
        for m in ns.members:
            if m.name == path[0]:
                if len(path) == 1:
                    return m
                if isinstance(m, Namespace):
                    return self._find_in_namespace(m, path[1:])
        return None

    def _find_by_name_in_ns(self, ns: Namespace, name: str) -> Optional[SysMLElement]:
        """在命名空间中按名称查找一个元素（非递归）"""
        for m in ns.members:
            if m.name == name:
                return m
        return None

    def _resolve_namespace(self, qual_name: Optional[str]) -> Optional[Namespace]:
        """解析命名空间限定名"""
        if not qual_name:
            return None
        parts = qual_name.split("::")
        for elem in self.root_elements:
            if isinstance(elem, Namespace) and elem.name == parts[0]:
                if len(parts) == 1:
                    return elem
                # 递归查找子命名空间
                return self._find_namespace_in(elem, parts[1:])
        return None

    def _find_namespace_in(self, ns: Namespace, path: List[str]) -> Optional[Namespace]:
        for m in ns.members:
            if isinstance(m, Namespace) and m.name == path[0]:
                if len(path) == 1:
                    return m
                return self._find_namespace_in(m, path[1:])
        return None

    def remove_element(self, element: SysMLElement) -> bool:
        owner_ns = self._find_owner_namespace(element)
        if owner_ns is not None:
            # 安全断言：命名空间中只包含符合条件的成员
            owner_ns.members.remove(cast(Union[Definition, Usage, Alias, Import], element))
            return True
        if element in self.root_elements:
            self.root_elements.remove(element)
            return True
        return False

    def _find_owner_namespace(self, target: SysMLElement) -> Optional[Namespace]:
        if target in self.root_elements:
            return None
        def search_in(ns: Namespace):
            if target in ns.members:
                return ns
            for m in ns.members:
                if isinstance(m, Namespace):
                    found = search_in(m)
                    if found:
                        return found
            return None
        for elem in self.root_elements:
            if isinstance(elem, Namespace):
                found = search_in(elem)
                if found:
                    return found
        return None

    def update_element(self, element: SysMLElement, **kwargs) -> None:
        """根据关键字更新元素的常见属性。"""
        if hasattr(element, 'name') and 'name' in kwargs:
            element.name = kwargs['name']
        if hasattr(element, 'short_name') and 'short_name' in kwargs:
            element.short_name = kwargs['short_name']
        if isinstance(element, Definition) and 'supertypes' in kwargs:
            element.supertypes = kwargs['supertypes']
        if isinstance(element, Usage) and 'type_refs' in kwargs:
            element.type_refs = kwargs['type_refs']
        if isinstance(element, Usage) and 'subsetted' in kwargs:
            element.subsetted = kwargs['subsetted']
        if isinstance(element, Usage) and 'redefined' in kwargs:
            element.redefined = kwargs['redefined']
        if isinstance(element, Usage) and 'multiplicity' in kwargs:
            element.multiplicity = kwargs['multiplicity']
        if isinstance(element, Usage) and 'direction' in kwargs:
            element.direction = kwargs['direction']
        if isinstance(element, Usage) and 'value_expr' in kwargs:
            element.value_expr = kwargs['value_expr']

    # ── Entity CRUD ─────────────────────────────────────────────

    def add_entity_with_metadata(
        self,
        entity_type: str,
        name: str,
        parent_package: Optional[str] = None,
        description: Optional[str] = None,
        aliases: Optional[List[str]] = None,
        source_sections: Optional[List[str]] = None,
        source_text: Optional[str] = None,
        properties: Optional[Dict[str, Any]] = None,
        supertypes: Optional[List[str]] = None,
        short_name: Optional[str] = None,
    ) -> Optional[SysMLElement]:
        cls = ENTITY_CLASS_MAP.get(entity_type)
        if cls is None:
            return None

        parent_ns: Optional[Namespace] = None
        if parent_package:
            resolved = self._resolve_namespace(parent_package)
            if resolved is None:
                pkg = Package(name=parent_package)
                self.root_elements.append(pkg)
                parent_ns = pkg
            elif isinstance(resolved, Namespace):
                parent_ns = resolved
            else:
                return None

        if issubclass(cls, Package):
            element = cls(name=name, short_name=short_name)
        elif issubclass(cls, Definition):
            sn = short_name if short_name and short_name.strip() != name.strip() else None
            element = cls(name=name, short_name=sn, supertypes=supertypes or [])
        elif issubclass(cls, Usage):
            sn = short_name if short_name and short_name.strip() != name.strip() else None
            element = cls(name=name, short_name=sn)
        else:
            element = cls(name=name, short_name=short_name)  # type: ignore[call-arg]

        self.add_element(element, parent=parent_ns)
        qn = element.qualified_name

        self._entity_metadata[qn] = {
            "description": description or "",
            "source_sections": source_sections or [],
            "source_text": source_text or "",
            "properties": properties or {},
        }
        self._alias_registry.register(qn, aliases or [])
        self._alias_registry.register(qn, [name])
        return element

    def update_entity_metadata(self, qualified_name: str, **kwargs) -> bool:
        entity = self.find_entity_by_qn(qualified_name)
        if entity is None:
            return False

        meta = self._entity_metadata.setdefault(qualified_name, {})
        if "append_description" in kwargs and kwargs["append_description"]:
            prev = meta.get("description", "")
            meta["description"] = (prev + "\n" + kwargs["append_description"]).strip()
        if "append_source_sections" in kwargs and kwargs["append_source_sections"]:
            existing = set(meta.get("source_sections", []))
            for s in kwargs["append_source_sections"]:
                if s and s not in existing:
                    existing.add(s)
            meta["source_sections"] = list(existing)
        if "merge_aliases" in kwargs and kwargs["merge_aliases"]:
            self._alias_registry.register(qualified_name, kwargs["merge_aliases"])
        if "update_properties" in kwargs and kwargs["update_properties"]:
            meta.setdefault("properties", {}).update(kwargs["update_properties"])
        if "new_name" in kwargs and kwargs["new_name"]:
            old_qn = entity.qualified_name
            entity.name = kwargs["new_name"]
            new_qn = entity.qualified_name
            if old_qn != new_qn:
                self._entity_metadata[new_qn] = self._entity_metadata.pop(old_qn, {})
                self._alias_registry.transfer(old_qn, new_qn)
                self._alias_registry.register(new_qn, [kwargs["new_name"]])
        if "supertypes" in kwargs and isinstance(entity, Definition):
            entity.supertypes = kwargs["supertypes"]
        if "type_refs" in kwargs and isinstance(entity, Usage):
            entity.type_refs = kwargs["type_refs"]
        return True

    def find_entity_by_qn(self, qualified_name: str) -> Optional[SysMLElement]:
        parts = qualified_name.split("::")
        if not parts:
            return None
        for elem in self.root_elements:
            if isinstance(elem, Namespace) and elem.name == parts[0]:
                if len(parts) == 1:
                    return elem
                return self._find_in_namespace(elem, parts[1:])
        return None

    def get_entity_metadata(self, qualified_name: str) -> Dict[str, Any]:
        return self._entity_metadata.get(qualified_name, {})

    def delete_entity(self, qualified_name: str) -> bool:
        entity = self.find_entity_by_qn(qualified_name)
        if entity is None:
            return False
        ok = self.remove_element(entity)
        if ok:
            self._entity_metadata.pop(qualified_name, None)
            self._alias_registry.remove(qualified_name)
        return ok

    def add_alias(self, qualified_name: str, alias: str) -> bool:
        entity = self.find_entity_by_qn(qualified_name)
        if entity is None:
            return False
        self._alias_registry.register(qualified_name, [alias])
        return True

    # ── Enhanced Search ─────────────────────────────────────────

    def search_entities(self, query: str, threshold: float = 0.3,
                        regex_pattern: Optional[str] = None) -> List[Dict[str, Any]]:
        alias_matches = self._alias_registry.search(query, threshold=threshold,
                                                     regex_pattern=regex_pattern)
        matched_qns = {m["qualified_name"] for m in alias_matches}
        results: List[Dict[str, Any]] = []

        for match in alias_matches:
            qn = match["qualified_name"]
            entity = self.find_entity_by_qn(qn)
            if entity is None:
                continue
            meta = self._entity_metadata.get(qn, {})
            results.append({
                "qualified_name": qn,
                "name": getattr(entity, "name", ""),
                "type": type(entity).__name__,
                "confidence": match["confidence"],
                "match_reason": match["match_reason"],
                "aliases": self._alias_registry.get_aliases(qn),
                "description": meta.get("description", "")[:200],
                "source_sections": meta.get("source_sections", []),
            })

        entities = self.get_all_entities()
        for entity in entities:
            qn = entity.qualified_name
            if qn in matched_qns:
                continue
            entity_name = entity.name or ""
            entity_norm = AliasRegistry.normalize(entity_name)
            query_norm = AliasRegistry.normalize(query)

            score = 0.0
            reason = ""
            if query_norm and entity_norm:
                if entity_norm == query_norm:
                    score = 1.0
                    reason = "exact"
                elif query_norm in entity_norm:
                    score = 0.7 + 0.2 * (len(query_norm) / max(len(entity_norm), 1))
                    reason = f"substring:{score:.2f}"
                elif entity_norm in query_norm:
                    score = 0.6
                    reason = "superstring"

            if score >= threshold:
                meta = self._entity_metadata.get(qn, {})
                results.append({
                    "qualified_name": qn,
                    "name": entity_name,
                    "type": type(entity).__name__,
                    "confidence": round(score, 3),
                    "match_reason": reason,
                    "aliases": self._alias_registry.get_aliases(qn),
                    "description": meta.get("description", "")[:200],
                    "source_sections": meta.get("source_sections", []),
                })

        results.sort(key=lambda x: -x["confidence"])
        return results[:30]

    # ── Relation CRUD ───────────────────────────────────────────

    def add_relation(
        self,
        relation_type: str,
        source_name: str,
        target_name: str,
        name: Optional[str] = None,
        parent_package: Optional[str] = None,
        description: Optional[str] = None,
        role_source: Optional[str] = None,
        role_target: Optional[str] = None,
        source_sections: Optional[List[str]] = None,
        source_text: Optional[str] = None,
    ) -> Optional[Usage]:
        cls = RELATION_CLASS_MAP.get(relation_type)
        if cls is None:
            return None

        if name is None:
            name = f"{source_name}_{target_name}_{relation_type}"

        ends = [
            ConnectionEnd(ref=source_name, role=role_source),
            ConnectionEnd(ref=target_name, role=role_target),
        ]

        if cls is InterfaceUsage or cls is AllocationUsage:
            rel = cls(name=name, ends=ends)
        else:
            rel = ConnectionUsage(name=name, ends=ends)

        parent_ns: Optional[Namespace] = None
        if parent_package:
            resolved = self._resolve_namespace(parent_package)
            if resolved is None:
                pkg = Package(name=parent_package)
                self.root_elements.append(pkg)
                parent_ns = pkg
            elif isinstance(resolved, Namespace):
                parent_ns = resolved
            else:
                return None

        self.add_element(rel, parent=parent_ns)
        qn = rel.qualified_name
        if description or source_sections:
            self._entity_metadata[qn] = {
                "description": description or "",
                "source_sections": source_sections or [],
                "source_text": source_text or "",
                "properties": {},
            }
        return rel

    def delete_relation(self, name: str, parent_package: Optional[str] = None) -> bool:
        found: Optional[SysMLElement] = None
        parent_ns: Optional[Namespace] = None

        if parent_package:
            parent_ns = self._resolve_namespace(parent_package)
        if parent_ns:
            for m in parent_ns.members:
                if isinstance(m, (ConnectionUsage, InterfaceUsage, AllocationUsage)) and m.name == name:
                    found = m
                    break
        else:
            for elem in self.root_elements:
                if isinstance(elem, Namespace):
                    found = self._find_relation_in(elem, name)
                    if found:
                        break
                elif isinstance(elem, (ConnectionUsage, InterfaceUsage, AllocationUsage)) and elem.name == name:
                    found = elem
                    break

        if found is None:
            return False
        return self.remove_element(found)

    def _find_relation_in(self, ns: Namespace, name: str) -> Optional[SysMLElement]:
        for m in ns.members:
            if isinstance(m, (ConnectionUsage, InterfaceUsage, AllocationUsage)) and m.name == name:
                return m
            if isinstance(m, Namespace):
                found = self._find_relation_in(m, name)
                if found:
                    return found
        return None

    # ── Merge / Dedup ───────────────────────────────────────────

    def suggest_merges(self, threshold: float = 0.6) -> List[Dict[str, Any]]:
        suggestions: List[Dict[str, Any]] = []
        entities = self.get_all_entities()
        entity_infos: List[Dict[str, Any]] = []

        for e in entities:
            qn = e.qualified_name
            qn_norm = AliasRegistry.normalize(qn)
            name_norm = AliasRegistry.normalize(e.name or "")
            entity_infos.append({
                "qualified_name": qn,
                "name": e.name,
                "type": type(e).__name__,
                "qn_norm": qn_norm,
                "name_norm": name_norm,
                "aliases": self._alias_registry.get_aliases(qn),
            })

        seen_pairs: set = set()
        n = len(entity_infos)
        for i in range(n):
            for j in range(i + 1, n):
                a = entity_infos[i]
                b = entity_infos[j]
                if a["type"] != b["type"]:
                    continue
                pair_key = tuple(sorted([a["qualified_name"], b["qualified_name"]]))
                if pair_key in seen_pairs:
                    continue
                seen_pairs.add(pair_key)

                score = 0.0
                reasons: List[str] = []

                if a["qn_norm"] == b["qn_norm"]:
                    score = 1.0
                    reasons.append("identical_name")
                elif a["name_norm"] == b["name_norm"]:
                    score = max(score, 0.95)
                    reasons.append("same_name_different_package")
                if a["name_norm"] in b["qn_norm"] or b["name_norm"] in a["qn_norm"]:
                    score = max(score, 0.7)
                    reasons.append("name_is_substring")

                for alias_a in a["aliases"]:
                    norm_a = AliasRegistry.normalize(alias_a)
                    if norm_a == b["name_norm"]:
                        score = max(score, 0.85)
                        reasons.append(f"alias_matches_name:{alias_a}")
                    for alias_b in b["aliases"]:
                        if AliasRegistry.normalize(alias_b) == norm_a:
                            score = max(score, 0.9)
                            reasons.append(f"shared_alias:{alias_a}")
                            break

                if score >= threshold:
                    suggestions.append({
                        "entity_a": {"qualified_name": a["qualified_name"], "name": a["name"], "type": a["type"]},
                        "entity_b": {"qualified_name": b["qualified_name"], "name": b["name"], "type": b["type"]},
                        "confidence": round(score, 3),
                        "reasons": reasons,
                    })

        suggestions.sort(key=lambda x: -x["confidence"])
        return suggestions

    def merge_entities(self, source_qn: str, target_qn: str) -> Optional[str]:
        source = self.find_entity_by_qn(source_qn)
        target = self.find_entity_by_qn(target_qn)
        if source is None or target is None:
            return None

        source_name = source.name

        # 收集 source 所有可能的引用形式（用于后续关系重定向）
        source_all_refs: set = {source_qn}
        if source_name:
            source_all_refs.add(source_name)
            source_all_refs.add(AliasRegistry._normalize(source_name))
        for alias in self._alias_registry.get_aliases(source_qn):
            source_all_refs.add(alias)
            source_all_refs.add(AliasRegistry._normalize(alias))
        for alias_key, mapped_qn in list(self._alias_registry._alias_map.items()):
            if mapped_qn == source_qn:
                source_all_refs.add(alias_key)

        # 1. 将 source 自身名称注册为 target 的别名
        if source_name:
            self._alias_registry.register(target_qn, [source_name])

        # 2. 合并 metadata
        source_meta = self._entity_metadata.pop(source_qn, {})
        target_meta = self._entity_metadata.setdefault(target_qn, {
            "description": "", "source_sections": [], "source_text": "", "properties": {},
        })

        if source_meta.get("description"):
            prev = target_meta.get("description", "")
            if prev and source_meta["description"] not in prev:
                target_meta["description"] = (prev + "\n" + source_meta["description"]).strip()
            elif not prev:
                target_meta["description"] = source_meta["description"]

        existing_sections = set(target_meta.get("source_sections", []))
        for s in source_meta.get("source_sections", []):
            if s and s not in existing_sections:
                existing_sections.add(s)
        target_meta["source_sections"] = list(existing_sections)

        if source_meta.get("properties"):
            target_meta.setdefault("properties", {}).update(source_meta["properties"])

        # 3. 转移别名注册表
        self._alias_registry.transfer(source_qn, target_qn)

        # 4. 重定向 source 的所有关系到 target
        target_name = target.name
        for rel in self.get_all_relations():
            if not hasattr(rel, "ends"):
                continue
            for end in rel.ends:
                if end.ref in source_all_refs:
                    end.ref = target_name
            if rel.name and source_name:
                for old_ref in source_all_refs:
                    if old_ref in rel.name:
                        rel.name = rel.name.replace(old_ref, target_name)
                        break

        # 5. 删除 source 元素
        self.remove_element(source)
        return target_qn


# 全局单例（便于工具调用）
_manager_instance: Optional[SysMLManager] = None

def get_sysml_manager() -> SysMLManager:
    global _manager_instance
    if _manager_instance is None:
        _manager_instance = SysMLManager()
    return _manager_instance