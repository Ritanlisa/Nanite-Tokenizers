"""
SysML 模型管理器：加载、保存、查询、修改模型。
"""

from pathlib import Path
from typing import Optional, List, Union, TypeVar, cast
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
)
from .sysml_parser import parse_sysml_text

T = TypeVar('T', bound=SysMLElement)

class SysMLManager:
    def __init__(self, workspace_root: Optional[Path] = None):
        self.workspace_root = workspace_root or Path.cwd()
        self.root_elements: List[SysMLElement] = []  # 顶层元素（通常为包）
        self.current_model_file: Optional[Path] = None

    def load_from_file(self, file_path: Union[str, Path]) -> None:
        path = Path(file_path)
        if not path.is_absolute():
            path = self.workspace_root / path
        with open(path, 'r', encoding='utf-8') as f:
            text = f.read()
        self.root_elements = parse_sysml_text(text)
        self.current_model_file = path

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
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(text)
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
            if isinstance(elem, Namespace):
                found = search_in(elem)
                if found:
                    return found
        return None

    def get_all_entities(self) -> List[Definition]:
        """获取所有定义（用于知识图谱节点）"""
        entities = []
        def collect(ns: Namespace):
            for m in ns.members:
                if isinstance(m, Definition):
                    entities.append(m)
                if isinstance(m, Namespace):
                    collect(m)
        for elem in self.root_elements:
            if isinstance(elem, Namespace):
                collect(elem)
        return entities

    def get_all_relations(self) -> List[Union[ConnectionUsage, InterfaceUsage, AllocationUsage]]:
        """获取所有关系使用（连接、接口、分配）"""
        rels = []
        def collect(ns: Namespace):
            for m in ns.members:
                if isinstance(m, (ConnectionUsage, InterfaceUsage, AllocationUsage)):
                    rels.append(m)
                if isinstance(m, Namespace):
                    collect(m)
        for elem in self.root_elements:
            if isinstance(elem, Namespace):
                collect(elem)
        return rels

    def find_element(self, qualified_name: Optional[str] = None,
                    name: Optional[str] = None,
                    parent_package: Optional[str] = None) -> Optional[SysMLElement]:
        """
        查找元素。支持通过限定名，或者名称+父包定位。
        """
        # 优先使用限定名
        if qualified_name:
            parts = qualified_name.split("::")
            # 简化：假设第一部分是顶层包名
            for elem in self.root_elements:
                if isinstance(elem, Package) and elem.name == parts[0]:
                    if len(parts) == 1:
                        return elem
                    # 递归在包内查找
                    return self._find_in_namespace(elem, parts[1:])
            return None

        # 通过名称和父包查找
        if name:
            search_ns = self._resolve_namespace(parent_package) if parent_package else None
            if search_ns is None:
                # 在所有顶层元素及递归中查找
                for elem in self.root_elements:
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
        """递归搜索包含target的命名空间。"""
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
        # 可根据需要扩展其他属性

# 全局单例（便于工具调用）
_manager_instance: Optional[SysMLManager] = None

def get_sysml_manager() -> SysMLManager:
    global _manager_instance
    if _manager_instance is None:
        _manager_instance = SysMLManager()
    return _manager_instance