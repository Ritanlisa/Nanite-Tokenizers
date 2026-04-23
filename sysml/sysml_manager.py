"""
SysML 模型管理器：加载、保存、查询、修改模型。
"""

import os
from pathlib import Path
from typing import Optional, List, Union
from sysml_model import *
from sysml_parser import parse_sysml_text


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
            if not isinstance(element, (Definition, Usage, Alias, Import)):
                raise TypeError(
                    f"Cannot add element of type '{type(element).__name__}' to a namespace. "
                    "Only Definition, Usage, Alias, or Import are allowed."
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


# 全局单例（便于工具调用）
_manager_instance: Optional[SysMLManager] = None

def get_sysml_manager() -> SysMLManager:
    global _manager_instance
    if _manager_instance is None:
        _manager_instance = SysMLManager()
    return _manager_instance