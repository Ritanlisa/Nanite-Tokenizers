#!/usr/bin/env python3
"""
手工注入已知 SysML 实体到知识图谱。
将文档中已知的运维命令、机柜映射、主机名标识写入 KG，
使 sysml_retrieve 可以直接匹配 mn0, yhst, R1P3 等用户查询关键字。

用法:
  python scripts/add_manual_kg_entities.py [--db AIOPS_New]
"""

import sys
import argparse
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from scripts.sysml_rag_mcp_server import (
    sysml_load_model,
    _sysml_save_model,
    sysml_add_command,
    sysml_set_hostname,
    sysml_add_cabinet_instance,
    sysml_add_relation,
    sysml_model_summary,
    _get_manager,
    _global_manager,
    _loaded_files,
)

DEFAULT_DB = "AIOPS_New"


def inject(db_name: str) -> None:
    kg_path = ROOT_DIR / "database" / db_name / "knowledge_graph.sysml"

    # ── Reset & load
    _global_manager = None
    _loaded_files.clear()
    result = sysml_load_model(str(kg_path))
    if not result.get("ok"):
        print(f"ERROR: Failed to load KG: {result}")
        return

    summary = sysml_model_summary()
    print(f"Loaded: {summary['total_entities']} entities, {summary['total_relations']} relations")

    # ── 1. 注入主机名标识 ──────────────────────────────────────
    print("\n--- Setting hostnames ---")
    r = sysml_set_hostname("ManagementNode", "mn0")
    print(f"  ManagementNode → mn0: {r.get('ok')} @ {r.get('qualified_name','?')}")
    print(f"    aliases: {r.get('aliases',[])}")

    # ── 2. 注入机柜实例 ────────────────────────────────────────
    print("\n--- Adding cabinet instances ---")
    r = sysml_add_cabinet_instance(
        cabinet_id="R1P3",
        cabinet_type="CustomCabinet",
        sub_units=["R1P3a", "R1P3b", "R1P3c", "R1P3d"],
        description="R1列P3机柜，含4个子机柜a/b/c/d，对应4个CMU",
        location="R1-P03",
    )
    print(f"  R1P3: {r.get('ok')} → {r.get('cabinet','?')}")

    # Connect R1P3 to compute_module
    r2 = sysml_add_relation(
        relation_type="connection",
        source="R1P3",
        target="compute_module",
        name=f"R1P3_contains_compute_module",
        parent_package="Cabinets",
        description="R1P3机柜安装计算模块",
        role_source="cabinet",
        role_target="module",
    )
    print(f"  R1P3 → compute_module: {r2.get('ok')}")

    # ── 3. 注入运维命令 ────────────────────────────────────────
    print("\n--- Adding commands ---")

    r = sysml_add_command(
        name="yhst",
        command_text="yhst",
        target_device="CMU",
        description="查看所有结点的加电信息 (文档6.3节)",
        invocation="smu_tranfer_cmd r1.p03a.m yhst",
        source_section="6.3",
        aliases=["加电查询", "yhst命令", "加电", "power_status"],
    )
    print(f"  yhst: {r.get('ok')} → {r.get('qualified_name','?')} connected_to={r.get('connected_to','?')}")

    r = sysml_add_command(
        name="smu_tranfer_cmd",
        command_text="smu_tranfer_cmd",
        target_device="ManagementNode",
        description="通过SMU向指定CMU转发命令 (文档6.3节)",
        invocation="smu_tranfer_cmd r1.p03a.m yhst",
        source_section="6.3",
        aliases=["smu_tranfer", "smu转发", "SMU"],
    )
    print(f"  smu_tranfer_cmd: {r.get('ok')} → {r.get('qualified_name','?')} connected_to={r.get('connected_to','?')}")

    r = sysml_add_command(
        name="ncid",
        command_text="ncid",
        target_device="CMU",
        description="转换节点号为可读名称 (文档6.2节)",
        source_section="6.2",
        aliases=["节点号转换", "node_id"],
    )
    print(f"  ncid: {r.get('ok')} → {r.get('qualified_name','?')}")

    # Connect smu_tranfer_cmd → yhst (forwarding relationship)
    r = sysml_add_relation(
        relation_type="connection",
        source="smu_tranfer_cmd",
        target="yhst",
        name="smu_tranfer_cmd_forwards_to_yhst",
        parent_package="Commands",
        description="mn0通过SMU转发 yhst 命令到CMU执行",
        role_source="invokes",
        role_target="executed",
    )
    print(f"  smu_tranfer_cmd → yhst: {r.get('ok')}")

    # Connect yhst → R1P3 (command targets cabinet)
    r = sysml_add_relation(
        relation_type="connection",
        source="yhst",
        target="R1P3",
        name="yhst_queries_R1P3",
        parent_package="Commands",
        description="yhst命令查询R1P3机柜的加电信息",
        role_source="queries",
        role_target="queried",
    )
    print(f"  yhst → R1P3: {r.get('ok')}")

    # ── 4. 保存 ────────────────────────────────────────────────
    print("\n--- Saving KG ---")
    save_result = _sysml_save_model(str(kg_path))
    print(f"  Saved: {save_result}")

    # Final summary
    summary = sysml_model_summary()
    print(f"\nFinal: {summary['total_entities']} entities, {summary['total_relations']} relations")
    print("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inject known entities into SysML KG")
    parser.add_argument("--db", default=DEFAULT_DB, help=f"DB name (default: {DEFAULT_DB})")
    args = parser.parse_args()
    inject(args.db)
