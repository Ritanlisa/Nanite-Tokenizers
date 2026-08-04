# Known Issues / 遗留问题清单

## 已解决

- **[已修复 2026-08-04] scripts/test_build_pipeline.py 3/4 失败：settings.yaml KG_EXTRACTION_ENABLED=false 与断言期望 True 冲突（产品逻辑漂移）**
  - 根因：settings.yaml 第 5 行 `KG_EXTRACTION_ENABLED: false` 是用户有意配置（git 2274373 工作站同步带入，真实运行时关闭 KG 提取）；config.py 默认值 `True` 正确（yaml 覆盖优先级更高）。三个失败断言硬编码假设运行时必须为 True，属测试自身逻辑 bug。
  - 修复（commit 20572ee，未改产品代码/settings.yaml/config.py）：
    1. `test_config_kg_settings`：改为断言 `Settings.model_fields["KG_EXTRACTION_ENABLED"].default is True`（验证 config.py 产品默认）+ 运行时值为 bool。
    2. `test_kg_build_hook_runs_with_disabled_setting`：恢复后断言 `is original`（原测试恢复后硬编码断言 True，逻辑 bug）。
    3. `test_section_extraction`：mock 文案过短（47/14 字符）被产品 50 字符过滤阈值剔除；截断断言 8000 过期（产品现截断至 4000）。已按产品行为校正 mock 数据与断言。
  - 验证：`scripts/test_build_pipeline.py -m ''` → 4 passed；`tests/` 回归 → 29 passed（7 deselected）不变。
