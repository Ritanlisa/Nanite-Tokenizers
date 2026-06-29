# 全局 AGENTS.md — 任务列表管理规则

## TL;DR
在 `~/.config/opencode/AGENTS.md` 末尾新增"二十四、任务列表与工作连续性"章节，
强制每次工作开始前/结束后检查并同步 plan 文件的 TODO checkbox。

## TODOs

- [ ] 1. 在 `~/.config/opencode/AGENTS.md` 末尾追加新章节

  **What to do**:
  - 读取 `/home/ritanlisa/.config/opencode/AGENTS.md`
  - 在文件末尾 `- 及时报告覆盖率瓶颈：...` 之后追加以下内容：

  ```markdown
  ## 二十四、任务列表与工作连续性

  ### 24.1 TODO 列表是唯一真实状态源

  Boulder continuation 系统依赖 plan 文件中的 `- [ ]` / `- [x]` / `- [~]` 来判断进度。
  **永远不要依赖内存中的"我记得做完了"** ——系统只看 `.omo/plans/*.md` 中的 checkbox 状态。

  ### 24.2 开始工作前：读取 TODO 列表

  每次会话启动（包括自动 continuation），**第一步必须**：
  1. `Read(".omo/plans/{plan-name}.md")` — 读取计划文件
  2. 统计 `- [x]` 和 `- [ ]` 数量 — 验证与上次结束时的预期一致
  3. 对每个 `- [x]` 任务：确认对应的代码修改确实已提交（`git log --oneline` 验证）
  4. 如果发现任务实际已完成但 checkbox 未打勾 → **立即补勾**
  5. 如果发现任务 checkbox 已打勾但实际未完成 → **立即取消勾选并标记 `- [~]`**

  ### 24.3 结束工作后：更新 TODO 列表

  每次会话结束或转向新计划前，**最后一步必须**：
  1. 对每个已完成的任务 → `Edit` 计划文件，将 `- [ ]` 改为 `- [x]`
  2. 对每个阻塞的任务 → `Edit` 计划文件，将 `- [ ]` 改为 `- [~]`（附阻塞原因）
  3. Commit 计划文件的修改
  4. `Read(".omo/plans/{plan-name}.md")` — **二次确认** checkbox 已正确更新

  ### 24.4 自动继续时的一致性检查

  当 Boulder continuation 触发时，**在继续执行任何任务前**：
  1. 读取计划文件，列出所有任务状态
  2. 检查是否有 `- [~]` 任务变成了可恢复状态（阻塞条件已解除）
  3. 检查是否有 `- [ ]` 任务实际在上一会话中已完成但忘记打勾
  4. 如果发现状态不一致 → **优先修复状态**，再继续工作
  5. **绝不能**因为系统提示"继续工作"就盲目执行——先确认 TODO 的准确性

  ### 24.5 反模式：checkbox 与工作分离

  **❌ 禁止**：
  - 完成代码修改后不打勾 checkbox
  - 依赖"我脑子记得"判断进度
  - 依赖 `todowrite` 工具而不同步 plan 文件
  - 跳过 Step 2（结束后确认）直接开始新任务

  **✅ 必须**：
  - 每个任务完成后立即更新 plan 文件的 checkbox
  - 更新后 `Read` 二次确认
  - todoWrite 和 plan checkbox 同步保持
  ```

  **Commit**: YES
  - Message: `docs: add TODO management discipline rules`
  - Files: `/home/ritanlisa/.config/opencode/AGENTS.md`
