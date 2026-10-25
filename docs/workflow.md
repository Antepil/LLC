# 单人负责人、多 Agent 与 PR 工作流

## 权威入口与任务系统配置

远程仓库：https://github.com/Antepil/LLC 。主分支 master；用户负责最终合并决策。
本文件是 grill-me / to-spec / to-tickets / code-review 的项目任务系统说明。无需假定额外 setup Skill 已运行。

- tracker 类型：本地 Markdown；GitHub PR 用于代码审阅和最终交付，不自动另建 GitHub Issues。
- feature-slug 使用任务编号加名称，如 LLC-001-pr-workflow。Spec 存在 docs/tasks/<feature-slug>/spec.md。
- Ticket 一票一文件：docs/tasks/<feature-slug>/tickets/<NN>-<slug>.md，按依赖排序；覆盖 to-tickets 的 .scratch 默认路径。
- 本地标签/状态词汇：needs-clarification、ready-for-agent、in-progress、in-review、blocked、done、cancelled。ready-for-agent 是分诊状态，不证明其依赖已经完成。
- 来源 Spec、任务依赖、完成证据均通过相对链接记录；历史 docs/superpowers 文档可引用，不迁移或重写历史事实。
- [任务索引](tasks/index.md) 是人查看进度的入口；Spec 管需求，Ticket 管执行，PR 管最终交付。design.md 按需建立。
- docs/experiments/ 按需保存小型实验报告；原始日志/权重留在被忽略的 artifacts/ 或指定外部存储。记录位置和校验标识，删除后标注不可复现。

本地 tracker 模式下，to-spec 把规格写入上述位置并标注 ready-for-agent；保留其测试边界确认步骤。
to-tickets 在用户确认任务粒度和依赖后发布本地文件；不自动修改或关闭外部父 issue。
Skill 的长用户故事格式适用于复杂需求；用户明确要求简化时，以用户要求为准。

## 开发顺序与职责

1. 主 Agent 检查分支、暂存区、工作区与远程，不混入已有修改。每项独立变更从 master 建 codex/ 任务分支；文档也如此。
2. 用 grilling 调查事实、澄清决策；按其要求由用户确认共同理解。to-spec 固化目标、契约、验收和非目标；to-tickets 拆成可独立验证的完整行为。
3. 主 Agent 按依赖调度。子 Agent 接收 Spec、Ticket、base SHA、工作目录/分支、文件修改范围、接口约束、验证方法及证据要求。仅分派依赖已完成的任务。
4. 并行任务必须无写冲突。需要隔离时使用独立 worktree；GPU、MATLAB 会话和输出目录需独立分配或串行。主 Agent 负责整合检查，子 Agent 不自行合并 master。
5. 运行与变更匹配的检查，形成待审提交后推送并创建 Draft PR。记录分支创建时的 base SHA；审查使用固定 base/head，不能只审未提交代码。
6. code-review 的 Standards/Spec 两条线独立并行，遵循 [REVIEW.md](../REVIEW.md)。主 Agent 核实、修复、验证并归档 [审查报告](templates/review-report.md)。
7. 验收和必要检查通过、阻塞发现解决、证据适用于最终 head 后，将 PR 更新为可审阅状态。用户决定是否合并；没有明确合并授权不得直接合并或启用自动合并。
8. 合并后读取实际合并提交，同步本地 master（不得丢弃修改或强制重置），核验最终差异和状态，按行为变化补充验证。未合并时任务仍为 in-review。

小任务可由主 Agent 实现；不要为了多 Agent 而拆任务。主 Agent 只对需求、验收门限或重大方案变化请求决策。
AI 的审查不能替代规定的人类决策。PR 创建/更新在已授权交付范围内执行，合并单独由用户决定。

## 状态与阻塞

needs-clarification → ready-for-agent → in-progress → in-review → done。
任一阶段可进入 blocked，注明原因、需要的决策和恢复步骤；取消时使用 cancelled 并记录理由。
主 Agent 维护状态，子 Agent 报告完成不等于整项任务 done。任务分支中报告可写“待合并”；合并事实以 PR 为准。
任务索引在下一次获授权的文档变更中补充已合并状态，禁止为了更新索引直接写 master。

## 验证、风险与交付

- 文档：检查链接、示例和规则一致性。代码：运行相关测试；数值行为变化另需解析对照或仿真。
- 训练链路可运行、模型精度达标、闭环有效是不同验收项。报告命令、运行环境、退出状态、代码版本、配置/数据/模型身份。
- 必要环境不可用时标为未验证，不能假称通过；若它阻塞验收，PR 保持 Draft。无关的完整训练不因文档改动而强行运行。
- 高影响变更（数据划分、尺度/单位、事件积分、闭环反馈、CI 权限）需要独立证据核实和用户重点审阅；普通变更也由用户决定合并。
- 当前没有配置 GitHub Actions 或分支保护；本地证据不等于远程自动门禁。选择 runner、依赖安装与 MATLAB/GPU 能力后再单独配置，不能宣称已经启用。
- PR 描述使用 [交付模板](../.github/pull_request_template.md)，作为最终交付记录，更新为最终实现而非对话流水账。任务索引链接 PR，避免复制维护另一份长 delivery 文档。
- 代码回退通过新分支/新 PR；同时恢复匹配的权重、归一化尺度与运行配置。保留已知可用模型，禁止覆盖唯一副本。
- 交接临时文档继续遵循 AGENTS.md 的 /tmp 约定，不作为长期交付报告。
