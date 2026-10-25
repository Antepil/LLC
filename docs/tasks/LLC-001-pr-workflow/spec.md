# LLC-001：PR 与多 Agent 工作流

Status: ready-for-agent

## 问题与方案

项目负责人需要在单个 Agent 客户端中协调开发、独立审查，并通过 GitHub PR 验收与追溯变更。
使用仓库内 Markdown 记录 Spec/Tickets、固定版本双轴审查、GitHub PR 交付；负责人决定合并。

## 验收标准

- AGENTS.md 明确多 Agent 权责与 PR 合并边界，不再自动合并 master。
- 工作流明确本地 tracker、Skill 路径适配、状态、依赖、验证及产物归档规则。
- REVIEW.md 包含项目风险、证据要求、发现分类、核实及最多三轮后的重新评估机制。
- 提供 Ticket、审查报告与 PR 交付模板；人可从任务索引找到规格和 PR。
- 建立远程 master 基线，独立任务分支发布 PR；保留已有三个未跟踪文件。
- 本次文档验证与独立审查如实记录，不宣称模型效果、远程 CI 或分支保护已验证/启用。

## 决策与验证

沿用现有 master 历史；本次只修改流程文档和模板，不更改全局 Skill。
检查 Markdown 文件链接、git diff --check、范围与双轴独立审查。
本次为用户已讨论的工作流落地，不重新发起 Grill 或扩大到网络实现。

## 非目标

不更改训练/仿真代码，不运行完整训练，不合并 PR，不设置托管平台权限或自动化门禁。
GitHub Actions 与分支保护另行实现；本次不把 runner/许可/依赖尚未验证的配置写成已运行能力。
