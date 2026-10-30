## 使用方法
streamlit run app.py
访问http://localhost:8501

## 开发问题与解决记录

### 1. DeepSeek API 返回 JSON 不完整

**问题现象：**
- 点击 AI 分析后，提示 JSON 解析失败。
- 错误信息：`Expecting property name enclosed in double quotes`。
- DeepSeek API 请求成功，但返回的 JSON 不完整。

**排查过程：**
- 检查文档解析：正常，提取 13,287 字符。
- 检查 API 请求：正常，耗时约 44.56 秒。
- 检查模型响应：`finish_reason=length`，`completion_tokens=8193`。
- 定位原因：模型输出达到 Token 上限，导致 JSON 被截断。

**解决方案：**
- 增加模型响应状态检查，识别 `finish_reason=length`。
- 将一次性生成全部分析结果改为分阶段生成。
- 每个阶段独立进行 JSON 解析和 Pydantic 校验，最后合并结果。
- 增加异常日志，记录模型响应、Token Usage 和错误位置。

**经验总结：**
LLM 返回 HTTP/API 请求成功，不代表生成内容完整。需要同时校验响应结束原因、JSON 格式及业务数据结构。

### 2. 思考模式下四阶段输出稳定性

DeepSeek 默认思考模式下，reasoning_tokens 与 JSON 输出共享
`completion_tokens`。固定 8192 会在推理阶段耗尽预算，导致 JSON 截断。

当前方案：

- 保持模型默认思考模式，不发送 `thinking disabled` 或
  `reasoning_effort=none`。
- 四个阶段使用独立预算：产品概述 10240、业务规则 14336、潜在风险
  14336、需求问题 20480；上限分别为 16384 / 24576 / 24576 / 32768。
- 预算包含 reasoning 预留，并根据文档长度、模块数量动态上调。
- 需求问题阶段在模块数大于 3 时，按模块组（最多 2 组）与
  冲突 / 歧义 / 缺失 / 待确认 4 类问题交叉拆分，并追加 1 个跨模块
  一致性检查子任务，最后在 Python 侧合并去重。
- `finish_reason=length` 一律判定为截断，先拆分任务；无法拆分时才按
  实际 reasoning_tokens 增量提升预算。
- JSON 解析或 Pydantic 校验失败时，先用“缩小单次输出范围”的修复提示
  重试一次；仍失败则拆分任务重试。
- 每个阶段记录耗时、Prompt / Completion / Reasoning Tokens、
  finish_reason、请求与重试次数，并在 Streamlit 进度与调试面板展示。
- 若模型拒绝当前 `max_tokens`，自动减半后重试（最多 3 次），
  兼容不同 DeepSeek 版本的输出上限。

### 3. 业务规则、风险与待确认问题质量优化

`business_rules`、`potential_risks`、`ambiguous_requirements` 保持
`list[str]` 结构不变，通过字段级输出契约和本地后处理提升可读性与可测试性：

- 业务规则：每条必须是原子规则，格式为
  `[模块][规则类型] 触发条件/操作 → 预期结果`，不混写风险或待确认问题。
- 潜在风险：每条必须包含触发场景和具体影响，格式为
  `[模块][风险类型] 触发场景 → 可能影响`，同一根因合并。
- 待确认问题：按 `P0`、`P1`、`P2` 排序，格式为
  `[P0|P1|P2][模块][歧义|冲突|缺失] 待确认点：...；需确认：...`。
- Python 侧统一去除模型偶发的列表序号、项目符号和多余空白，
  对字段内及跨字段的完全重复项做确定性去重。
- Streamlit 报告展示各字段条目数，并将待确认问题按优先级排序。

可调环境变量：

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `DEEPSEEK_MAX_OUTPUT_TOKENS` | 32768 | 全局输出预算上限 |
| `DEEPSEEK_GENERATION_1_MAX_TOKENS` | 10240 | 阶段 1 预算（显式设置后为硬上限） |
| `DEEPSEEK_GENERATION_2_MAX_TOKENS` | 14336 | 阶段 2 预算（同上） |
| `DEEPSEEK_GENERATION_3_MAX_TOKENS` | 14336 | 阶段 3 预算（同上） |
| `DEEPSEEK_GENERATION_4_MAX_TOKENS` | 20480 | 阶段 4 预算（同上） |
| `DEEPSEEK_MAX_VALIDATION_RETRIES` | 1 | JSON / 校验失败修复重试次数 |
| `DEEPSEEK_MAX_TRUNCATION_RECOVERIES` | 2 | 截断拆分恢复层数 |

## 4、AI 需求分析质量优化

### 1. 问题发现

使用真实 PRD 文档进行测试后，发现 AI 需求分析存在以下问题：

- **内容重复**：相同业务规则和需求问题重复出现。
- **推测性过强**：将 PRD 未定义的行为直接推断为系统风险。
- **分类不清晰**：需求冲突、歧义、缺失和待确认问题混合。
- **缺少原文溯源**：分析结果无法快速定位对应的需求依据。
- **缺少质量评估**：无法客观衡量 AI 分析结果的准确性。

### 2. 优化方案

- **内容去重**：对重复业务规则、风险和问题进行语义归并。
- **问题分类**：拆分为需求冲突、需求歧义、需求缺失、待确认问题。
- **原文溯源**：为分析结果增加来源章节、原文片段和分析说明。
- **风险分级**：区分明确风险、潜在风险和通用安全建议，避免将 AI 推测当作实际缺陷。
- **质量评估**：建立需求覆盖率、问题准确率、内容重复率、无依据内容比例、原文溯源准确率等评估指标。

### 3. 优化目标

提高 AI 需求分析的准确性、可追溯性和可验证性，为后续测试点生成、测试用例生成提供可靠的数据基础。

**核心原则：AI 辅助分析，人工验证结论。**

## 5. 第二轮分析质量优化

### evidence_type 统一定义

- `explicit`：当前结论由 PRD 原文直接明确描述。
- `inferred`：当前结论由 AI 根据 PRD 内容推导得出。
- `potential_risks` 一律使用 `inferred`；即使引用明确业务规则，也不把风险结论标记为 `explicit`。
- `source_text` 始终引用真实 PRD 原文，不因证据类型调整而改变。

### 四类问题分类

- `requirement_conflicts`：PRD 中存在相互矛盾的要求。
- `requirement_ambiguities`：已有描述，但存在多种会实质影响实现或测试的合理解释。
- `missing_requirements`：影响实现或测试的必要规则没有定义。
- `pending_questions`：PRD 明确列出的待产品确认事项，保留 Q1-Q4 原始编号。

### 关联与归并

- 需求问题增加稳定 `requirement_id`；原始待确认问题增加 `question_id`。
- 派生问题使用 `related_question_id` 关联 Q 编号，使用 `related_requirement_id` 关联其他需求问题。
- 风险与需求问题指向同一问题时保留两侧信息，通过关联字段建立关系，不机械删除。
- 同一原文、同一模块、同一角色的并列权限操作可以合并；不同校验、权限或状态规则不强行合并。

### 数据结构变化

- 风险条目新增 `test_suggestion`、`expected_behavior`、`needs_product_confirmation`、`related_question_id`、`related_requirement_id`。
- 需求问题新增 `requirement_id`、`question_id`、`related_question_id`、`related_requirement_id`。
- PRD 未定义预期行为时，`expected_behavior` 为空，`needs_product_confirmation=true`。

### 会议室 PRD 复分析结果

结果保存在 `src/meeting-room-product_analysis_round2.json`：业务规则 27 条、潜在风险 6 条、冲突 0 条、歧义 4 条、缺失 5 条、待确认问题 4 条。管理员新增/编辑/停用会议室已合并为一条权限规则，Q1-Q4 保留在待确认问题中。
