from __future__ import annotations

import json
import logging
import os
import re
import time
import traceback
from difflib import SequenceMatcher
from dataclasses import dataclass, field
from typing import Any, Callable, Literal, Mapping, Optional, Type

from dotenv import load_dotenv
from openai import APIConnectionError, APITimeoutError, APIError, OpenAI
from pydantic import BaseModel, Field, ValidationError, model_validator

load_dotenv()

logger = logging.getLogger(__name__)

AnalysisEventCallback = Callable[['AnalysisEvent'], None]

MODEL_OUTPUT_TOKEN_LIMITS = {
    'deepseek-chat': 8192,
    'deepseek-flash': 384 * 1024,
    'deepseek-reasoner': 8192,
}
DEFAULT_MODEL_OUTPUT_TOKEN_LIMIT = 8192
DEFAULT_MAX_OUTPUT_TOKENS = 32768
DEFAULT_MAX_TRUNCATION_RECOVERIES = 2
DEFAULT_MAX_VALIDATION_RETRIES = 1
MODULE_BATCH_SIZE = 3
STAGE_OUTPUT_TOKEN_BUDGETS = {
    'generation_1': 10240,
    'generation_2': 14336,
    'generation_3': 14336,
    'generation_4': 20480,
}
STAGE_OUTPUT_TOKEN_CEILINGS = {
    'generation_1': 16384,
    'generation_2': 24576,
    'generation_3': 24576,
    'generation_4': 32768,
}
# 思考模式会占用 completion_tokens，预算必须包含推理开销。
STAGE_REASONING_TOKEN_RESERVES = {
    'generation_1': 3072,
    'generation_2': 4096,
    'generation_3': 4096,
    'generation_4': 6144,
}
# (基础输出 tokens, 每个模块输出 tokens, 文档字符/token)
STAGE_OUTPUT_ESTIMATES = {
    'generation_1': (500, 150, 60),
    'generation_2': (700, 600, 22),
    'generation_3': (700, 550, 22),
    'generation_4': (900, 650, 20),
}
MAX_MODULE_BATCHES_FOR_AMBIGUITY = 2
MIN_SPLIT_DOCUMENT_CHARS = 1200
MIN_TRUNCATION_BUDGET_STEP = 4096
MIN_OUTPUT_TOKEN_FLOOR = 2048
MAX_OUTPUT_BUDGET_REDUCTIONS = 3
DEFAULT_TIMEOUT_SECONDS = 120.0
DEFAULT_MAX_RETRIES = 2
DEFAULT_RETRY_BACKOFF_SECONDS = 1.0
ANALYSIS_LIST_FIELDS = (
    'business_rules',
    'potential_risks',
    'requirement_conflicts',
    'requirement_ambiguities',
    'missing_requirements',
    'pending_questions',
)
REQUIREMENT_LIST_FIELDS = (
    'requirement_conflicts',
    'requirement_ambiguities',
    'missing_requirements',
    'pending_questions',
)


@dataclass
class AnalysisEvent:
    stage: str
    state: Literal['running', 'success', 'error']
    message: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class AIRequestTrace:
    key: str
    label: str
    status: Literal['pending', 'running', 'success', 'error'] = 'running'
    source_context: list[str] = field(default_factory=list)
    prompt_char_count: int = 0
    system_prompt: str = ''
    user_prompt: str = ''
    attempts: int = 0
    request_elapsed_seconds: Optional[float] = None
    response_id: Optional[str] = None
    response_model: Optional[str] = None
    finish_reason: Optional[str] = None
    usage: dict[str, Any] = field(default_factory=dict)
    output_char_count: Optional[int] = None
    raw_content: Optional[str] = None
    response_metadata: dict[str, Any] = field(default_factory=dict)
    json_error: Optional[str] = None
    validation_error: Optional[str] = None
    validation_field_errors: list[dict[str, Any]] = field(default_factory=list)
    truncated: bool = False
    exception_type: Optional[str] = None
    exception_message: Optional[str] = None
    exception_traceback: Optional[str] = None
    request_thinking_config: dict[str, Any] = field(default_factory=dict)
    request_reasoning_effort: Optional[str] = None
    request_max_tokens: int = 0
    retry_count: int = 0


@dataclass
class AIStageTrace:
    key: str
    label: str
    status: Literal['pending', 'running', 'success', 'error'] = 'pending'
    prompt_char_count: int = 0
    system_prompt: str = ''
    user_prompt: str = ''
    attempts: int = 0
    request_elapsed_seconds: Optional[float] = None
    response_id: Optional[str] = None
    response_model: Optional[str] = None
    finish_reason: Optional[str] = None
    usage: dict[str, Any] = field(default_factory=dict)
    output_char_count: Optional[int] = None
    raw_content: Optional[str] = None
    response_metadata: dict[str, Any] = field(default_factory=dict)
    json_error: Optional[str] = None
    validation_error: Optional[str] = None
    validation_field_errors: list[dict[str, Any]] = field(default_factory=list)
    truncated: bool = False
    exception_type: Optional[str] = None
    exception_message: Optional[str] = None
    exception_traceback: Optional[str] = None
    request_thinking_config: dict[str, Any] = field(default_factory=dict)
    request_reasoning_effort: Optional[str] = None
    request_max_tokens: int = 0
    subtask_count: int = 0
    completed_subtask_count: int = 0
    retry_count: int = 0
    result_count: int = 0
    result_field_counts: dict[str, int] = field(default_factory=dict)
    request_traces: list[AIRequestTrace] = field(default_factory=list)


@dataclass
class AIAnalysisTrace:
    model: str
    model_output_limit: int = 0
    max_output_tokens: int = 0
    document_char_count: int = 0
    prompt_char_count: int = 0
    system_prompt: str = ''
    user_prompt: str = ''
    request_elapsed_seconds: Optional[float] = None
    response_id: Optional[str] = None
    response_model: Optional[str] = None
    finish_reason: Optional[str] = None
    usage: dict[str, Any] = field(default_factory=dict)
    raw_content: Optional[str] = None
    response_metadata: dict[str, Any] = field(default_factory=dict)
    json_error: Optional[str] = None
    validation_error: Optional[str] = None
    validation_field_errors: list[dict[str, Any]] = field(default_factory=list)
    truncated: bool = False
    stage_traces: list[AIStageTrace] = field(default_factory=list)
    exception_type: Optional[str] = None
    exception_message: Optional[str] = None
    exception_traceback: Optional[str] = None


class AIAnalysisError(Exception):
    error_code = 'ai_analysis_error'

    def __init__(
        self,
        message: str,
        *,
        trace: AIAnalysisTrace,
        stage: str,
        details: Optional[dict[str, Any]] = None,
    ) -> None:
        super().__init__(message)
        self.trace = trace
        self.stage = stage
        self.details = details or {}


class AIConfigurationError(AIAnalysisError):
    error_code = 'ai_configuration_error'


class AIRequestError(AIAnalysisError):
    error_code = 'ai_request_error'


class AIRequestTimeoutError(AIRequestError):
    error_code = 'ai_request_timeout'


class AIResponseStructureError(AIAnalysisError):
    error_code = 'ai_response_structure_error'


class AIEmptyResponseError(AIAnalysisError):
    error_code = 'ai_empty_response'


class AIOutputTruncatedError(AIAnalysisError):
    error_code = 'ai_output_truncated'


class AIJSONParseError(AIAnalysisError):
    error_code = 'ai_json_parse_error'


class AIValidationError(AIAnalysisError):
    error_code = 'ai_validation_error'


class AIIntegrityError(AIAnalysisError):
    error_code = 'ai_merge_integrity_error'


class ProductOverviewSection(BaseModel):
    product_summary: str = Field(
        description='产品整体概述，2-4 句话'
    )
    core_features: list[str] = Field(
        description='产品核心功能模块'
    )


EvidenceType = Literal['explicit', 'inferred']
RequirementPriority = Literal['P0', 'P1', 'P2']


class AnalysisItem(BaseModel):
    content: str = Field(
        min_length=1,
        description='具体业务规则、风险或需求问题',
    )
    evidence_type: EvidenceType = Field(
        description='explicit 表示 PRD 原文明确描述；inferred 表示基于原文推断',
    )
    source_section: str = Field(
        default='',
        description='PRD 来源章节；无法定位时为空',
    )
    source_text: str = Field(
        default='',
        description='PRD 原文片段，必须逐字引用；无法定位时为空',
    )
    analysis: str = Field(
        min_length=1,
        description='对该条目的分析说明；inferred 时必须说明推断依据',
    )

    @model_validator(mode='after')
    def validate_evidence_contract(self) -> 'AnalysisItem':
        if self.evidence_type == 'explicit' and not self.source_text.strip():
            raise ValueError(
                'evidence_type=explicit 时必须提供 source_text 原文依据'
            )
        if self.evidence_type == 'inferred':
            analysis = self.analysis.strip().lower()
            inference_markers = (
                '推断依据',
                '推测依据',
                '依据：',
                '基于',
                'infer',
            )
            if not any(marker in analysis for marker in inference_markers):
                raise ValueError(
                    'evidence_type=inferred 时 analysis 必须说明推断依据'
                )
        return self


class PotentialRiskItem(AnalysisItem):
    risk_type: str = Field(
        min_length=1,
        description='风险类型，如并发、权限安全、数据一致性',
    )


class RequirementItem(AnalysisItem):
    priority: RequirementPriority = Field(
        description='P0 阻塞主流程，P1 影响关键规则，P2 体验或非关键边界',
    )


class BusinessRulesSection(BaseModel):
    business_rules: list[AnalysisItem] = Field(
        description=(
            '原子、可验证的业务规则；content 采用 '
            '[模块][规则类型] 触发条件/操作 → 预期结果 格式'
        )
    )


class PotentialRisksSection(BaseModel):
    potential_risks: list[PotentialRiskItem] = Field(
        description=(
            '潜在风险；content 采用 '
            '[模块][风险类型] 触发场景 → 可能影响 格式'
        )
    )


class RequirementsAnalysisSection(BaseModel):
    requirement_conflicts: list[RequirementItem] = Field(
        description='PRD 原文存在相互矛盾描述的需求冲突',
    )
    requirement_ambiguities: list[RequirementItem] = Field(
        description='描述不明确、可能存在多种合理解释的需求歧义',
    )
    missing_requirements: list[RequirementItem] = Field(
        description='实现或测试所需且 PRD 未定义的信息',
    )
    pending_questions: list[RequirementItem] = Field(
        description='需要产品或业务人员进一步确认的问题',
    )


class ProductAnalysis(BaseModel):
    product_summary: str = Field(
        description='产品整体概述'
    )
    core_features: list[str] = Field(
        description='产品核心功能模块'
    )
    business_rules: list[AnalysisItem] = Field(
        description='原子、可验证的产品业务规则'
    )
    potential_risks: list[PotentialRiskItem] = Field(
        description='带触发场景和影响说明的潜在风险'
    )
    requirement_conflicts: list[RequirementItem] = Field(
        description='PRD 中存在明确矛盾描述的需求冲突'
    )
    requirement_ambiguities: list[RequirementItem] = Field(
        description='描述不明确或存在多种理解的需求歧义'
    )
    missing_requirements: list[RequirementItem] = Field(
        description='实现或测试所需但 PRD 未定义的信息'
    )
    pending_questions: list[RequirementItem] = Field(
        description='需要产品或业务人员进一步确认的问题'
    )


ANALYSIS_FIELD_MODELS: dict[str, Type[AnalysisItem]] = {
    'business_rules': AnalysisItem,
    'potential_risks': PotentialRiskItem,
    'requirement_conflicts': RequirementItem,
    'requirement_ambiguities': RequirementItem,
    'missing_requirements': RequirementItem,
    'pending_questions': RequirementItem,
}


@dataclass(frozen=True)
class _StageSpec:
    key: str
    label: str
    schema: Type[BaseModel]
    focus: str


@dataclass(frozen=True)
class _Subtask:
    key: str
    label: str
    focus: str
    modules: tuple[str, ...] = ()
    source_titles: tuple[str, ...] = ()
    document_text: Optional[str] = None


STAGE_SPECS = (
    _StageSpec(
        key='generation_1',
        label='产品概述与核心功能',
        schema=ProductOverviewSection,
        focus=(
            '只生成 product_summary 和 core_features。'
            'product_summary 用 2-4 句话概括产品目标；'
            'core_features 按核心功能模块列出，避免测试步骤细节。'
        ),
    ),
    _StageSpec(
        key='generation_2',
        label='业务规则',
        schema=BusinessRulesSection,
        focus=(
            '只生成 business_rules。提取状态流转、权限、校验、'
            '审批、通知、数据约束等可验证业务规则，禁止编造。'
        ),
    ),
    _StageSpec(
        key='generation_3',
        label='潜在风险',
        schema=PotentialRisksSection,
        focus=(
            '只生成 potential_risks。聚焦边界条件、异常场景、'
            '并发、安全、数据一致性和流程失败风险。'
        ),
    ),
    _StageSpec(
        key='generation_4',
        label='需求冲突、歧义、缺失与待确认问题',
        schema=RequirementsAnalysisSection,
        focus=(
            '只生成 requirement_conflicts、requirement_ambiguities、'
            'missing_requirements、pending_questions 四个列表。'
            '每个问题只能进入最合适的一个列表，绝不能重复归类。'
        ),
    ),
)


_STAGE_OUTPUT_CONTRACTS = {
    'generation_2': (
        '业务规则输出契约：\n'
        '- 每条只描述一个原子、可验证的业务规则；不同条件、角色或状态不得混写。\n'
        '- content 统一格式：[模块][规则类型] 触发条件/操作 → 预期结果。\n'
        '- 规则类型仅使用：状态流转、权限、条件约束、校验、数据约束、通知。\n'
        '- 优先提取文档明确表达的规则并标记 explicit；'
        '只有业务逻辑可由原文直接推出时才标记 inferred，'
        '且 analysis 必须写明“推断依据”。\n'
        '- 同一规则在多个章节重复出现时只输出一条，保留信息更完整、'
        '来源更直接的一条；不同触发条件、角色、状态或结果的规则不得合并。\n'
        '- 原文用并列动词列出多个独立操作时（如“新增、编辑和停用”），'
        '每个操作必须单独输出，不得合并成一条，也不得遗漏其中任一操作。\n'
        '- 原文没有依据的内容不要输出为业务规则。\n'
        '- source_text 必须逐字摘录 PRD；不得改写、概括或拼接出原文不存在的句子。\n'
        '- 不输出风险、待确认问题、实现建议或测试步骤。'
    ),
    'generation_3': (
        '潜在风险输出契约：\n'
        '- 每条只描述一个独立风险，必须由文档中的明确行为、状态、角色或约束触发。\n'
        '- 每个模块最多保留 2 条影响最大的风险；优先级相同的情况下，'
        '选择最可能破坏主流程、数据一致性或权限边界的风险。\n'
        '- 不要假设系统“没有事务锁”“没有服务端校验”等实现缺陷；'
        '除非 PRD 明确说明该行为，否则不得把通用技术缺陷当作本项目风险。\n'
        '- content 统一格式：[模块][风险类型] 触发场景 → 可能影响。\n'
        '- risk_type 仅使用：边界条件、异常流程、并发、权限安全、'
        '数据一致性、性能可用性、集成兼容。\n'
        '- 影响必须具体到错误状态、数据损坏、越权或泄露、流程中断或统计失真。\n'
        '- 同一根因或同一触发场景只输出一条，合并语义相同但措辞不同的风险，'
        '保留信息更完整的一条；不同根因的风险不得合并。\n'
        '- 能从 PRD 行为直接验证的标 explicit；基于业务逻辑推断的标 inferred，'
        '并在 analysis 中写明“推断依据”，且依据必须来自具体 PRD 行为、'
        '状态或角色限制，不得只写“通常系统会……”；不得把推断包装成确定事实。\n'
        '- source_text 必须逐字摘录触发该风险的 PRD 原文；找不到依据时置空，'
        '并在 analysis 中说明原因，禁止伪造引用。\n'
        '- 禁止只写“存在风险”、泛化安全口号、假设性实现缺陷或一般性改进建议。\n'
        '- 不输出待确认问题、实现建议或测试用例。'
    ),
    'generation_4': (
        '需求问题输出契约：\n'
        '- JSON 必须包含且仅包含四个列表：requirement_conflicts、'
        'requirement_ambiguities、missing_requirements、pending_questions。\n'
        '- 每个问题只能进入一个列表，按最根本的问题性质分类；'
        '同一根因不得在不同列表或同一列表中重复出现。\n'
        '- requirement_conflicts：通常应为空；只有 PRD 中存在两条可逐字引用的描述，'
        '且二者语义直接矛盾时才使用；source_text 用“｜”连接两段原文。'
        '缺少信息、表述含糊或仅凭推测可能冲突的，不得判为冲突。'
        '“一般性授权 + 更具体的限制/例外”是分层规则，不是冲突，'
        '除非两条描述对同一条件的结论无法同时成立。\n'
        '- requirement_ambiguities：同一描述存在两种以上会实质影响实现、'
        '验收或测试的合理解释时使用；不要与冲突或缺失重复。\n'
        '- missing_requirements：实现或测试确实需要、但 PRD 完全未定义的信息。'
        '只有当缺失会影响功能实现、验收标准或测试设计时才输出；'
        'PRD 未提及但不影响当前功能的通用最佳实践不得输出。\n'
        '- pending_questions：PRD 已明确提出的待确认问题，或必须由产品/业务'
        '做取舍而无法归入前三类的问题。已经能归入歧义或缺失的问题不要重复放入。\n'
        '- 优先级：P0 表示阻塞主流程或无法开发/测试；P1 表示影响关键规则、'
        '数据或权限；P2 表示体验或非关键边界。\n'
        '- content 统一格式：[模块] 具体问题；如需决策，再写“需确认：...”。\n'
        '- 每条都必须填写 evidence_type、source_section、source_text、analysis、'
        'priority；source_text 必须逐字摘录 PRD，找不到依据时置空并说明原因。\n'
        '- 不输出风险分析、实现建议、测试用例或一般性建议。\n'
        '- 同一模块的每个列表最多保留 2 条影响最大的问题；P2 仅保留'
        '确实影响验收或测试判定的事项，避免穷举实现细节。\n'
        '- 同一模块按 P0、P1、P2 排序，只保留影响最大的问题，避免穷举。'
    ),
}


class AIAnalyzer:

    def __init__(
        self,
        client: Optional[OpenAI] = None,
        model: Optional[str] = None,
        timeout_seconds: Optional[float] = None,
        max_output_tokens: Optional[int] = None,
        max_retries: Optional[int] = None,
        retry_backoff_seconds: Optional[float] = None,
        max_truncation_recoveries: Optional[int] = None,
        max_validation_retries: Optional[int] = None,
    ) -> None:
        self._client = client
        self.model = model or os.getenv('DEEPSEEK_MODEL', 'deepseek-flash')
        self.model_output_limit = self._get_model_output_limit(self.model)

        requested_output_tokens = (
            max_output_tokens
            if max_output_tokens is not None
            else self._read_int_env(
                'DEEPSEEK_MAX_OUTPUT_TOKENS',
                DEFAULT_MAX_OUTPUT_TOKENS,
                minimum=1,
            )
        )
        if requested_output_tokens > self.model_output_limit:
            logger.warning(
                'Requested max_output_tokens=%d exceeds model limit=%d '
                'for model=%s; using the model limit',
                requested_output_tokens,
                self.model_output_limit,
                self.model,
            )
        self.max_output_tokens = max(
            1,
            min(requested_output_tokens, self.model_output_limit),
        )
        self.timeout_seconds = (
            timeout_seconds
            if timeout_seconds is not None
            else self._read_float_env(
                'DEEPSEEK_TIMEOUT_SECONDS',
                DEFAULT_TIMEOUT_SECONDS,
                minimum=1.0,
            )
        )
        self.max_retries = (
            max_retries
            if max_retries is not None
            else self._read_int_env(
                'DEEPSEEK_MAX_RETRIES',
                DEFAULT_MAX_RETRIES,
                minimum=0,
            )
        )
        self.retry_backoff_seconds = (
            retry_backoff_seconds
            if retry_backoff_seconds is not None
            else self._read_float_env(
                'DEEPSEEK_RETRY_BACKOFF_SECONDS',
                DEFAULT_RETRY_BACKOFF_SECONDS,
                minimum=0.0,
            )
        )
        self.max_truncation_recoveries = (
            max_truncation_recoveries
            if max_truncation_recoveries is not None
            else self._read_int_env(
                'DEEPSEEK_MAX_TRUNCATION_RECOVERIES',
                DEFAULT_MAX_TRUNCATION_RECOVERIES,
                minimum=0,
            )
        )
        self.max_validation_retries = (
            max_validation_retries
            if max_validation_retries is not None
            else self._read_int_env(
                'DEEPSEEK_MAX_VALIDATION_RETRIES',
                DEFAULT_MAX_VALIDATION_RETRIES,
                minimum=0,
            )
        )
        self.stage_output_budgets: dict[str, int] = {}
        self.stage_budget_overrides: set[str] = set()
        for spec in STAGE_SPECS:
            default_budget = STAGE_OUTPUT_TOKEN_BUDGETS[spec.key]
            env_name = f'DEEPSEEK_{spec.key.upper()}_MAX_TOKENS'
            if os.getenv(env_name) is not None:
                self.stage_budget_overrides.add(spec.key)
            requested_stage_budget = self._read_int_env(
                env_name,
                default_budget,
                minimum=1,
            )
            stage_ceiling = min(
                self.max_output_tokens,
                STAGE_OUTPUT_TOKEN_CEILINGS[spec.key],
            )
            self.stage_output_budgets[spec.key] = min(
                requested_stage_budget,
                stage_ceiling,
            )
        self.last_trace = AIAnalysisTrace(
            model=self.model,
            model_output_limit=self.model_output_limit,
            max_output_tokens=self.max_output_tokens,
        )
        logger.info(
            'AIAnalyzer configured: model=%s output_limit=%d '
            'max_output_tokens=%d timeout=%.1fs max_retries=%d '
            'max_truncation_recoveries=%d max_validation_retries=%d '
            'stage_budgets=%s',
            self.model,
            self.model_output_limit,
            self.max_output_tokens,
            self.timeout_seconds,
            self.max_retries,
            self.max_truncation_recoveries,
            self.max_validation_retries,
            json.dumps(self.stage_output_budgets, sort_keys=True),
        )

    @staticmethod
    def _get_model_output_limit(model: str) -> int:
        normalized_model = model.strip().lower()
        output_limit = MODEL_OUTPUT_TOKEN_LIMITS.get(normalized_model)
        if output_limit is None:
            logger.warning(
                'Unknown model=%s; using conservative output limit=%d',
                model,
                DEFAULT_MODEL_OUTPUT_TOKEN_LIMIT,
            )
            return DEFAULT_MODEL_OUTPUT_TOKEN_LIMIT
        return output_limit

    @staticmethod
    def _read_int_env(
        name: str,
        default: int,
        minimum: int = 0,
    ) -> int:
        raw_value = os.getenv(name)
        if raw_value is None:
            return default
        try:
            value = int(raw_value)
        except (TypeError, ValueError):
            logger.warning(
                'Invalid %s=%r; using %d',
                name,
                raw_value,
                default,
            )
            return default
        if value < minimum:
            logger.warning(
                'Invalid %s=%r; using %d',
                name,
                raw_value,
                default,
            )
            return default
        return value

    @staticmethod
    def _read_float_env(
        name: str,
        default: float,
        minimum: float = 0.0,
    ) -> float:
        raw_value = os.getenv(name)
        if raw_value is None:
            return default
        try:
            value = float(raw_value)
        except (TypeError, ValueError):
            logger.warning(
                'Invalid %s=%r; using %.1f',
                name,
                raw_value,
                default,
            )
            return default
        if value < minimum:
            logger.warning(
                'Invalid %s=%r; using %.1f',
                name,
                raw_value,
                default,
            )
            return default
        return value

    def _resolve_client(self) -> OpenAI:
        if self._client is not None:
            return self._client

        api_key = os.getenv('DEEPSEEK_API_KEY')
        if not api_key:
            raise ValueError('未配置 DEEPSEEK_API_KEY')

        self._client = OpenAI(
            api_key=api_key,
            base_url=os.getenv(
                'DEEPSEEK_BASE_URL',
                'https://api.deepseek.com',
            ),
            timeout=self.timeout_seconds,
            max_retries=0,
        )
        return self._client

    def analyze(
        self,
        document_content: str,
        on_event: Optional[AnalysisEventCallback] = None,
    ) -> ProductAnalysis:
        if not isinstance(document_content, str):
            raise TypeError('document_content 必须是字符串')

        trace = AIAnalysisTrace(
            model=self.model,
            model_output_limit=self.model_output_limit,
            max_output_tokens=self.max_output_tokens,
            document_char_count=len(document_content),
        )
        self.last_trace = trace

        stage_runs = []
        for spec in STAGE_SPECS:
            stage_trace = AIStageTrace(
                key=spec.key,
                label=spec.label,
            )
            system_prompt = _build_system_prompt(spec)
            user_prompt = _build_user_prompt(document_content)
            stage_trace.system_prompt = system_prompt
            stage_trace.user_prompt = user_prompt
            stage_trace.prompt_char_count = (
                len(system_prompt) + len(user_prompt)
            )
            trace.prompt_char_count += stage_trace.prompt_char_count
            trace.system_prompt = system_prompt
            trace.user_prompt = user_prompt
            trace.stage_traces.append(stage_trace)
            stage_runs.append(
                (spec, stage_trace, system_prompt, user_prompt)
            )

        logger.info(
            'AI staged prompts built: model=%s document_chars=%d '
            'stages=%d total_prompt_chars=%d max_output_tokens=%d',
            self.model,
            trace.document_char_count,
            len(STAGE_SPECS),
            trace.prompt_char_count,
            trace.max_output_tokens,
        )
        self._emit(
            on_event,
            AnalysisEvent(
                stage='prompt',
                state='success',
                message=(
                    f'{len(STAGE_SPECS)} 个阶段 Prompt 已构建，'
                    f'合计 {trace.prompt_char_count:,} 字符'
                ),
                metadata={
                    'document_char_count': trace.document_char_count,
                    'prompt_char_count': trace.prompt_char_count,
                    'stage_count': len(STAGE_SPECS),
                },
            ),
        )

        try:
            client = self._resolve_client()
        except ValueError as exc:
            self._record_exception(trace, exc)
            self._emit(
                on_event,
                AnalysisEvent(
                    stage='request',
                    state='error',
                    message=f'配置错误：{exc}',
                ),
            )
            raise AIConfigurationError(
                'AI 配置不完整，请检查 DEEPSEEK_API_KEY',
                trace=trace,
                stage='request',
            ) from exc

        stage_payloads: dict[str, dict[str, Any]] = {}
        for spec, stage_trace, system_prompt, user_prompt in stage_runs:
            stage_trace.status = 'running'
            self._emit(
                on_event,
                AnalysisEvent(
                    stage=spec.key,
                    state='running',
                    message=(
                        f'请求模型 {self.model}，'
                        f'Prompt {stage_trace.prompt_char_count:,} 字符'
                    ),
                    metadata={
                        'model': self.model,
                        'prompt_char_count': (
                            stage_trace.prompt_char_count
                        ),
                        'max_output_tokens': (
                            self.stage_output_budgets.get(
                                spec.key,
                                self.max_output_tokens,
                            )
                        ),
                    },
                ),
            )

            try:
                payload = self._run_stage(
                    client=client,
                    spec=spec,
                    stage_trace=stage_trace,
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    document_content=document_content,
                    completed_stage_payloads=stage_payloads,
                    trace=trace,
                    on_event=on_event,
                )
            except AIAnalysisError as exc:
                stage_trace.status = 'error'
                if not stage_trace.exception_type:
                    self._record_stage_exception(stage_trace, exc)
                    self._record_exception(trace, exc)
                self._emit(
                    on_event,
                    AnalysisEvent(
                        stage=spec.key,
                        state='error',
                        message=_stage_error_message(stage_trace, exc),
                        metadata={
                            'error_code': exc.error_code,
                            'stage_label': spec.label,
                            'elapsed_seconds': (
                                stage_trace.request_elapsed_seconds
                            ),
                            'usage': stage_trace.usage,
                            'finish_reason': stage_trace.finish_reason,
                            'truncated': stage_trace.truncated,
                        },
                    ),
                )
                raise
            except Exception as exc:
                stage_trace.status = 'error'
                self._record_stage_exception(stage_trace, exc)
                self._record_exception(trace, exc)
                self._emit(
                    on_event,
                    AnalysisEvent(
                        stage=spec.key,
                        state='error',
                        message=(
                            f'阶段「{spec.label}」发生未预期异常：'
                            f'{type(exc).__name__}: '
                            f'{_sanitize_text(str(exc))}'
                        ),
                    ),
                )
                raise AIRequestError(
                    f'阶段「{spec.label}」发生未预期异常：'
                    f'{type(exc).__name__}: '
                    f'{_sanitize_text(str(exc))}',
                    trace=trace,
                    stage=spec.key,
                    details={'exception_type': type(exc).__name__},
                ) from exc

            stage_trace.status = 'success'
            stage_payloads[spec.key] = payload.model_dump()
            self._emit(
                on_event,
                AnalysisEvent(
                    stage=spec.key,
                    state='success',
                    message=_stage_success_message(stage_trace),
                    metadata={
                        'stage_label': spec.label,
                        'elapsed_seconds': (
                            stage_trace.request_elapsed_seconds
                        ),
                        'usage': stage_trace.usage,
                        'response_id': stage_trace.response_id,
                        'finish_reason': stage_trace.finish_reason,
                        'retry_count': stage_trace.retry_count,
                        'subtask_count': stage_trace.subtask_count,
                        'completed_subtask_count': (
                            stage_trace.completed_subtask_count
                        ),
                    },
                ),
            )

        try:
            analysis = self._merge_stage_results(
                stage_payloads,
                trace,
                document_content,
            )
        except AIAnalysisError as exc:
            self._record_exception(trace, exc)
            self._emit(
                on_event,
                AnalysisEvent(
                    stage='final',
                    state='error',
                    message=f'最终合并校验失败：{exc}',
                    metadata={'error_code': exc.error_code},
                ),
            )
            raise

        self._emit(
            on_event,
            AnalysisEvent(
                stage='final',
                state='success',
                message=(
                    f'{len(STAGE_SPECS)} 个阶段已合并并通过最终校验，'
                    f'总耗时 {trace.request_elapsed_seconds or 0:.2f} 秒'
                ),
                metadata={
                    'total_elapsed_seconds': (
                        trace.request_elapsed_seconds
                    ),
                    'usage': trace.usage,
                    'stage_count': len(STAGE_SPECS),
                },
            ),
        )
        logger.info(
            'Staged AI analysis completed: model=%s stages=%d '
            'elapsed=%.2fs usage=%s',
            self.model,
            len(STAGE_SPECS),
            trace.request_elapsed_seconds or 0.0,
            json.dumps(trace.usage, ensure_ascii=False, default=str),
        )
        return analysis

    def _record_exception(
        self,
        trace: AIAnalysisTrace,
        exc: Exception,
    ) -> None:
        trace.exception_type = type(exc).__name__
        trace.exception_message = _sanitize_text(str(exc))
        trace.exception_traceback = _sanitize_text(traceback.format_exc())

    def _record_stage_exception(
        self,
        stage_trace: AIStageTrace,
        exc: Exception,
    ) -> None:
        stage_trace.exception_type = type(exc).__name__
        stage_trace.exception_message = _sanitize_text(str(exc))
        stage_trace.exception_traceback = _sanitize_text(
            traceback.format_exc()
        )

    @staticmethod
    def _emit(
        callback: Optional[AnalysisEventCallback],
        event: AnalysisEvent,
    ) -> None:
        if callback is None:
            return
        try:
            callback(event)
        except Exception:
            logger.exception('Failed to emit analysis progress event')


    def _run_stage(
        self,
        *,
        client: OpenAI,
        spec: _StageSpec,
        stage_trace: AIStageTrace,
        system_prompt: str,
        user_prompt: str,
        document_content: str,
        completed_stage_payloads: Mapping[str, Mapping[str, Any]],
        trace: AIAnalysisTrace,
        on_event: Optional[AnalysisEventCallback],
    ) -> BaseModel:
        try:
            return self._run_stage_subtasks(
                client=client,
                spec=spec,
                stage_trace=stage_trace,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                document_content=document_content,
                completed_stage_payloads=completed_stage_payloads,
                trace=trace,
                on_event=on_event,
            )
        except AIAnalysisError:
            raise
        except Exception as exc:
            self._record_stage_exception(stage_trace, exc)
            self._record_exception(trace, exc)
            raise AIRequestError(
                f'阶段「{spec.label}」处理失败：'
                f'{type(exc).__name__}: {_sanitize_text(str(exc))}',
                trace=trace,
                stage=spec.key,
                details={'exception_type': type(exc).__name__},
            ) from exc

    def _run_stage_subtasks(
        self,
        *,
        client: OpenAI,
        spec: _StageSpec,
        stage_trace: AIStageTrace,
        system_prompt: str,
        user_prompt: str,
        document_content: str,
        completed_stage_payloads: Mapping[str, Mapping[str, Any]],
        trace: AIAnalysisTrace,
        on_event: Optional[AnalysisEventCallback],
    ) -> BaseModel:
        subtasks = _build_subtasks(
            spec=spec,
            completed_stage_payloads=completed_stage_payloads,
            document_content=document_content,
        )
        stage_trace.subtask_count = len(subtasks)
        stage_trace.completed_subtask_count = 0
        payloads: list[dict[str, Any]] = []

        for index, subtask in enumerate(subtasks, start=1):
            budget = self._request_token_budget(spec, subtask)
            self._emit(
                on_event,
                AnalysisEvent(
                    stage=spec.key,
                    state='running',
                    message=(
                        f'阶段「{spec.label}」子任务 '
                        f'{index}/{len(subtasks)}：{subtask.label}'
                    ),
                    metadata={
                        'stage_label': spec.label,
                        'subtask_key': subtask.key,
                        'subtask_label': subtask.label,
                        'subtask_index': index,
                        'subtask_count': len(subtasks),
                        'max_output_tokens': budget,
                        'thinking_mode': 'model_default',
                    },
                ),
            )
            subtask_payloads = self._run_subtask_with_recovery(
                client=client,
                spec=spec,
                stage_trace=stage_trace,
                subtask=subtask,
                base_system_prompt=system_prompt,
                base_user_prompt=user_prompt,
                trace=trace,
                on_event=on_event,
                recovery_depth=0,
                max_tokens=budget,
            )
            payloads.extend(
                payload.model_dump() for payload in subtask_payloads
            )
            stage_trace.completed_subtask_count += 1

        merged = self._merge_subtask_payloads(spec, payloads)
        stage_trace.result_field_counts = {
            field_name: (
                len(value) if isinstance(value, list)
                else 1 if value else 0
            )
            for field_name, value in merged.items()
        }
        stage_trace.result_count = sum(
            stage_trace.result_field_counts.values()
        )
        validated = spec.schema.model_validate(merged)
        logger.info(
            'Stage completed: stage=%s subtasks=%d/%d requests=%d '
            'retries=%d elapsed=%.2fs prompt_tokens=%s '
            'completion_tokens=%s reasoning_tokens=%s '
            'finish_reason=%s results=%s usage=%s',
            spec.key,
            stage_trace.completed_subtask_count,
            len(subtasks),
            len(stage_trace.request_traces),
            stage_trace.retry_count,
            stage_trace.request_elapsed_seconds or 0.0,
            stage_trace.usage.get('prompt_tokens'),
            stage_trace.usage.get('completion_tokens'),
            _usage_reasoning_tokens(stage_trace.usage),
            stage_trace.finish_reason,
            json.dumps(
                stage_trace.result_field_counts,
                ensure_ascii=False,
                sort_keys=True,
            ),
            json.dumps(stage_trace.usage, ensure_ascii=False, default=str),
        )
        return validated

    def _run_subtask_with_recovery(
        self,
        *,
        client: OpenAI,
        spec: _StageSpec,
        stage_trace: AIStageTrace,
        subtask: _Subtask,
        base_system_prompt: str,
        base_user_prompt: str,
        trace: AIAnalysisTrace,
        on_event: Optional[AnalysisEventCallback],
        recovery_depth: int,
        max_tokens: int,
        repair_note: str = '',
        validation_retries: int = 0,
    ) -> list[BaseModel]:
        current_subtask = subtask
        current_budget = max_tokens
        current_repair_note = repair_note

        while True:
            try:
                payload = self._run_subtask_once(
                    client=client,
                    spec=spec,
                    stage_trace=stage_trace,
                    subtask=current_subtask,
                    base_system_prompt=base_system_prompt,
                    base_user_prompt=base_user_prompt,
                    max_tokens=current_budget,
                    trace=trace,
                    repair_note=current_repair_note,
                )
                return [payload]
            except AIOutputTruncatedError:
                if recovery_depth >= self.max_truncation_recoveries:
                    raise

                split_subtasks = self._split_subtask(current_subtask)
                if split_subtasks:
                    stage_trace.retry_count += 1
                    self._emit(
                        on_event,
                        AnalysisEvent(
                            stage=spec.key,
                            state='running',
                            message=(
                                f'子任务「{current_subtask.label}」输出截断，'
                                f'缩小任务范围后拆分为 '
                                f'{len(split_subtasks)} 个更小任务重试'
                            ),
                            metadata={
                                'subtask_key': current_subtask.key,
                                'subtask_label': current_subtask.label,
                                'truncated': True,
                                'split_count': len(split_subtasks),
                                'recovery_depth': recovery_depth + 1,
                            },
                        ),
                    )
                    return self._run_split_subtasks(
                        client=client,
                        spec=spec,
                        stage_trace=stage_trace,
                        split_subtasks=split_subtasks,
                        base_system_prompt=base_system_prompt,
                        base_user_prompt=base_user_prompt,
                        trace=trace,
                        on_event=on_event,
                        recovery_depth=recovery_depth + 1,
                    )

                next_budget = self._grown_token_budget(
                    spec,
                    stage_trace,
                    current_budget,
                )
                if next_budget <= current_budget:
                    raise
                logger.warning(
                    'Subtask cannot be split; retrying with larger budget: '
                    'stage=%s subtask=%s previous=%d next=%d',
                    spec.key,
                    current_subtask.key,
                    current_budget,
                    next_budget,
                )
                stage_trace.retry_count += 1
                self._emit(
                    on_event,
                    AnalysisEvent(
                        stage=spec.key,
                        state='running',
                        message=(
                            f'子任务「{current_subtask.label}」无法继续拆分，'
                            f'Token 预算 {current_budget} -> {next_budget}'
                        ),
                        metadata={
                            'subtask_key': current_subtask.key,
                            'previous_max_tokens': current_budget,
                            'next_max_tokens': next_budget,
                            'recovery_depth': recovery_depth + 1,
                        },
                    ),
                )
                current_budget = next_budget
                current_repair_note = ''
                recovery_depth += 1
            except (
                AIJSONParseError,
                AIValidationError,
                AIEmptyResponseError,
            ) as exc:
                if validation_retries < self.max_validation_retries:
                    validation_retries += 1
                    stage_trace.retry_count += 1
                    current_repair_note = _build_repair_note(spec, exc)
                    logger.warning(
                        'Subtask output rejected; retrying with narrower '
                        'output scope: stage=%s subtask=%s reason=%s '
                        'validation_retry=%d/%d',
                        spec.key,
                        current_subtask.key,
                        type(exc).__name__,
                        validation_retries,
                        self.max_validation_retries,
                    )
                    self._emit(
                        on_event,
                        AnalysisEvent(
                            stage=spec.key,
                            state='running',
                            message=(
                                f'子任务「{current_subtask.label}」'
                                f'未通过 JSON/字段校验，'
                                f'缩小单次输出范围后重试 '
                                f'({validation_retries}/'
                                f'{self.max_validation_retries})'
                            ),
                            metadata={
                                'subtask_key': current_subtask.key,
                                'subtask_label': current_subtask.label,
                                'error_code': exc.error_code,
                                'validation_retry': validation_retries,
                            },
                        ),
                    )
                    continue

                if recovery_depth >= self.max_truncation_recoveries:
                    raise
                split_subtasks = self._split_subtask(current_subtask)
                if not split_subtasks:
                    raise
                stage_trace.retry_count += 1
                self._emit(
                    on_event,
                    AnalysisEvent(
                        stage=spec.key,
                        state='running',
                        message=(
                            f'子任务「{current_subtask.label}」多次校验失败，'
                            f'拆分为 {len(split_subtasks)} 个更小任务重试'
                        ),
                        metadata={
                            'subtask_key': current_subtask.key,
                            'subtask_label': current_subtask.label,
                            'error_code': exc.error_code,
                            'split_count': len(split_subtasks),
                            'recovery_depth': recovery_depth + 1,
                        },
                    ),
                )
                return self._run_split_subtasks(
                    client=client,
                    spec=spec,
                    stage_trace=stage_trace,
                    split_subtasks=split_subtasks,
                    base_system_prompt=base_system_prompt,
                    base_user_prompt=base_user_prompt,
                    trace=trace,
                    on_event=on_event,
                    recovery_depth=recovery_depth + 1,
                )

    def _run_split_subtasks(
        self,
        *,
        client: OpenAI,
        spec: _StageSpec,
        stage_trace: AIStageTrace,
        split_subtasks: list[_Subtask],
        base_system_prompt: str,
        base_user_prompt: str,
        trace: AIAnalysisTrace,
        on_event: Optional[AnalysisEventCallback],
        recovery_depth: int,
    ) -> list[BaseModel]:
        recovered: list[BaseModel] = []
        for split_subtask in split_subtasks:
            recovered.extend(
                self._run_subtask_with_recovery(
                    client=client,
                    spec=spec,
                    stage_trace=stage_trace,
                    subtask=split_subtask,
                    base_system_prompt=base_system_prompt,
                    base_user_prompt=base_user_prompt,
                    trace=trace,
                    on_event=on_event,
                    recovery_depth=recovery_depth,
                    max_tokens=self._request_token_budget(
                        spec,
                        split_subtask,
                    ),
                )
            )
        return recovered

    def _grown_token_budget(
        self,
        spec: _StageSpec,
        stage_trace: AIStageTrace,
        current_budget: int,
    ) -> int:
        ceiling = min(
            self.max_output_tokens,
            STAGE_OUTPUT_TOKEN_CEILINGS[spec.key],
        )
        reasoning_tokens = 0
        if stage_trace.request_traces:
            value = _usage_reasoning_tokens(
                stage_trace.request_traces[-1].usage
            )
            if isinstance(value, int) and not isinstance(value, bool):
                reasoning_tokens = value
        step = max(
            MIN_TRUNCATION_BUDGET_STEP,
            reasoning_tokens + 1024,
        )
        return min(ceiling, current_budget + step)

    def _run_subtask_once(
        self,
        *,
        client: OpenAI,
        spec: _StageSpec,
        stage_trace: AIStageTrace,
        subtask: _Subtask,
        base_system_prompt: str,
        base_user_prompt: str,
        max_tokens: int,
        trace: AIAnalysisTrace,
        repair_note: str = '',
    ) -> BaseModel:
        system_prompt, user_prompt = self._build_subtask_prompts(
            spec=spec,
            subtask=subtask,
            base_system_prompt=base_system_prompt,
            base_user_prompt=base_user_prompt,
            repair_note=repair_note,
        )
        source_context = list(
            dict.fromkeys(
                list(subtask.modules) + list(subtask.source_titles)
            )
        ) or ['完整文档']
        request_trace = AIRequestTrace(
            key=subtask.key,
            label=subtask.label,
            source_context=source_context,
            prompt_char_count=len(system_prompt) + len(user_prompt),
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        )
        stage_trace.request_traces.append(request_trace)

        try:
            response = self._request_with_retries(
                client=client,
                spec=spec,
                request_trace=request_trace,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                max_tokens=max_tokens,
                trace=trace,
                on_event=None,
            )
            payload = self._parse_and_validate(
                spec=spec,
                stage_trace=request_trace,
                response=response,
                trace=trace,
            )
            request_trace.status = 'success'
            return payload
        except Exception as exc:
            request_trace.status = 'error'
            if not request_trace.exception_type:
                self._record_stage_exception(request_trace, exc)
            raise
        finally:
            self._aggregate_stage_request(stage_trace, request_trace)

    @staticmethod
    def _aggregate_stage_request(
        stage_trace: AIStageTrace,
        request_trace: AIRequestTrace,
    ) -> None:
        stage_trace.attempts += request_trace.attempts
        stage_trace.retry_count += max(
            0,
            request_trace.attempts - 1,
        )
        stage_trace.request_elapsed_seconds = (
            (stage_trace.request_elapsed_seconds or 0.0)
            + (request_trace.request_elapsed_seconds or 0.0)
        )
        stage_trace.usage = _merge_usage(
            stage_trace.usage,
            request_trace.usage,
        )
        stage_trace.prompt_char_count += request_trace.prompt_char_count
        stage_trace.output_char_count = (
            (stage_trace.output_char_count or 0)
            + (request_trace.output_char_count or 0)
        )
        stage_trace.response_id = request_trace.response_id
        stage_trace.response_model = request_trace.response_model
        stage_trace.finish_reason = request_trace.finish_reason
        stage_trace.raw_content = request_trace.raw_content
        stage_trace.response_metadata = request_trace.response_metadata
        stage_trace.json_error = request_trace.json_error
        stage_trace.validation_error = request_trace.validation_error
        stage_trace.validation_field_errors = (
            request_trace.validation_field_errors
        )
        stage_trace.truncated = (
            stage_trace.truncated or request_trace.truncated
        )
        stage_trace.system_prompt = request_trace.system_prompt
        stage_trace.user_prompt = request_trace.user_prompt
        stage_trace.request_thinking_config = (
            request_trace.request_thinking_config
        )
        stage_trace.request_reasoning_effort = (
            request_trace.request_reasoning_effort
        )
        stage_trace.request_max_tokens = request_trace.request_max_tokens

    def _request_token_budget(
        self,
        spec: _StageSpec,
        subtask: _Subtask,
    ) -> int:
        floor_budget = self.stage_output_budgets[spec.key]
        ceiling = min(
            self.max_output_tokens,
            STAGE_OUTPUT_TOKEN_CEILINGS[spec.key],
        )
        if spec.key in self.stage_budget_overrides:
            return min(floor_budget, ceiling)
        document_chars = len(subtask.document_text or '')
        module_count = len(subtask.modules)
        (
            base_tokens,
            per_module_tokens,
            chars_per_token,
        ) = STAGE_OUTPUT_ESTIMATES[spec.key]
        document_tokens = min(
            document_chars // max(chars_per_token, 1),
            6000,
        )
        estimated_output_tokens = (
            base_tokens
            + per_module_tokens * max(module_count, 1)
            + document_tokens
        )
        estimated_budget = (
            estimated_output_tokens
            + STAGE_REASONING_TOKEN_RESERVES[spec.key]
        )
        return min(
            ceiling,
            max(
                MIN_OUTPUT_TOKEN_FLOOR,
                floor_budget,
                estimated_budget,
            ),
        )

    @staticmethod
    def _build_subtask_prompts(
        *,
        spec: _StageSpec,
        subtask: _Subtask,
        base_system_prompt: str,
        base_user_prompt: str,
        repair_note: str = '',
    ) -> tuple[str, str]:
        scope_lines = []
        if subtask.modules:
            scope_lines.append(
                'Only analyze these feature modules: '
                + '; '.join(subtask.modules)
            )
        if subtask.source_titles:
            scope_lines.append(
                'Source sections: ' + '; '.join(subtask.source_titles)
            )
        scope = '\n'.join(scope_lines)
        repair_block = f'{repair_note}\n' if repair_note else ''
        system_prompt = (
            f'{base_system_prompt}\n\n'
            f'Current subtask: {subtask.label}\n'
            f'{subtask.focus}\n'
            f'{repair_block}'
            'Keep the model thinking mode enabled. Return only valid JSON. '
            'Do not omit requested analysis fields.'
        )
        document_text = subtask.document_text or ''
        user_prompt = (
            f'{base_user_prompt}\n\n'
            f'Current subtask: {subtask.label}\n'
            f'{scope}\n\n'
            f'Source document scope:\n{document_text}'
        )
        return system_prompt, user_prompt

    @staticmethod
    def _merge_subtask_payloads(
        spec: _StageSpec,
        payloads: list[dict[str, Any]],
    ) -> dict[str, Any]:
        merged: dict[str, Any] = {}
        for field_name in spec.schema.model_fields:
            values = [
                payload.get(field_name)
                for payload in payloads
            ]
            if any(isinstance(value, list) for value in values):
                merged[field_name] = _merge_raw_list_items(
                    item
                    for value in values
                    if isinstance(value, list)
                    for item in value
                )
                continue

            selected = next(
                (
                    value for value in values
                    if isinstance(value, str) and value.strip()
                ),
                None,
            )
            if selected is None and values:
                selected = values[0]
            merged[field_name] = selected
        return merged

    @staticmethod
    def _split_subtask(subtask: _Subtask) -> list[_Subtask]:
        if len(subtask.modules) > 1:
            midpoint = max(1, len(subtask.modules) // 2)
            module_groups = (
                subtask.modules[:midpoint],
                subtask.modules[midpoint:],
            )
            return [
                _Subtask(
                    key=f'{subtask.key}_part{index}',
                    label=(
                        f'{subtask.label}（拆分子任务 {index}）'
                    ),
                    focus=subtask.focus,
                    modules=modules,
                    source_titles=subtask.source_titles,
                    document_text=subtask.document_text,
                )
                for index, modules in enumerate(module_groups, start=1)
                if modules
            ]

        document_text = subtask.document_text or ''
        if len(document_text) < MIN_SPLIT_DOCUMENT_CHARS:
            return []
        sections = _split_document_sections(document_text)
        if len(sections) > 1:
            section_groups = _split_sections_in_half(sections)
            return [
                _Subtask(
                    key=f'{subtask.key}_section{index}',
                    label=(
                        f'{subtask.label}（文档片段 {index}）'
                    ),
                    focus=subtask.focus,
                    modules=subtask.modules,
                    source_titles=tuple(title for title, _ in group),
                    document_text='\n\n'.join(
                        text for _, text in group
                    ),
                )
                for index, group in enumerate(section_groups, start=1)
                if group
            ]

        text_parts = _split_text_in_half(document_text)
        if len(text_parts) > 1:
            return [
                _Subtask(
                    key=f'{subtask.key}_text{index}',
                    label=(
                        f'{subtask.label}（文本片段 {index}）'
                    ),
                    focus=subtask.focus,
                    modules=subtask.modules,
                    source_titles=subtask.source_titles,
                    document_text=text,
                )
                for index, text in enumerate(text_parts, start=1)
                if text.strip()
            ]
        return []

    def _request_with_retries(
        self,
        *,
        client: OpenAI,
        spec: _StageSpec,
        request_trace: AIRequestTrace,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int,
        trace: AIAnalysisTrace,
        on_event: Optional[AnalysisEventCallback],
    ) -> Any:
        total_attempts = self.max_retries + 1
        request_started = time.perf_counter()
        request_options = {
            'model': self.model,
            'messages': [
                {
                    'role': 'system',
                    'content': system_prompt,
                },
                {
                    'role': 'user',
                    'content': user_prompt,
                },
            ],
            'response_format': {'type': 'json_object'},
            'temperature': 0.2,
            'max_tokens': max_tokens,
            'timeout': self.timeout_seconds,
        }
        request_trace.request_thinking_config = {
            'mode': 'model_default',
            'reasoning_enabled': True,
        }
        request_trace.request_reasoning_effort = None
        request_trace.request_max_tokens = request_options['max_tokens']

        logger.info(
            'Stage API request config: stage=%s model=%s '
            'max_tokens=%d thinking=%s '
            'response_format=%s timeout=%.1fs',
            spec.key,
            request_options['model'],
            request_options['max_tokens'],
            json.dumps(
                request_trace.request_thinking_config,
                ensure_ascii=False,
                sort_keys=True,
            ),
            json.dumps(
                request_options['response_format'],
                ensure_ascii=False,
                sort_keys=True,
            ),
            request_options['timeout'],
        )

        attempt = 0
        failed_attempts = 0
        budget_reductions = 0

        while True:
            attempt += 1
            request_trace.attempts = attempt
            try:
                response = client.chat.completions.create(**request_options)
            except APITimeoutError as exc:
                failed_attempts += 1
                if failed_attempts < total_attempts:
                    self._emit_retry_event(
                        spec=spec,
                        attempt=failed_attempts,
                        total_attempts=total_attempts,
                        reason='请求超时',
                        on_event=on_event,
                    )
                    continue
                elapsed = self._finish_request_elapsed(
                    request_trace,
                    trace,
                    request_started,
                )
                self._record_stage_exception(request_trace, exc)
                self._record_exception(trace, exc)
                logger.error(
                    'Stage request timed out: stage=%s model=%s '
                    'attempt=%d elapsed=%.2fs',
                    spec.key,
                    self.model,
                    failed_attempts,
                    elapsed,
                )
                raise AIRequestTimeoutError(
                    f'阶段「{spec.label}」请求超时：'
                    f'{self.timeout_seconds:.1f} 秒，尝试 '
                    f'{failed_attempts} 次，累计耗时 {elapsed:.2f} 秒',
                    trace=trace,
                    stage=spec.key,
                    details={'attempts': failed_attempts},
                ) from exc
            except APIError as exc:
                safe_message = _sanitize_text(str(exc))
                status_code = getattr(exc, 'status_code', None)
                current_max_tokens = request_options['max_tokens']
                if (
                    _is_max_tokens_rejection(exc)
                    and budget_reductions < MAX_OUTPUT_BUDGET_REDUCTIONS
                    and current_max_tokens > MIN_OUTPUT_TOKEN_FLOOR
                ):
                    reported_limit = _extract_max_tokens_limit(exc)
                    if (
                        reported_limit is not None
                        and reported_limit < current_max_tokens
                    ):
                        reduced_tokens = reported_limit
                    else:
                        reduced_tokens = current_max_tokens // 2
                    reduced_tokens = max(
                        MIN_OUTPUT_TOKEN_FLOOR,
                        reduced_tokens,
                    )
                    if reduced_tokens < current_max_tokens:
                        budget_reductions += 1
                        request_options['max_tokens'] = reduced_tokens
                        request_trace.request_max_tokens = reduced_tokens
                        logger.warning(
                            'Model rejected max_tokens; reducing budget: '
                            'stage=%s previous=%d next=%d status=%s',
                            spec.key,
                            current_max_tokens,
                            reduced_tokens,
                            status_code,
                        )
                        self._emit(
                            on_event,
                            AnalysisEvent(
                                stage=spec.key,
                                state='running',
                                message=(
                                    f'模型不接受 max_tokens='
                                    f'{current_max_tokens}，'
                                    f'降为 {reduced_tokens} 后重试'
                                ),
                                metadata={
                                    'previous_max_tokens': (
                                        current_max_tokens
                                    ),
                                    'next_max_tokens': reduced_tokens,
                                    'budget_reduction': budget_reductions,
                                },
                            ),
                        )
                        continue

                failed_attempts += 1
                if (
                    _is_retryable_api_error(exc)
                    and failed_attempts < total_attempts
                ):
                    self._emit_retry_event(
                        spec=spec,
                        attempt=failed_attempts,
                        total_attempts=total_attempts,
                        reason=(
                            f'API 错误 {type(exc).__name__}'
                            f'（status={status_code}）'
                        ),
                        on_event=on_event,
                    )
                    continue

                elapsed = self._finish_request_elapsed(
                    request_trace,
                    trace,
                    request_started,
                )
                self._record_stage_exception(request_trace, exc)
                self._record_exception(trace, exc)
                logger.error(
                    'Stage API request failed: stage=%s model=%s '
                    'status_code=%s attempt=%d elapsed=%.2fs '
                    'error_type=%s error=%s',
                    spec.key,
                    self.model,
                    status_code,
                    failed_attempts,
                    elapsed,
                    type(exc).__name__,
                    safe_message,
                )
                raise AIRequestError(
                    f'阶段「{spec.label}」API 请求失败：'
                    f'{type(exc).__name__}: {safe_message}',
                    trace=trace,
                    stage=spec.key,
                    details={
                        'exception_type': type(exc).__name__,
                        'status_code': status_code,
                        'attempts': failed_attempts,
                    },
                ) from exc
            except Exception as exc:
                elapsed = self._finish_request_elapsed(
                    request_trace,
                    trace,
                    request_started,
                )
                self._record_stage_exception(request_trace, exc)
                self._record_exception(trace, exc)
                logger.error(
                    'Unexpected stage request failure: stage=%s '
                    'model=%s attempt=%d elapsed=%.2fs '
                    'error_type=%s error=%s\n%s',
                    spec.key,
                    self.model,
                    attempt,
                    elapsed,
                    type(exc).__name__,
                    _sanitize_text(str(exc)),
                    request_trace.exception_traceback,
                )
                raise AIRequestError(
                    f'阶段「{spec.label}」请求发生未预期异常：'
                    f'{type(exc).__name__}: '
                    f'{_sanitize_text(str(exc))}',
                    trace=trace,
                    stage=spec.key,
                    details={'exception_type': type(exc).__name__},
                ) from exc

            elapsed = self._finish_request_elapsed(
                request_trace,
                trace,
                request_started,
            )
            logger.info(
                'Stage response received: stage=%s model=%s '
                'attempt=%d budget_reductions=%d elapsed=%.2fs',
                spec.key,
                self.model,
                attempt,
                budget_reductions,
                elapsed,
            )
            return response

    def _emit_retry_event(
        self,
        *,
        spec: _StageSpec,
        attempt: int,
        total_attempts: int,
        reason: str,
        on_event: Optional[AnalysisEventCallback],
    ) -> None:
        delay = self._retry_delay(attempt)
        logger.warning(
            'Retrying stage request: stage=%s attempt=%d/%d '
            'reason=%s delay=%.2fs',
            spec.key,
            attempt,
            total_attempts,
            reason,
            delay,
        )
        self._emit(
            on_event,
            AnalysisEvent(
                stage=spec.key,
                state='running',
                message=(
                    f'{reason}，{delay:.1f} 秒后进行第 '
                    f'{attempt + 1}/{total_attempts} 次尝试'
                ),
                metadata={
                    'attempt': attempt,
                    'total_attempts': total_attempts,
                    'retry_delay_seconds': delay,
                },
            ),
        )
        if delay > 0:
            time.sleep(delay)

    def _retry_delay(self, attempt: int) -> float:
        delay = self.retry_backoff_seconds * (2 ** (attempt - 1))
        return min(delay, 10.0)

    @staticmethod
    def _finish_request_elapsed(
        request_trace: AIRequestTrace,
        trace: AIAnalysisTrace,
        started_at: float,
    ) -> float:
        elapsed = time.perf_counter() - started_at
        request_trace.request_elapsed_seconds = elapsed
        trace.request_elapsed_seconds = (
            (trace.request_elapsed_seconds or 0.0) + elapsed
        )
        return elapsed

    def _parse_and_validate(
        self,
        *,
        spec: _StageSpec,
        stage_trace: AIStageTrace,
        response: Any,
        trace: AIAnalysisTrace,
    ) -> BaseModel:
        metadata = _build_response_metadata(response)
        stage_trace.response_metadata = metadata
        stage_trace.response_id = metadata.get('response_id')
        stage_trace.response_model = metadata.get('response_model')
        stage_trace.finish_reason = metadata.get('finish_reason')
        stage_trace.usage = metadata.get('usage', {})

        trace.response_id = stage_trace.response_id
        trace.response_model = stage_trace.response_model
        trace.finish_reason = stage_trace.finish_reason
        trace.response_metadata = metadata
        trace.usage = _merge_usage(trace.usage, stage_trace.usage)

        logger.info(
            'Stage response metadata: stage=%s response_id=%s '
            'finish_reason=%s usage=%s',
            spec.key,
            stage_trace.response_id,
            stage_trace.finish_reason,
            json.dumps(stage_trace.usage, ensure_ascii=False, default=str),
        )

        if response is None:
            raise AIResponseStructureError(
                f'阶段「{spec.label}」响应对象为空',
                trace=trace,
                stage=spec.key,
                details=metadata,
            )

        choices = getattr(response, 'choices', None)
        if not choices:
            raise AIResponseStructureError(
                f'阶段「{spec.label}」响应缺少 choices 或 choices 为空',
                trace=trace,
                stage=spec.key,
                details=metadata,
            )

        message = getattr(choices[0], 'message', None)
        if message is None:
            raise AIResponseStructureError(
                f'阶段「{spec.label}」响应缺少 message',
                trace=trace,
                stage=spec.key,
                details=metadata,
            )

        content = getattr(message, 'content', None)
        if not isinstance(content, str) or not content.strip():
            truncated = stage_trace.finish_reason == 'length'
            stage_trace.truncated = truncated
            trace.truncated = trace.truncated or truncated
            summary = _response_summary(metadata)
            truncation_note = (
                '；输出已达到 Token 上限并被截断'
                if truncated
                else ''
            )
            logger.warning(
                'Stage returned empty content: stage=%s '
                'truncated=%s metadata=%s',
                spec.key,
                truncated,
                json.dumps(metadata, ensure_ascii=False, default=str),
            )
            if truncated:
                raise AIOutputTruncatedError(
                    f'阶段「{spec.label}」输出为空且达到 Token 上限'
                    f'（{summary}）',
                    trace=trace,
                    stage=spec.key,
                    details={
                        **metadata,
                        'truncated': True,
                        'max_tokens': (
                            stage_trace.request_max_tokens
                            or self.max_output_tokens
                        ),
                    },
                )
            raise AIEmptyResponseError(
                f'阶段「{spec.label}」模型返回空内容'
                f'{truncation_note}（{summary}）',
                trace=trace,
                stage=spec.key,
                details={
                    **metadata,
                    'truncated': truncated,
                },
            )

        stage_trace.raw_content = content
        stage_trace.output_char_count = len(content)
        trace.raw_content = content

        reasoning_tokens = _usage_reasoning_tokens(stage_trace.usage)
        requested_max_tokens = (
            stage_trace.request_max_tokens or self.max_output_tokens
        )
        logger.info(
            'Stage output received: stage=%s response_id=%s '
            'finish_reason=%s completion_tokens=%s '
            'reasoning_tokens=%s output_chars=%d max_tokens=%d '
            'request_thinking=%s request_reasoning_effort=%s',
            spec.key,
            stage_trace.response_id,
            stage_trace.finish_reason,
            stage_trace.usage.get('completion_tokens'),
            reasoning_tokens,
            len(content),
            requested_max_tokens,
            json.dumps(
                stage_trace.request_thinking_config,
                ensure_ascii=False,
                sort_keys=True,
            ),
            stage_trace.request_reasoning_effort,
        )
        if isinstance(reasoning_tokens, int) and reasoning_tokens > 0:
            logger.info(
                'Thinking mode active: stage=%s finish_reason=%s '
                'reasoning_tokens=%s max_tokens=%d thinking=%s',
                spec.key,
                stage_trace.finish_reason,
                reasoning_tokens,
                requested_max_tokens,
                json.dumps(
                    stage_trace.request_thinking_config,
                    ensure_ascii=False,
                    sort_keys=True,
                ),
            )

        if stage_trace.finish_reason == 'length':
            stage_trace.truncated = True
            trace.truncated = True
            json_error = _inspect_json_error(content)
            stage_trace.json_error = json_error
            trace.json_error = json_error
            logger.error(
                'Stage output truncated: stage=%s response_id=%s '
                'finish_reason=%s completion_tokens=%s '
                'reasoning_tokens=%s output_chars=%d max_tokens=%d '
                'json_error=%s',
                spec.key,
                stage_trace.response_id,
                stage_trace.finish_reason,
                stage_trace.usage.get('completion_tokens'),
                reasoning_tokens,
                len(content),
                requested_max_tokens,
                json_error,
            )
            raise AIOutputTruncatedError(
                f'阶段「{spec.label}」输出被截断：'
                f'finish_reason=length，'
                f'max_tokens={requested_max_tokens}，'
                f'输出 {len(content):,} 字符'
                + (
                    f'；截断位置：{json_error}'
                    if json_error
                    else ''
                ),
                trace=trace,
                stage=spec.key,
                details={
                    **metadata,
                    'truncated': True,
                    'max_tokens': requested_max_tokens,
                    'json_error': json_error,
                    'request_thinking_config': (
                        stage_trace.request_thinking_config
                    ),
                    'request_reasoning_effort': (
                        stage_trace.request_reasoning_effort
                    ),
                },
            )

        try:
            data = json.loads(content)
        except json.JSONDecodeError as exc:
            json_error = _format_json_decode_error(exc)
            stage_trace.json_error = json_error
            trace.json_error = json_error
            logger.error(
                'Stage JSON parse failed: stage=%s response_id=%s '
                'finish_reason=%s completion_tokens=%s '
                'reasoning_tokens=%s output_chars=%d max_tokens=%d '
                'error=%s',
                spec.key,
                stage_trace.response_id,
                stage_trace.finish_reason,
                stage_trace.usage.get('completion_tokens'),
                reasoning_tokens,
                len(content),
                requested_max_tokens,
                json_error,
            )
            raise AIJSONParseError(
                f'阶段「{spec.label}」返回内容不是合法 JSON：'
                f'{json_error}',
                trace=trace,
                stage=spec.key,
                details={
                    'response_id': stage_trace.response_id,
                    'json_error': json_error,
                },
            ) from exc

        try:
            payload = spec.schema.model_validate(data)
        except ValidationError as exc:
            field_errors = _extract_validation_errors(exc)
            stage_trace.validation_error = _sanitize_text(str(exc))
            stage_trace.validation_field_errors = field_errors
            trace.validation_error = stage_trace.validation_error
            trace.validation_field_errors = field_errors
            logger.error(
                'Stage Pydantic validation failed: stage=%s fields=%s',
                spec.key,
                json.dumps(field_errors, ensure_ascii=False, default=str),
            )
            raise AIValidationError(
                f'阶段「{spec.label}」字段校验失败：'
                f'{_format_field_errors(field_errors)}',
                trace=trace,
                stage=spec.key,
                details={'field_errors': field_errors},
            ) from exc

        logger.info(
            'Stage validation succeeded: stage=%s schema=%s '
            'output_chars=%d',
            spec.key,
            spec.schema.__name__,
            len(content),
        )
        return payload

    def _merge_stage_results(
        self,
        stage_payloads: Mapping[str, Mapping[str, Any]],
        trace: AIAnalysisTrace,
        document_content: str = '',
    ) -> ProductAnalysis:
        expected_stage_keys = [spec.key for spec in STAGE_SPECS]
        missing_stages = [
            key for key in expected_stage_keys
            if key not in stage_payloads
        ]
        if missing_stages:
            missing_stage_text = ', '.join(missing_stages)
            raise AIIntegrityError(
                f'结果合并失败，缺少阶段：{missing_stage_text}',
                trace=trace,
                stage='final',
                details={'missing_stages': missing_stages},
            )

        merged: dict[str, Any] = {}
        field_sources: dict[str, str] = {}
        requirement_items: dict[str, list[RequirementItem]] = {
            field_name: [] for field_name in REQUIREMENT_LIST_FIELDS
        }
        normalized_document = _normalize_evidence_text(document_content)
        for spec in STAGE_SPECS:
            payload = stage_payloads[spec.key]
            expected_fields = set(spec.schema.model_fields)
            actual_fields = set(payload)
            missing_fields = expected_fields - actual_fields
            unexpected_fields = actual_fields - expected_fields
            if missing_fields or unexpected_fields:
                raise AIIntegrityError(
                    f'阶段「{spec.label}」字段不完整或存在多余字段',
                    trace=trace,
                    stage='final',
                    details={
                        'stage_key': spec.key,
                        'missing_fields': sorted(missing_fields),
                        'unexpected_fields': sorted(unexpected_fields),
                    },
                )

            for field_name in expected_fields:
                if field_name in merged:
                    raise AIIntegrityError(
                        f'结果合并失败，字段重复：{field_name}',
                        trace=trace,
                        stage='final',
                        details={'duplicate_field': field_name},
                    )
                if field_name in ANALYSIS_LIST_FIELDS:
                    model_cls = ANALYSIS_FIELD_MODELS[field_name]
                    grounded_items: list[AnalysisItem] = []
                    try:
                        for value in payload[field_name]:
                            item = (
                                value
                                if isinstance(value, model_cls)
                                else model_cls.model_validate(value)
                            )
                            grounded_items.append(
                                _ground_analysis_item(
                                    item,
                                    normalized_document,
                                    document_content,
                                )
                            )
                    except ValidationError as exc:
                        field_errors = _extract_validation_errors(exc)
                        trace.validation_error = _sanitize_text(str(exc))
                        trace.validation_field_errors = field_errors
                        raise AIValidationError(
                            f'字段「{field_name}」校验失败：'
                            f'{_format_field_errors(field_errors)}',
                            trace=trace,
                            stage='final',
                            details={
                                'field_name': field_name,
                                'field_errors': field_errors,
                            },
                        ) from exc

                    deduped_items = _dedupe_analysis_models(
                        grounded_items,
                        same_source_is_duplicate=(
                            field_name in REQUIREMENT_LIST_FIELDS
                        ),
                    )
                    duplicate_count = len(grounded_items) - len(deduped_items)
                    if duplicate_count:
                        logger.info(
                            'Merged semantically duplicate analysis items: '
                            'field=%s removed=%d',
                            field_name,
                            duplicate_count,
                        )
                    if field_name in REQUIREMENT_LIST_FIELDS:
                        requirement_items[field_name] = [
                            item
                            for item in deduped_items
                            if isinstance(item, RequirementItem)
                        ]
                    else:
                        merged[field_name] = [
                            item.model_dump() for item in deduped_items
                        ]
                else:
                    merged[field_name] = payload[field_name]
                field_sources[field_name] = spec.key

        deduped_requirements = _dedupe_requirement_items(
            requirement_items
        )
        for field_name in REQUIREMENT_LIST_FIELDS:
            merged[field_name] = [
                item.model_dump()
                for item in deduped_requirements[field_name]
            ]

        expected_fields = set(ProductAnalysis.model_fields)
        missing_fields = expected_fields - set(merged)
        unexpected_fields = set(merged) - expected_fields
        if missing_fields or unexpected_fields:
            raise AIIntegrityError(
                '最终结果字段与 ProductAnalysis 不一致',
                trace=trace,
                stage='final',
                details={
                    'missing_fields': sorted(missing_fields),
                    'unexpected_fields': sorted(unexpected_fields),
                },
            )

        try:
            analysis = ProductAnalysis.model_validate(merged)
        except ValidationError as exc:
            field_errors = _extract_validation_errors(exc)
            trace.validation_error = _sanitize_text(str(exc))
            trace.validation_field_errors = field_errors
            logger.error(
                'Final merged Pydantic validation failed: fields=%s',
                json.dumps(field_errors, ensure_ascii=False, default=str),
            )
            raise AIValidationError(
                '分阶段结果最终合并校验失败：'
                f'{_format_field_errors(field_errors)}',
                trace=trace,
                stage='final',
                details={
                    'field_errors': field_errors,
                    'field_sources': field_sources,
                },
            ) from exc

        normalized = analysis.model_dump()
        for field_name in expected_fields:
            if normalized[field_name] != merged[field_name]:
                raise AIIntegrityError(
                    f'结果合并后字段内容发生变化：{field_name}',
                    trace=trace,
                    stage='final',
                    details={'field_name': field_name},
                )

        if not analysis.product_summary.strip():
            raise AIValidationError(
                '最终结果 product_summary 为空',
                trace=trace,
                stage='final',
            )
        if not analysis.core_features:
            raise AIValidationError(
                '最终结果 core_features 为空',
                trace=trace,
                stage='final',
            )

        logger.info(
            'Stage results merged successfully: fields=%s',
            ', '.join(sorted(expected_fields)),
        )
        return analysis


def _build_subtasks(
    *,
    spec: _StageSpec,
    completed_stage_payloads: Mapping[str, Mapping[str, Any]],
    document_content: str,
) -> list[_Subtask]:
    modules = _extract_modules(
        completed_stage_payloads.get('generation_1', {}).get(
            'core_features',
            [],
        )
    )

    if spec.key == 'generation_1':
        return [
            _Subtask(
                key='overview',
                label='产品概述与核心功能',
                focus=(
                    f'{spec.focus}\n'
                    'For every core feature, include the module name and '
                    'its primary functions in one concise item.'
                ),
                source_titles=('完整文档',),
                document_text=document_content,
            )
        ]

    if spec.key in {'generation_2', 'generation_3'}:
        if spec.key == 'generation_2':
            category_focus = (
                'For the listed modules, extract business process rules, '
                'state transition rules, permission rules, data validation '
                'rules, and conditional constraints. Do not omit a category.'
            )
        else:
            category_focus = (
                'For the listed modules, identify exception cases, boundary '
                'conditions, permission and security risks, data consistency '
                'risks, state anomalies, and interrupted flows.'
            )
        batches = _batch_modules(modules, MODULE_BATCH_SIZE)
        if not batches:
            batches = [()]
        return [
            _Subtask(
                key=f'{spec.key}_batch_{index}',
                label=(
                    '全部模块' if not batch
                    else '模块组：' + '；'.join(
                        _short_text(module, 24) for module in batch
                    )
                ),
                focus=f'{spec.focus}\n{category_focus}',
                modules=tuple(batch),
                source_titles=('完整文档',),
                document_text=document_content,
            )
            for index, batch in enumerate(batches, start=1)
        ]

    if spec.key == 'generation_4':
        subtask_specs = (
            (
                'conflicts',
                'A. 需求冲突识别',
                (
                    'Only populate requirement_conflicts. Set the other three '
                    'lists to []. A conflict requires two explicit PRD '
                    'statements that cannot both be true. Quote both exact '
                    'excerpts in source_text separated by “｜”. Missing '
                    'details or speculative concerns are not conflicts.'
                ),
            ),
            (
                'ambiguities',
                'B. 需求歧义识别',
                (
                    'Only populate requirement_ambiguities. Set the other '
                    'three lists to []. Report only phrasing with at least '
                    'two materially different interpretations that affect '
                    'implementation, acceptance, or testing. Do not report '
                    'conflicts or product decisions already raised in PRD.'
                ),
            ),
            (
                'missing',
                'C. 需求缺失识别',
                (
                    'Only populate missing_requirements. Set the other three '
                    'lists to []. Report a required implementation or test '
                    'detail that PRD does not define and whose absence changes '
                    'behavior or acceptance. Do not report generic best '
                    'practices, speculative improvements, or conflicts.'
                ),
            ),
            (
                'pending',
                'D. 待确认问题识别',
                (
                    'Only populate pending_questions. Set the other three '
                    'lists to []. Report questions explicitly raised in PRD '
                    'or a product/business decision that cannot be classified '
                    'as conflict, ambiguity, or missing. Do not restate an '
                    'issue that belongs to one of the first three lists.'
                ),
            ),
        )
        module_groups = _batch_modules_for_max_batches(
            modules,
            MAX_MODULE_BATCHES_FOR_AMBIGUITY,
        )
        if not module_groups:
            module_groups = [()]

        subtasks: list[_Subtask] = []
        for batch_index, module_group in enumerate(
            module_groups,
            start=1,
        ):
            batch_suffix = (
                ''
                if len(module_groups) == 1
                else f'_modules{batch_index}'
            )
            module_scope = ''
            if module_group:
                module_scope = (
                    '\nModules in scope: '
                    + '; '.join(
                        _short_text(module, 24)
                        for module in module_group
                    )
                )
            for key, label, focus in subtask_specs:
                subtasks.append(
                    _Subtask(
                        key=f'generation_4_{key}{batch_suffix}',
                        label=(
                            label
                            if len(module_groups) == 1
                            else f'{label}（模块组 {batch_index}）'
                        ),
                        focus=f'{spec.focus}\n{focus}{module_scope}',
                        modules=tuple(module_group),
                        source_titles=(
                            tuple(module_group)
                            if module_group
                            else ('完整文档',)
                        ),
                        document_text=document_content,
                    )
                )

        if len(module_groups) > 1:
            subtasks.append(
                _Subtask(
                    key='generation_4_cross_module',
                    label='D. 跨模块冲突与一致性',
                    focus=(
                        f'{spec.focus}\n'
                        'Only identify cross-module contradictions or '
                        'inconsistencies that span feature modules, roles, or '
                        'states. Classify each into exactly one of the four '
                        'lists. Do not repeat per-module issues.'
                    ),
                    source_titles=('完整文档',),
                    document_text=document_content,
                )
            )
        return subtasks

    return [
        _Subtask(
            key=spec.key,
            label=spec.label,
            focus=spec.focus,
            source_titles=('完整文档',),
            document_text=document_content,
        )
    ]


def _extract_modules(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    modules = []
    for item in value:
        if isinstance(item, str) and item.strip():
            modules.append(item.strip())
    return list(dict.fromkeys(modules))


def _batch_modules(
    modules: list[str],
    batch_size: int,
) -> list[tuple[str, ...]]:
    if not modules:
        return []
    return [
        tuple(modules[index:index + batch_size])
        for index in range(0, len(modules), batch_size)
    ]


def _batch_modules_for_max_batches(
    modules: list[str],
    max_batches: int,
) -> list[tuple[str, ...]]:
    if not modules or max_batches <= 0:
        return []
    if len(modules) <= MODULE_BATCH_SIZE:
        return [tuple(modules)]
    batch_size = (len(modules) + max_batches - 1) // max_batches
    return _batch_modules(modules, max(1, batch_size))


_LIST_ITEM_MARKER_RE = re.compile(
    r'^\s*(?:[-*•·]|(?:\d+|[一二三四五六七八九十]+)[.)、]|'
    r'[（(](?:\d+|[一二三四五六七八九十]+)[)）])\s*'
)


def _normalize_analysis_item(value: str) -> str:
    text = value.strip()
    while text:
        stripped = _LIST_ITEM_MARKER_RE.sub('', text, count=1).strip()
        if stripped == text:
            break
        text = stripped
    return re.sub(r'\s+', ' ', text).strip()


def _text_fingerprint(text: str) -> str:
    return re.sub(
        r'[\s，。；、,.!?！？:：\-—（）()\[\]【】]+',
        '',
        text,
    ).casefold()


def _dedupe_texts(values: Any) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        if not isinstance(value, str):
            continue
        text = _normalize_analysis_item(value)
        if not text:
            continue
        normalized = _text_fingerprint(text)
        if normalized in seen:
            continue
        seen.add(normalized)
        result.append(text)
    return result


def _merge_raw_list_items(values: Any) -> list[Any]:
    result: list[Any] = []
    seen: set[tuple[str, str]] = set()
    for value in values:
        if isinstance(value, str):
            text = _normalize_analysis_item(value)
            if not text:
                continue
            key = ('text', _text_fingerprint(text))
            item: Any = text
        elif isinstance(value, dict):
            key = (
                'dict',
                json.dumps(
                    value,
                    ensure_ascii=False,
                    sort_keys=True,
                    default=str,
                ),
            )
            item = value
        else:
            continue
        if key in seen:
            continue
        seen.add(key)
        result.append(item)
    return result


def _normalize_evidence_text(text: str) -> str:
    return re.sub(
        r'[\s#>*_`~“”"\'‘’（）()\[\]【】{}，。；、,.!?！？:：\-—|｜]+',
        '',
        text,
    ).casefold()


def _evidence_source_fragments(source_text: str) -> set[str]:
    return {
        normalized
        for part in re.split(r'\s*(?:｜|\|\||\n+)\s*', source_text)
        if part.strip()
        for normalized in (_normalize_evidence_text(part),)
        if normalized
    }


def _source_text_is_grounded(
    source_text: str,
    normalized_document: str,
) -> bool:
    normalized_source = _normalize_evidence_text(source_text)
    if not normalized_source:
        return False
    if normalized_source in normalized_document:
        return True

    parts = list(_evidence_source_fragments(source_text))
    return (
        len(parts) > 1
        and all(len(part) >= 4 for part in parts)
        and all(part in normalized_document for part in parts)
    )


def _sections_for_source_text(
    source_text: str,
    document_content: str,
) -> list[str]:
    fragments = sorted(
        _evidence_source_fragments(source_text),
        key=len,
        reverse=True,
    )
    if not fragments:
        return []

    matched_titles: list[str] = []
    for title, section_text in _split_document_sections(document_content):
        normalized_section = _normalize_evidence_text(section_text)
        if any(fragment in normalized_section for fragment in fragments):
            if title not in matched_titles:
                matched_titles.append(title)
    return matched_titles


def _canonicalize_source_section(
    source_section: str,
    document_content: str,
) -> str:
    raw_parts = [
        part.strip()
        for part in re.split(r'\s*[；;]\s*', source_section)
        if part.strip()
    ]
    if not raw_parts:
        return ''

    section_titles = [
        title
        for title, _ in _split_document_sections(document_content)
    ]
    normalized_titles = {
        _normalize_evidence_text(title): title
        for title in section_titles
    }
    canonical_parts: list[str] = []
    for part in raw_parts:
        normalized_part = _normalize_evidence_text(part)
        canonical = normalized_titles.get(normalized_part)
        if canonical is None:
            candidates = [
                title
                for normalized_title, title in normalized_titles.items()
                if (
                    normalized_part
                    and (
                        normalized_part in normalized_title
                        or normalized_title in normalized_part
                    )
                )
            ]
            if len(candidates) == 1:
                canonical = candidates[0]
        if canonical and canonical not in canonical_parts:
            canonical_parts.append(canonical)
    return '；'.join(canonical_parts)


def _ground_analysis_item(
    item: AnalysisItem,
    normalized_document: str,
    document_content: str = '',
) -> AnalysisItem:
    if not normalized_document or not item.source_text.strip():
        if not document_content or not item.source_section.strip():
            return item
        source_section = _canonicalize_source_section(
            item.source_section,
            document_content,
        )
        if source_section == item.source_section:
            return item
        data = item.model_dump()
        data['source_section'] = source_section
        return type(item).model_validate(data)
    if _source_text_is_grounded(item.source_text, normalized_document):
        if not document_content:
            return item
        source_sections = _sections_for_source_text(
            item.source_text,
            document_content,
        )
        source_section = '；'.join(source_sections)
        if source_section == item.source_section:
            return item
        data = item.model_dump()
        data['source_section'] = source_section
        return type(item).model_validate(data)

    analysis = item.analysis.strip()
    note = (
        '模型提供的原文片段未能在 PRD 中逐字定位，已移除该引用。'
    )
    if item.evidence_type == 'inferred':
        analysis = f'{note}推断依据：{analysis}'
    else:
        analysis = (
            f'{note}本项已由 explicit 降级为 inferred，'
            f'推断依据：{analysis}'
        )

    data = item.model_dump()
    data.update(
        evidence_type='inferred',
        source_section='',
        source_text='',
        analysis=analysis,
    )
    return type(item).model_validate(data)


def _analysis_content_key(text: str) -> str:
    normalized = _text_fingerprint(text)
    for phrase in (
        '系统',
        '功能',
        '需要',
        '必须',
        '应当',
        '要求',
        '可以',
        '进行',
        '实现',
    ):
        normalized = normalized.replace(phrase, '')
    return normalized


_REQUIREMENT_FOCUS_STOP_PHRASES = (
    '会议室',
    '管理员',
    '普通员工',
    '员工',
    '预约',
    '创建',
    '取消',
    '记录',
    '管理',
    '功能',
    '系统',
    '规则',
    '要求',
    '需要',
    '必须',
    '可以',
    '进行',
    '实现',
    '未定义',
    '需确认',
    '存在',
    '是否',
    '以及',
    '如果',
    '可能',
    '影响',
    '当前',
    '不同',
    '包括',
)


def _requirement_focus_key(text: str) -> str:
    normalized = _text_fingerprint(text)
    for phrase in _REQUIREMENT_FOCUS_STOP_PHRASES:
        normalized = normalized.replace(phrase, '')
    return normalized


def _requirement_focus_similarity(left: str, right: str) -> float:
    left_key = _requirement_focus_key(left)
    right_key = _requirement_focus_key(right)
    if not left_key or not right_key:
        return 0.0
    return SequenceMatcher(None, left_key, right_key).ratio()


def _char_ngrams(text: str, size: int = 2) -> set[str]:
    if len(text) <= size:
        return {text} if text else set()
    return {text[index:index + size] for index in range(len(text) - size + 1)}


def _semantic_similarity(left: str, right: str) -> float:
    left_key = _analysis_content_key(left)
    right_key = _analysis_content_key(right)
    if not left_key or not right_key:
        return 0.0
    if left_key == right_key:
        return 1.0

    sequence_score = SequenceMatcher(
        None,
        left_key,
        right_key,
    ).ratio()
    left_grams = _char_ngrams(left_key)
    right_grams = _char_ngrams(right_key)
    union = left_grams | right_grams
    jaccard_score = (
        len(left_grams & right_grams) / len(union)
        if union
        else 0.0
    )
    containment_score = (
        min(len(left_key), len(right_key))
        / max(len(left_key), len(right_key))
        if (left_key in right_key or right_key in left_key)
        else 0.0
    )
    return max(sequence_score, jaccard_score, containment_score)


_DISTINCT_CONTEXT_GROUPS = (
    (
        ('管理员',),
        ('普通员工', '员工'),
    ),
    (
        ('新增', '添加', '创建', '新建'),
        ('编辑', '修改', '调整'),
        ('停用', '禁用', '关闭'),
        ('启用', '开启'),
        ('取消', '撤销'),
        ('查看', '查询', '浏览'),
        ('筛选', '过滤'),
        ('审核', '审批'),
        ('删除',),
    ),
    (
        ('待开始',),
        ('进行中', '已开始'),
        ('已结束',),
        ('已取消',),
        ('已停用',),
    ),
)


def _marker_groups(text: str) -> list[set[int]]:
    return [
        {
            index
            for index, markers in enumerate(groups)
            if any(marker in text for marker in markers)
        }
        for groups in _DISTINCT_CONTEXT_GROUPS
    ]


def _has_distinct_context(left: AnalysisItem, right: AnalysisItem) -> bool:
    left_groups = _marker_groups(left.content)
    right_groups = _marker_groups(right.content)
    return any(
        left_markers
        and right_markers
        and left_markers.isdisjoint(right_markers)
        for left_markers, right_markers in zip(left_groups, right_groups)
    )


def _items_equivalent(
    left: AnalysisItem,
    right: AnalysisItem,
    *,
    same_source_is_duplicate: bool = False,
) -> bool:
    if _has_distinct_context(left, right):
        return False
    similarity = _semantic_similarity(left.content, right.content)
    left_sources = _evidence_source_fragments(left.source_text)
    right_sources = _evidence_source_fragments(right.source_text)
    shared_source = bool(left_sources & right_sources)
    if (
        same_source_is_duplicate
        and shared_source
    ):
        return (
            _requirement_focus_similarity(
                left.content,
                right.content,
            )
            >= 0.13
        )
    if similarity >= 0.9:
        return True

    if shared_source:
        return similarity >= 0.5

    if (
        left.source_section.strip()
        and left.source_section.strip() == right.source_section.strip()
        and similarity >= 0.82
    ):
        return True
    return False


def _item_completeness(item: AnalysisItem) -> float:
    score = 0.0
    if item.evidence_type == 'explicit':
        score += 500.0
    if item.source_text.strip():
        score += 200.0
    if item.source_section.strip():
        score += 30.0
    score += min(len(item.content.strip()), 500)
    score += min(len(item.analysis.strip()), 500) * 0.5

    risk_type = getattr(item, 'risk_type', '')
    if isinstance(risk_type, str) and risk_type.strip():
        score += 15.0
    priority = getattr(item, 'priority', None)
    score += {'P0': 12.0, 'P1': 6.0, 'P2': 0.0}.get(priority, 0.0)
    return score


def _dedupe_analysis_models(
    items: list[AnalysisItem],
    *,
    same_source_is_duplicate: bool = False,
) -> list[AnalysisItem]:
    if len(items) < 2:
        return items

    parents = list(range(len(items)))

    def find(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parents[right_root] = left_root

    for left_index, left_item in enumerate(items):
        for right_index in range(left_index + 1, len(items)):
            if _items_equivalent(
                left_item,
                items[right_index],
                same_source_is_duplicate=same_source_is_duplicate,
            ):
                union(left_index, right_index)

    groups: dict[int, list[AnalysisItem]] = {}
    for index, item in enumerate(items):
        groups.setdefault(find(index), []).append(item)
    return [
        max(group, key=_item_completeness)
        for group in groups.values()
    ]


def _priority_sort_key(item: RequirementItem) -> tuple[int, float]:
    return (
        {'P0': 0, 'P1': 1, 'P2': 2}.get(item.priority, 3),
        -_item_completeness(item),
    )


def _dedupe_requirement_items(
    items_by_field: Mapping[str, list[RequirementItem]],
) -> dict[str, list[RequirementItem]]:
    flattened: list[tuple[str, RequirementItem]] = []
    for field_name in REQUIREMENT_LIST_FIELDS:
        for item in items_by_field.get(field_name, []):
            flattened.append((field_name, item))

    if not flattened:
        return {field_name: [] for field_name in REQUIREMENT_LIST_FIELDS}

    parents = list(range(len(flattened)))

    def find(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parents[right_root] = left_root

    for left_index, (_, left_item) in enumerate(flattened):
        for right_index in range(left_index + 1, len(flattened)):
            if _items_equivalent(
                left_item,
                flattened[right_index][1],
                same_source_is_duplicate=True,
            ):
                union(left_index, right_index)

    field_rank = {
        field_name: index
        for index, field_name in enumerate(REQUIREMENT_LIST_FIELDS)
    }
    groups: dict[int, list[tuple[str, RequirementItem]]] = {}
    for index, pair in enumerate(flattened):
        groups.setdefault(find(index), []).append(pair)

    result = {field_name: [] for field_name in REQUIREMENT_LIST_FIELDS}
    for group in groups.values():
        selected_field, selected_item = max(
            group,
            key=lambda pair: (
                _item_completeness(pair[1]),
                -field_rank[pair[0]],
            ),
        )
        result[selected_field].append(selected_item)

    for field_name in REQUIREMENT_LIST_FIELDS:
        result[field_name].sort(key=_priority_sort_key)
    return result


def _split_document_sections(text: str) -> list[tuple[str, str]]:
    if not text.strip():
        return []
    matches = list(re.finditer(r'(?m)^#{1,6}\s+.+$', text))
    if not matches:
        return [('全文', text.strip())]

    sections: list[tuple[str, str]] = []
    prefix = text[:matches[0].start()].strip()
    if prefix:
        sections.append(('文档开头', prefix))
    for index, match in enumerate(matches):
        start = match.start()
        end = (
            matches[index + 1].start()
            if index + 1 < len(matches)
            else len(text)
        )
        title = match.group(0).lstrip('#').strip() or f'章节 {index + 1}'
        section_text = text[start:end].strip()
        if section_text:
            sections.append((title, section_text))
    return sections


def _split_sections_in_half(
    sections: list[tuple[str, str]],
) -> list[list[tuple[str, str]]]:
    if len(sections) < 2:
        return [sections]
    total_chars = sum(len(text) for _, text in sections)
    target = total_chars / 2
    running = 0
    split_index = 1
    for index, (_, text) in enumerate(sections):
        running += len(text)
        if running >= target and index + 1 < len(sections):
            split_index = index + 1
            break
    return [sections[:split_index], sections[split_index:]]


def _split_text_in_half(text: str) -> list[str]:
    if len(text) < 2:
        return [text]
    midpoint = len(text) // 2
    split_at = text.rfind('\n\n', 0, midpoint)
    if split_at <= 0:
        split_at = text.rfind('\n', 0, midpoint)
    if split_at <= 0:
        split_at = midpoint
    left = text[:split_at].strip()
    right = text[split_at:].strip()
    return [part for part in (left, right) if part]


def _short_text(value: str, limit: int) -> str:
    normalized = ' '.join(value.split())
    if len(normalized) <= limit:
        return normalized
    return normalized[:limit - 1] + '…'


def _build_system_prompt(spec: _StageSpec) -> str:
    required_fields = '、'.join(spec.schema.model_fields.keys())
    field_contract = _STAGE_OUTPUT_CONTRACTS.get(spec.key)
    contract_block = (
        f'\n    {field_contract}\n'
        if field_contract
        else ''
    )
    evidence_contract = ''
    if spec.key != 'generation_1':
        extra_fields = ''
        if spec.key == 'generation_3':
            extra_fields = ',\n      "risk_type": "并发"'
        elif spec.key == 'generation_4':
            extra_fields = ',\n      "priority": "P1"'
        evidence_contract = f'''
    分析条目统一结构：
    {{
      "content": "具体内容",
      "evidence_type": "explicit 或 inferred",
      "source_section": "PRD 来源章节；无法定位时为空字符串",
      "source_text": "PRD 原文片段；无法定位时为空字符串",
      "analysis": "分析说明"{extra_fields}
    }}

    证据与引用规则：
    - explicit 表示 PRD 原文明确描述；必须提供逐字摘录的 source_text。
    - inferred 表示根据 PRD 原文推断；analysis 必须以“推断依据：”说明推断依据。
    - source_text 必须能在 PRD 中逐字找到，不得改写、概括、拼接或编造。
    - source_section 必须使用 PRD 中的实际章节标题；无法确认时置为空字符串。
    - 找不到原文依据时，把 source_section 和 source_text 置为空，并在 analysis 中说明原因。
    - 无法给出任何原文依据的内容，不得标记为 explicit，也不得伪装成确定事实。
    - 不允许把一般性建议、测试建议或推测写成产品需求。
    ''' + (
            '- potential_risks 的每一条都必须包含 risk_type，'
            '不得省略；缺少该字段的 JSON 视为无效。\n'
            if spec.key == 'generation_3'
            else ''
        ) + (
            '- 四类需求问题的每一条都必须包含 priority，取值为 P0、P1 或 P2。\n'
            if spec.key == 'generation_4'
            else ''
        )
    return f'''
    你是一名拥有10年以上经验的软件测试开发工程师。

    本次只执行一个独立分析阶段：{spec.label}。
    {spec.focus}
    {contract_block}
    {evidence_contract}

    硬性要求：
    - 只能依据用户提供的原始文档进行分析。
    - 不允许编造产品功能、业务规则或风险。
    - 只返回一个合法 JSON 对象，不要返回 Markdown 代码块。
    - JSON 必须且只能包含这些字段：{required_fields}。
    - 列表中的每一条应是独立、完整、可验证的结论，并标注所属模块。
    - 合并语义相同、表达不同的条目，保留信息更完整、描述更准确的一条。
    - 不得为了减少数量而合并触发条件、角色、状态或结果不同的实际业务规则。
    - 同一根因或同一需求问题只能出现一次，禁止同义改写、近义重复和跨分类重复。
    - 每个推断都必须在 analysis 中明确写出“推断依据：”，不得把推断伪装成明确需求。
    - 输出保持精炼，避免与当前阶段无关的重复内容。
    '''


def _build_user_prompt(document_content: str) -> str:
    return f'''
    请分析以下产品需求文档：

    {document_content}
    '''


def _stage_success_message(stage_trace: AIStageTrace) -> str:
    elapsed = stage_trace.request_elapsed_seconds or 0.0
    usage = stage_trace.usage
    prompt_tokens = usage.get('prompt_tokens', 'N/A')
    completion_tokens = usage.get('completion_tokens', 'N/A')
    reasoning_tokens = _usage_reasoning_tokens(usage)
    output_chars = stage_trace.output_char_count or 0
    finish_reason = stage_trace.finish_reason or 'N/A'
    return (
        f'耗时 {elapsed:.2f} 秒；'
        f'Prompt Tokens={prompt_tokens}；'
        f'Completion Tokens={completion_tokens}；'
        f'Reasoning Tokens={reasoning_tokens}；'
        f'输出 {output_chars:,} 字符；'
        f'finish_reason={finish_reason}；'
        f'重试 {stage_trace.retry_count} 次'
    )


def _stage_error_message(
    stage_trace: AIStageTrace,
    exc: AIAnalysisError,
) -> str:
    parts = [str(exc)]
    if stage_trace.request_elapsed_seconds is not None:
        parts.append(
            f'耗时 {stage_trace.request_elapsed_seconds:.2f} 秒'
        )
    if stage_trace.usage:
        parts.append(f'Token Usage={_usage_summary(stage_trace.usage)}')
    if stage_trace.finish_reason:
        parts.append(f'finish_reason={stage_trace.finish_reason}')
    if stage_trace.truncated:
        parts.append('已标记为输出截断')
    return '；'.join(parts)


def _build_response_metadata(response: Any) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        'response_id': getattr(response, 'id', None),
        'response_model': getattr(response, 'model', None),
        'object': getattr(response, 'object', None),
        'created': getattr(response, 'created', None),
        'system_fingerprint': getattr(response, 'system_fingerprint', None),
        'choices_count': None,
        'finish_reason': None,
        'choice_index': None,
        'has_message': False,
        'message_role': None,
        'content_present': False,
        'content_type': None,
        'content_length': None,
        'content_is_empty': True,
        'refusal_present': False,
        'usage': {},
    }

    choices = getattr(response, 'choices', None)
    if choices:
        metadata['choices_count'] = _safe_len(choices)
        choice = choices[0]
        metadata['finish_reason'] = getattr(choice, 'finish_reason', None)
        metadata['choice_index'] = getattr(choice, 'index', None)

        message = getattr(choice, 'message', None)
        metadata['has_message'] = message is not None
        if message is not None:
            metadata['message_role'] = getattr(message, 'role', None)
            metadata['content_present'] = hasattr(message, 'content')
            content = getattr(message, 'content', None)
            metadata['content_type'] = (
                type(content).__name__ if content is not None else None
            )
            metadata['content_length'] = (
                len(content) if isinstance(content, str) else None
            )
            metadata['content_is_empty'] = (
                not isinstance(content, str) or not content.strip()
            )
            metadata['refusal_present'] = bool(
                getattr(message, 'refusal', None)
            )

    metadata['usage'] = _to_plain_dict(getattr(response, 'usage', None))
    return metadata


def _response_summary(metadata: dict[str, Any]) -> str:
    usage_text = _usage_summary(metadata.get('usage') or {})
    response_id = metadata.get('response_id') or 'N/A'
    finish_reason = metadata.get('finish_reason') or 'N/A'
    return (
        f'Response ID={response_id}，'
        f'Finish Reason={finish_reason}，'
        f'Token Usage={usage_text}'
    )


def _usage_summary(usage: Mapping[str, Any]) -> str:
    prompt_tokens = usage.get('prompt_tokens', 'N/A')
    completion_tokens = usage.get('completion_tokens', 'N/A')
    total_tokens = usage.get('total_tokens', 'N/A')
    parts = [
        f'prompt={prompt_tokens}',
        f'completion={completion_tokens}',
        f'total={total_tokens}',
    ]
    details = usage.get('completion_tokens_details')
    if isinstance(details, Mapping):
        reasoning_tokens = details.get('reasoning_tokens')
        if reasoning_tokens is not None:
            parts.append(f'reasoning={reasoning_tokens}')
    return ', '.join(parts)


def _usage_reasoning_tokens(usage: Mapping[str, Any]) -> Any:
    details = usage.get('completion_tokens_details')
    if isinstance(details, Mapping):
        return details.get('reasoning_tokens', 'N/A')
    return 'N/A'


def _merge_usage(
    total: Mapping[str, Any],
    usage: Mapping[str, Any],
) -> dict[str, Any]:
    merged = dict(total)
    for key, value in usage.items():
        if isinstance(value, Mapping):
            existing = merged.get(key)
            if isinstance(existing, Mapping):
                merged[key] = _merge_usage(existing, value)
            else:
                merged[key] = dict(value)
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            existing = merged.get(key, 0)
            merged[key] = (
                existing + value
                if isinstance(existing, (int, float))
                else value
            )
        elif key not in merged:
            merged[key] = value
    return merged


def _extract_validation_errors(
    exc: ValidationError,
) -> list[dict[str, Any]]:
    return [
        {
            'field': '.'.join(str(part) for part in error['loc']),
            'message': error['msg'],
            'type': error['type'],
        }
        for error in exc.errors(include_url=False)
    ]


def _format_field_errors(field_errors: list[dict[str, Any]]) -> str:
    if not field_errors:
        return '未知字段错误'
    messages = []
    for error in field_errors:
        field = error.get('field', 'unknown')
        message = error.get('message', 'unknown error')
        messages.append(f'{field}: {message}')
    return '；'.join(messages)


def _to_plain_dict(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return dict(value)

    model_dump = getattr(value, 'model_dump', None)
    if callable(model_dump):
        return model_dump(exclude_none=True)

    raw_values = getattr(value, '__dict__', None)
    if isinstance(raw_values, dict):
        return {
            key: item
            for key, item in raw_values.items()
            if not key.startswith('_')
        }
    return {}


def _safe_len(value: Any) -> Optional[int]:
    try:
        return len(value)
    except TypeError:
        return None


def _sanitize_text(value: str) -> str:
    text = str(value)
    api_key = os.getenv('DEEPSEEK_API_KEY')
    if api_key:
        text = text.replace(api_key, '***')
    return re.sub(r'sk-[A-Za-z0-9_-]{8,}', 'sk-***', text)


def _inspect_json_error(content: str) -> Optional[str]:
    try:
        json.loads(content)
    except json.JSONDecodeError as exc:
        return _format_json_decode_error(exc)
    return None


def _format_json_decode_error(exc: json.JSONDecodeError) -> str:
    return (
        f'{exc.msg}（第 {exc.lineno} 行，第 {exc.colno} 列，'
        f'位置 {exc.pos}）'
    )


def _is_retryable_api_error(exc: APIError) -> bool:
    if isinstance(exc, (APITimeoutError, APIConnectionError)):
        return True
    status_code = getattr(exc, 'status_code', None)
    if status_code in {408, 409, 429}:
        return True
    return (
        isinstance(status_code, int)
        and 500 <= status_code < 600
    )


def _is_max_tokens_rejection(exc: APIError) -> bool:
    if getattr(exc, 'status_code', None) != 400:
        return False
    message = str(exc).lower()
    if 'max_tokens' in message or 'max tokens' in message:
        return True
    return (
        'token' in message
        and ('at most' in message or 'maximum' in message)
    )


def _extract_max_tokens_limit(exc: APIError) -> Optional[int]:
    message = str(exc)
    patterns = (
        r'at most\D{0,20}(\d{3,6})',
        r'maximum\D{0,30}(\d{3,6})',
        r'supports?\D{0,40}(\d{3,6})',
        r'max_tokens\D{0,40}(\d{3,6})',
    )
    for pattern in patterns:
        match = re.search(pattern, message, re.IGNORECASE)
        if match:
            value = int(match.group(1))
            if 256 <= value <= 1_000_000:
                return value
    return None


def _build_repair_note(spec: _StageSpec, exc: Exception) -> str:
    required_fields = '、'.join(spec.schema.model_fields.keys())
    reason = {
        AIOutputTruncatedError: '输出被截断',
        AIJSONParseError: '返回内容不是合法 JSON',
        AIValidationError: '返回字段缺失或类型不符合 schema',
        AIEmptyResponseError: '返回内容为空',
    }.get(type(exc), type(exc).__name__)
    return (
        f'上一次回复未通过校验（{reason}）。本次必须缩小单次输出范围：\n'
        f'- 只返回一个完整、可解析的 JSON 对象，'
        f'字段必须且只能包含：{required_fields}。\n'
        '- 每个列表最多给出 3 条最关键的结论，优先保证 JSON 完整闭合。\n'
        '- 即使某个列表为空，也要保留字段并返回 []。\n'
        '- 不要输出 Markdown 代码块、解释文字或思维过程。\n'
    )
