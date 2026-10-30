from __future__ import annotations

import os
import re
from typing import Any, Optional

import streamlit as st

from services.ai_analyzer import (
    AIAnalysisTrace,
    AIStageTrace,
    AnalysisItem,
    AnalysisEvent,
    PotentialRiskItem,
    ProductAnalysis,
    RequirementItem,
)

STAGE_TITLES = {
    'document_parse': '文档解析',
    'prompt': 'Prompt 构建',
    'generation_1': '阶段 1 · 产品概述与核心功能',
    'generation_2': '阶段 2 · 业务规则',
    'generation_3': '阶段 3 · 潜在风险',
    'generation_4': '阶段 4 · 冲突、歧义、缺失与待确认问题',
    'request': 'DeepSeek API 请求',
    'response': '模型响应',
    'json_parse': 'JSON 解析',
    'validation': 'Pydantic 校验',
    'final': '最终结果',
}

STATE_ICONS = {
    'pending': '⚪',
    'running': '⏳',
    'success': '✅',
    'error': '❌',
}


class AnalysisProgress:

    def __init__(self, status_container: Any) -> None:
        self._status = status_container
        self._slots: dict[str, Any] = {}

    def update(self, stage: str, state: str, message: str) -> None:
        slot = self._slots.get(stage)
        if slot is None:
            slot = self._status.empty()
            self._slots[stage] = slot

        title = STAGE_TITLES.get(stage, stage)
        icon = STATE_ICONS.get(state, '•')
        slot.markdown(f'{icon} **{title}** · {message}')

    def on_event(self, event: AnalysisEvent) -> None:
        self.update(event.stage, event.state, event.message)


def debug_panel_enabled() -> bool:
    environment = (os.getenv('APP_ENV') or 'production').strip().lower()
    debug_flag = (os.getenv('TESTFORGE_DEBUG') or '').strip().lower()
    return (
        debug_flag in {'1', 'true', 'yes', 'on'}
        and environment in {'local', 'dev', 'development', 'test'}
    )


def render_analysis_report(analysis: ProductAnalysis) -> None:
    st.divider()
    st.subheader('AI 产品分析报告')
    st.download_button(
        label='下载 AI 分析结果 JSON',
        data=analysis.model_dump_json(indent=2),
        file_name='product_analysis.json',
        mime='application/json',
        on_click='ignore',
    )

    st.markdown('### 产品概述')
    st.write(analysis.product_summary)

    st.markdown('### 核心功能模块')
    for item in analysis.core_features:
        st.write(f'- {item}')

    st.markdown(f'### 业务规则（{len(analysis.business_rules)} 条）')
    _render_analysis_items(
        analysis.business_rules,
        empty_text='未提取到有明确依据的业务规则。',
    )

    st.markdown(f'### 潜在风险（{len(analysis.potential_risks)} 条）')
    _render_analysis_items(
        analysis.potential_risks,
        empty_text='未发现可验证的潜在风险。',
        show_risk_type=True,
    )

    st.markdown(
        f'### 需求冲突（{len(analysis.requirement_conflicts)} 条）'
    )
    _render_analysis_items(
        analysis.requirement_conflicts,
        empty_text='未发现 PRD 原文之间存在明确矛盾的描述。',
        show_priority=True,
    )

    st.markdown(
        f'### 需求歧义（{len(analysis.requirement_ambiguities)} 条）'
    )
    _render_analysis_items(
        analysis.requirement_ambiguities,
        empty_text='未发现存在多种实质理解的需求描述。',
        show_priority=True,
    )

    st.markdown(
        f'### 需求缺失（{len(analysis.missing_requirements)} 条）'
    )
    _render_analysis_items(
        analysis.missing_requirements,
        empty_text='未发现影响实现或测试的需求缺失。',
        show_priority=True,
    )

    st.markdown(
        f'### 待确认问题（{len(analysis.pending_questions)} 条）'
    )
    _render_analysis_items(
        analysis.pending_questions,
        empty_text='当前没有需要产品或业务人员进一步确认的问题。',
        show_priority=True,
    )


def _render_analysis_items(
    items: list[AnalysisItem],
    *,
    empty_text: str,
    show_risk_type: bool = False,
    show_priority: bool = False,
) -> None:
    if not items:
        st.caption(empty_text)
        return

    for index, item in enumerate(items, start=1):
        _render_analysis_item(
            index,
            item,
            show_risk_type=show_risk_type,
            show_priority=show_priority,
        )


def _render_analysis_item(
    index: int,
    item: AnalysisItem,
    *,
    show_risk_type: bool,
    show_priority: bool,
) -> None:
    st.markdown(f'**{index}. {item.content}**')

    labels = [f'证据类型：`{item.evidence_type}`']
    if show_risk_type:
        risk_type = getattr(item, 'risk_type', '')
        if risk_type:
            labels.append(f'风险类型：`{risk_type}`')
    if show_priority and isinstance(item, RequirementItem):
        if item.requirement_id:
            labels.append(f'问题 ID：`{item.requirement_id}`')
        if item.question_id:
            labels.append(f'原始问题：`{item.question_id}`')
        labels.append(f'优先级：`{item.priority}`')
    if item.related_question_id:
        labels.append(f'关联问题：`{item.related_question_id}`')
    if item.related_requirement_id:
        labels.append(f'关联需求问题：`{item.related_requirement_id}`')
    if isinstance(item, PotentialRiskItem) and item.needs_product_confirmation:
        labels.append('预期行为待产品确认')
    labels.append(
        f'来源章节：{item.source_section or "未定位"}'
    )
    st.caption(' · '.join(labels))

    with st.expander('查看原文依据', expanded=False):
        source_parts = [
            part.strip()
            for part in re.split(r'\s*(?:｜|\|\||\n+)\s*', item.source_text)
            if part.strip()
        ]
        if source_parts:
            for part in source_parts:
                st.markdown(f'> {part}')
        else:
            st.caption(
                'PRD 中未定位到可逐字引用的原文。'
                '该项应结合 AI 分析说明和人工复核判断。'
            )

    with st.expander('查看 AI 分析说明', expanded=False):
        st.write(item.analysis)

    if isinstance(item, PotentialRiskItem):
        st.markdown('**测试建议**')
        st.write(item.test_suggestion)
        st.markdown('**预期行为**')
        if item.expected_behavior:
            st.write(item.expected_behavior)
        else:
            st.caption('PRD 未定义该风险的明确预期行为，需要产品确认。')
    st.markdown('')


def render_debug_panel(
    trace: Optional[AIAnalysisTrace],
    exception_traceback: Optional[str] = None,
) -> None:
    if trace is None or not debug_panel_enabled():
        return

    with st.expander('查看 AI 调试详情', expanded=False):
        st.warning(
            '调试详情仅限本地开发环境使用，'
            '其中可能包含上传文档内容和模型原始输出。'
        )

        elapsed_text = _format_elapsed(trace.request_elapsed_seconds)
        st.markdown(f'**实际请求模型**：`{trace.model}`')
        st.markdown(
            f'**模型输出上限**：`{trace.model_output_limit}` tokens；'
            f'**本次 max_tokens**：`{trace.max_output_tokens}`'
        )
        st.markdown(f'**总请求耗时**：{elapsed_text}')
        st.markdown(f'**累计 Token Usage**：`{trace.usage}`')
        st.markdown(
            f'**Prompt 总长度**：{trace.prompt_char_count:,} 字符 '
            f'（文档 {trace.document_char_count:,} 字符）'
        )
        truncated_text = '是' if trace.truncated else '否'
        st.markdown(
            f'**是否发生截断**：'
            f'`{truncated_text}`'
        )

        for index, stage_trace in enumerate(trace.stage_traces, start=1):
            _render_stage_debug(index, stage_trace)

        if trace.validation_error:
            st.markdown('#### 最终 Pydantic ValidationError')
            st.code(trace.validation_error)
            if trace.validation_field_errors:
                st.json(trace.validation_field_errors)

        st.markdown('#### 最终异常')
        if trace.exception_type:
            exception_message = trace.exception_message or '无消息'
            st.code(
                f'{trace.exception_type}: '
                f'{exception_message}'
            )
        else:
            st.code('无')

        st.markdown('#### 完整异常堆栈')
        st.code(
            exception_traceback
            or trace.exception_traceback
            or '无'
        )


def _render_stage_debug(index: int, stage_trace: AIStageTrace) -> None:
    icon = STATE_ICONS.get(stage_trace.status, '•')
    title = f'{icon} {index}. {stage_trace.label}'
    expanded = stage_trace.status == 'error'

    with st.expander(title, expanded=expanded):
        finish_reason = stage_trace.finish_reason or 'N/A'
        response_id = stage_trace.response_id or 'N/A'
        st.markdown(
            f'**Request thinking config**: '
            f'`{stage_trace.request_thinking_config}`'
        )
        st.markdown(
            f'**Request reasoning_effort**: '
            f'`{stage_trace.request_reasoning_effort}`'
        )
        st.markdown(
            f'**Request max_tokens**: '
            f'`{stage_trace.request_max_tokens}`'
        )
        truncated_text = '是' if stage_trace.truncated else '否'
        st.markdown(f'**状态**：`{stage_trace.status}`')
        st.markdown(
            f'**耗时**：{_format_elapsed(stage_trace.request_elapsed_seconds)}'
        )
        st.markdown(f'**请求次数**：{stage_trace.attempts}')
        st.markdown(f'**重试次数**：{stage_trace.retry_count}')
        st.markdown(
            f'**子任务**：{stage_trace.completed_subtask_count}/'
            f'{stage_trace.subtask_count}'
        )
        st.markdown(
            f'**Prompt 长度**：{stage_trace.prompt_char_count:,} 字符'
        )
        st.markdown(
            f'**Token Usage**：`{stage_trace.usage}`'
        )
        prompt_tokens = stage_trace.usage.get('prompt_tokens', 'N/A')
        completion_tokens = stage_trace.usage.get(
            'completion_tokens',
            'N/A',
        )
        reasoning_tokens = _reasoning_tokens(stage_trace.usage)
        st.markdown(f'**Prompt Tokens**：{prompt_tokens}')
        st.markdown(f'**Completion Tokens**：{completion_tokens}')
        st.markdown(f'**Reasoning Tokens**：{reasoning_tokens}')
        st.markdown(
            f'**Finish Reason**：`{finish_reason}`'
        )
        st.markdown(
            f'**输出截断**：'
            f'`{truncated_text}`'
        )
        st.markdown(
            f'**输出字符数**：'
            f'{stage_trace.output_char_count or 0:,}'
        )
        st.markdown(
            f'**Response ID**：'
            f'`{response_id}`'
        )

        st.markdown('##### Prompt')
        st.text_area(
            'System Prompt',
            value=stage_trace.system_prompt,
            height=140,
            disabled=True,
            key=f'debug_{stage_trace.key}_system',
        )
        st.text_area(
            'User Prompt',
            value=stage_trace.user_prompt,
            height=180,
            disabled=True,
            key=f'debug_{stage_trace.key}_user',
        )

        st.markdown('##### 模型原始返回')
        st.code(stage_trace.raw_content or '（空）', language='json')

        st.markdown('##### 响应元数据')
        st.json(stage_trace.response_metadata)

        if stage_trace.json_error:
            st.markdown('##### JSON 解析错误')
            st.code(stage_trace.json_error)

        if stage_trace.validation_error:
            st.markdown('##### Pydantic ValidationError')
            st.code(stage_trace.validation_error)
            if stage_trace.validation_field_errors:
                st.json(stage_trace.validation_field_errors)

        if stage_trace.exception_type:
            stage_exception_message = (
                stage_trace.exception_message or '无消息'
            )
            st.markdown('##### 阶段异常')
            st.code(
                f'{stage_trace.exception_type}: '
                f'{stage_exception_message}'
            )
            if stage_trace.exception_traceback:
                with st.expander('阶段异常堆栈', expanded=False):
                    st.code(stage_trace.exception_traceback)


def _format_elapsed(value: Optional[float]) -> str:
    if value is None:
        return 'N/A'
    return f'{value:.3f} 秒'


def _reasoning_tokens(usage: Any) -> Any:
    if not isinstance(usage, dict):
        return 'N/A'
    details = usage.get('completion_tokens_details')
    if not isinstance(details, dict):
        return 'N/A'
    return details.get('reasoning_tokens', 'N/A')
