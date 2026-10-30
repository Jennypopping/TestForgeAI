import json
import unittest
from types import SimpleNamespace

import httpx
from openai import APIConnectionError, BadRequestError, OpenAI

from services.ai_analyzer import (
    AIAnalysisTrace,
    AIEmptyResponseError,
    AIJSONParseError,
    AIOutputTruncatedError,
    AIValidationError,
    AIAnalyzer,
    AnalysisItem,
    PotentialRiskItem,
    RequirementItem,
    STAGE_SPECS,
    _build_system_prompt,
    _build_subtasks,
    _dedupe_analysis_models,
    _dedupe_requirement_items,
    _dedupe_texts,
    _ground_analysis_item,
    _normalize_evidence_text,
)


def _build_response(
    content,
    finish_reason='stop',
    response_id='resp_test_001',
    prompt_tokens=100,
    completion_tokens=20,
    reasoning_tokens=5,
):
    usage = SimpleNamespace(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens,
        completion_tokens_details=SimpleNamespace(
            reasoning_tokens=reasoning_tokens
        ),
    )
    usage.model_dump = lambda exclude_none=True: {
        'prompt_tokens': prompt_tokens,
        'completion_tokens': completion_tokens,
        'total_tokens': prompt_tokens + completion_tokens,
        'completion_tokens_details': {
            'reasoning_tokens': reasoning_tokens,
        },
    }
    message = SimpleNamespace(
        content=content,
        role='assistant',
        refusal=None,
    )
    choice = SimpleNamespace(
        index=0,
        finish_reason=finish_reason,
        message=message,
    )
    return SimpleNamespace(
        id=response_id,
        model='deepseek-flash',
        object='chat.completion',
        created=1234567890,
        system_fingerprint=None,
        choices=[choice],
        usage=usage,
    )


class _FakeCompletions:

    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.requests = []

    def create(self, **kwargs):
        self.requests.append(kwargs)
        index = len(self.requests) - 1
        if index >= len(self.outcomes):
            raise AssertionError('unexpected extra API request')
        outcome = self.outcomes[index]
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def _build_analyzer(
    outcomes,
    max_retries=0,
    max_validation_retries=0,
    max_truncation_recoveries=None,
):
    completions = _FakeCompletions(outcomes)
    client = SimpleNamespace(
        chat=SimpleNamespace(completions=completions)
    )
    analyzer = AIAnalyzer(
        client=client,
        model='deepseek-flash',
        max_retries=max_retries,
        retry_backoff_seconds=0,
        max_validation_retries=max_validation_retries,
        max_truncation_recoveries=max_truncation_recoveries,
    )
    return analyzer, completions


def _business_rule(content, **overrides):
    payload = {
        'content': content,
        'evidence_type': 'explicit',
        'source_section': '3.2 创建预约',
        'source_text': '预约需审核',
        'analysis': 'PRD 明确要求该预约流程。',
    }
    payload.update(overrides)
    return payload


def _risk(content, **overrides):
    payload = {
        'content': content,
        'evidence_type': 'inferred',
        'source_section': '3.2 创建预约',
        'source_text': '预约需审核',
        'analysis': '推断依据：审核状态变化会产生并发处理需求。',
        'risk_type': '并发',
    }
    payload.update(overrides)
    return payload


def _requirement(content, priority='P1', **overrides):
    payload = {
        'content': content,
        'evidence_type': 'explicit',
        'source_section': '6 待确认问题',
        'source_text': '通知方式待确认',
        'analysis': 'PRD 明确将通知方式列为待确认事项。',
        'priority': priority,
    }
    payload.update(overrides)
    return payload


def _stage_payloads():
    return [
        {
            'product_summary': '访客管理系统',
            'core_features': ['访客预约', '访客审核'],
        },
        {
            'business_rules': [
                _business_rule('[审核][状态流转] 预约需审核 → pending'),
                _business_rule(
                    '[签到][权限] 黑名单人员禁止签到',
                    source_text='黑名单人员禁止签到',
                ),
            ],
        },
        {
            'potential_risks': [
                _risk('[审核][并发] 并发审核冲突 → 状态覆盖'),
                _risk(
                    '[签到][数据一致性] 重复签到 → 记录重复',
                    risk_type='数据一致性',
                ),
            ],
        },
        {
            'requirement_conflicts': [],
            'requirement_ambiguities': [],
            'missing_requirements': [],
            'pending_questions': [
                _requirement('[通知] 通知方式待确认；需确认：使用何种通知方式'),
            ],
        },
    ]


def _successful_outcomes(payloads=None):
    payloads = payloads or _stage_payloads()
    stage1, stage2, stage3, stage4 = payloads
    stage4_json = json.dumps(stage4, ensure_ascii=False)
    return [
        _build_response(
            json.dumps(stage1, ensure_ascii=False),
            response_id='resp_stage_1',
        ),
        _build_response(
            json.dumps(stage2, ensure_ascii=False),
            response_id='resp_stage_2',
        ),
        _build_response(
            json.dumps(stage3, ensure_ascii=False),
            response_id='resp_stage_3',
        ),
        _build_response(stage4_json, response_id='resp_stage_4a'),
        _build_response(stage4_json, response_id='resp_stage_4b'),
        _build_response(stage4_json, response_id='resp_stage_4c'),
        _build_response(stage4_json, response_id='resp_stage_4d'),
    ]


def _bad_request(message):
    return BadRequestError(
        message,
        response=httpx.Response(
            400,
            request=httpx.Request(
                'POST',
                'https://api.deepseek.com/chat/completions',
            ),
        ),
        body=None,
    )


class AIAnalyzerTests(unittest.TestCase):

    def test_stage_prompts_define_quality_contracts(self):
        specs = {spec.key: spec for spec in STAGE_SPECS}
        expected_fragments = {
            'generation_2': '[模块][规则类型]',
            'generation_3': '[模块][风险类型]',
            'generation_4': 'requirement_conflicts',
        }

        for stage_key, fragment in expected_fragments.items():
            prompt = _build_system_prompt(specs[stage_key])
            self.assertIn('输出契约', prompt)
            self.assertIn(fragment, prompt)
            self.assertIn('合并语义相同', prompt)
            self.assertIn('推断依据', prompt)
        self.assertIn(
            '并列动词',
            _build_system_prompt(specs['generation_2']),
        )

    def test_merge_normalizes_and_deduplicates_analysis_items(self):
        analyzer, _ = _build_analyzer([])
        payloads = _stage_payloads()
        payloads[1] = {
            'business_rules': [
                _business_rule(
                    '[审核][状态流转] 开启 → pending',
                    analysis='开启审核后进入 pending。',
                ),
                _business_rule(
                    '[审核][状态流转] 开启审核 → 预约进入 pending',
                    analysis='开启审核后进入 pending 状态，需被访人审核。',
                ),
                _business_rule(
                    '[审核][状态流转] 关闭 → scheduled',
                    source_text='关闭时自动通过',
                ),
            ],
        }
        payloads[2] = {
            'potential_risks': [
                _risk(
                    '[审核][状态流转] 开启 → pending',
                    risk_type='状态流转',
                ),
                _risk(
                    '[审核][权限安全] 越权审核 → 数据泄露',
                    risk_type='权限安全',
                ),
            ],
        }
        payloads[3] = {
            'requirement_conflicts': [],
            'requirement_ambiguities': [],
            'missing_requirements': [],
            'pending_questions': [],
        }
        stage_payloads = {
            'generation_1': payloads[0],
            'generation_2': payloads[1],
            'generation_3': payloads[2],
            'generation_4': payloads[3],
        }
        trace = AIAnalysisTrace(model='deepseek-flash')

        analysis = analyzer._merge_stage_results(stage_payloads, trace)

        self.assertEqual(len(analysis.business_rules), 2)
        self.assertEqual(
            analysis.business_rules[0].content,
            '[审核][状态流转] 开启审核 → 预约进入 pending',
        )
        self.assertIn('被访人审核', analysis.business_rules[0].analysis)
        self.assertEqual(
            analysis.business_rules[1].content,
            '[审核][状态流转] 关闭 → scheduled',
        )
        self.assertEqual(len(analysis.potential_risks), 2)

    def test_dedupe_texts_strips_list_markers_and_whitespace(self):
        self.assertEqual(
            _dedupe_texts([
                '- [A]  第一条',
                '1. [A] 第一条',
                ' [B] 第二条 ',
            ]),
            ['[A] 第一条', '[B] 第二条'],
        )

    def test_evidence_schema_requires_source_or_inference_basis(self):
        with self.assertRaises(ValueError):
            AnalysisItem(
                content='规则',
                evidence_type='explicit',
                source_text='',
                analysis='明确规则',
            )
        with self.assertRaises(ValueError):
            AnalysisItem(
                content='风险',
                evidence_type='inferred',
                source_text='原文',
                analysis='没有说明依据',
            )

    def test_fake_explicit_citation_is_downgraded(self):
        item = AnalysisItem(
            content='[预约] 预约时间不能重叠',
            evidence_type='explicit',
            source_section='3.2 创建预约',
            source_text='PRD 中不存在这句话',
            analysis='原文明确说明不能重叠。',
        )
        grounded = _ground_analysis_item(
            item,
            _normalize_evidence_text(
                '同一会议室的预约时间不能重叠。'
            ),
        )

        self.assertEqual(grounded.evidence_type, 'inferred')
        self.assertEqual(grounded.source_text, '')
        self.assertIn('推断依据', grounded.analysis)

    def test_grounding_uses_actual_prd_section_titles(self):
        document = (
            '# 产品需求文档\n\n'
            '## 3.2 创建预约\n\n'
            '- 同一会议室的预约时间不能重叠。\n\n'
            '## 6. 待确认问题\n\n'
            '- Q1：预约时间是否允许精确到分钟？\n'
        )
        item = AnalysisItem(
            content='[创建预约][校验] 预约时间不能重叠。',
            evidence_type='explicit',
            source_section='3.2 会议室预约（模型自拟标题）',
            source_text='同一会议室的预约时间不能重叠。',
            analysis='原文明确要求预约时间不能重叠。',
        )

        grounded = _ground_analysis_item(
            item,
            _normalize_evidence_text(document),
            document,
        )

        self.assertEqual(grounded.source_section, '3.2 创建预约')

    def test_grounding_keeps_all_sections_for_composite_evidence(self):
        document = (
            '# 产品需求文档\n\n'
            '## 3.5 会议室管理\n\n'
            '- 停用会议室前需要检查是否存在未结束的预约。\n\n'
            '## 6. 待确认问题\n\n'
            '- Q4：管理员停用存在未来预约的会议室时，是否需要自动取消这些预约？\n'
        )
        item = AnalysisItem(
            content='[会议室管理] 停用会议室时未来预约的处理方式未定。',
            evidence_type='explicit',
            source_section='3.5',
            source_text=(
                '停用会议室前需要检查是否存在未结束的预约。｜'
                'Q4：管理员停用存在未来预约的会议室时，是否需要自动取消这些预约？'
            ),
            analysis='两段原文共同说明停用流程存在待确认规则。',
        )

        grounded = _ground_analysis_item(
            item,
            _normalize_evidence_text(document),
            document,
        )

        self.assertEqual(
            grounded.source_section,
            '3.5 会议室管理；6. 待确认问题',
        )

    def test_semantic_dedupe_keeps_more_complete_item(self):
        shorter = AnalysisItem(
            content='[审核] 开启审核后预约进入待审核状态',
            evidence_type='explicit',
            source_section='3.2',
            source_text='开启审核后进入待审核状态',
            analysis='原文规定了状态。',
        )
        complete = AnalysisItem(
            content='[审核] 开启审核后预约进入 pending，关闭后自动通过',
            evidence_type='explicit',
            source_section='3.2 创建预约',
            source_text='开启审核后进入待审核状态',
            analysis='原文明确说明开启与关闭审核时的状态流转。',
        )

        merged = _dedupe_analysis_models([shorter, complete])

        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0].content, complete.content)
        self.assertEqual(merged[0].analysis, complete.analysis)

    def test_requirement_categories_are_mutually_exclusive(self):
        conflict_item = RequirementItem(
            content='[预约] 同一会议室的预约是否允许重叠',
            evidence_type='explicit',
            source_section='3.2 创建预约',
            source_text='同一会议室的预约时间不能重叠',
            analysis='该问题同时出现在冲突与歧义列表中。',
            priority='P0',
        )
        ambiguity_item = RequirementItem(
            content='[预约] 同一会议室预约时间能否重叠',
            evidence_type='explicit',
            source_section='3.2 创建预约',
            source_text='同一会议室的预约时间不能重叠',
            analysis='应归入最合适的唯一分类。',
            priority='P1',
        )

        merged = _dedupe_requirement_items({
            'requirement_conflicts': [conflict_item],
            'requirement_ambiguities': [ambiguity_item],
            'missing_requirements': [],
            'pending_questions': [],
        })

        total = sum(len(items) for items in merged.values())
        self.assertEqual(total, 1)
        self.assertEqual(len(merged['requirement_conflicts']), 1)

    def test_requirement_same_source_rewording_is_merged(self):
        first = RequirementItem(
            content='[创建预约] Q1 未确认预约时间精度，需确认精确到分钟还是整点/半点。',
            evidence_type='explicit',
            source_section='6 待确认问题',
            source_text='Q1：预约时间是否允许精确到分钟，还是只能选择整点或半点？',
            analysis='PRD 已明确列为待确认问题。',
            priority='P1',
        )
        second = RequirementItem(
            content='预约时间是否允许精确到分钟，还是只能选择整点或半点？需确认时间粒度。',
            evidence_type='explicit',
            source_section='6 待确认问题',
            source_text='Q1：预约时间是否允许精确到分钟，还是只能选择整点或半点？',
            analysis='PRD 已明确列为待确认问题，影响重叠判断。',
            priority='P1',
        )

        merged = _dedupe_requirement_items({
            'requirement_conflicts': [],
            'requirement_ambiguities': [],
            'missing_requirements': [],
            'pending_questions': [first, second],
        })

        self.assertEqual(len(merged['pending_questions']), 1)

    def test_requirement_same_source_distinct_focus_is_kept(self):
        name_scope = RequirementItem(
            content=(
                '“会议室名称不能重复”是否包含已停用会议室存在两种解释：'
                '全局唯一（含停用）或仅启用会议室唯一（停用后可复用名称）；'
                '需确认：停用会议室是否占用名称唯一性。'
            ),
            evidence_type='explicit',
            source_section='3.5 会议室管理',
            source_text='会议室名称不能重复。',
            analysis='需要确认停用会议室是否占用名称。',
            priority='P1',
        )
        edit_self = RequirementItem(
            content=(
                '编辑会议室时名称唯一性校验是否排除当前会议室自身未定义；'
                '需确认：仅修改其他信息而保留原名称是否允许保存。'
            ),
            evidence_type='inferred',
            source_section='3.5 会议室管理',
            source_text='会议室名称不能重复。',
            analysis='推断依据：编辑场景通常需要排除当前记录自身。',
            priority='P2',
        )

        merged = _dedupe_requirement_items({
            'requirement_conflicts': [],
            'requirement_ambiguities': [name_scope],
            'missing_requirements': [edit_self],
            'pending_questions': [],
        })

        total = sum(len(items) for items in merged.values())
        self.assertEqual(total, 2)

    def test_dedupe_keeps_distinct_roles_actions_and_states(self):
        source_text = '管理员可以新增、编辑和停用会议室。'
        items = [
            AnalysisItem(
                content='[会议室管理][权限] 管理员新增会议室 → 允许新增。',
                evidence_type='explicit',
                source_section='3.5 会议室管理',
                source_text=source_text,
                analysis='原文明确允许新增。',
            ),
            AnalysisItem(
                content='[会议室管理][权限] 管理员编辑会议室 → 允许编辑。',
                evidence_type='explicit',
                source_section='3.5 会议室管理',
                source_text=source_text,
                analysis='原文明确允许编辑。',
            ),
            AnalysisItem(
                content='[会议室管理][权限] 管理员停用会议室 → 允许停用。',
                evidence_type='explicit',
                source_section='3.5 会议室管理',
                source_text=source_text,
                analysis='原文明确允许停用。',
            ),
            AnalysisItem(
                content='[取消预约][权限] 员工取消预约 → 仅可取消自己的预约。',
                evidence_type='explicit',
                source_section='3.3 取消预约',
                source_text='员工可以取消自己创建的预约。',
                analysis='原文明确员工取消权限。',
            ),
            AnalysisItem(
                content='[取消预约][权限] 管理员取消预约 → 可取消任意预约。',
                evidence_type='explicit',
                source_section='3.3 取消预约',
                source_text='管理员可以取消任意预约。',
                analysis='原文明确管理员取消权限。',
            ),
        ]

        deduped = _dedupe_analysis_models(items)

        self.assertEqual(len(deduped), 5)

    def test_staged_generation_merges_all_fields(self):
        analyzer, completions = _build_analyzer(
            _successful_outcomes()
        )
        events = []

        analysis = analyzer.analyze('产品文档', on_event=events.append)

        self.assertEqual(analysis.product_summary, '访客管理系统')
        self.assertEqual(analysis.core_features[0], '访客预约')
        self.assertEqual(
            analysis.business_rules[0].content,
            '[审核][状态流转] 预约需审核 → pending',
        )
        self.assertEqual(
            analysis.potential_risks[0].content,
            '[审核][并发] 并发审核冲突 → 状态覆盖',
        )
        self.assertEqual(len(analysis.pending_questions), 1)
        self.assertIn(
            '通知方式待确认',
            analysis.pending_questions[0].content,
        )
        self.assertEqual(len(completions.requests), 7)
        expected_budgets = [
            analyzer.stage_output_budgets['generation_1'],
            analyzer.stage_output_budgets['generation_2'],
            analyzer.stage_output_budgets['generation_3'],
            analyzer.stage_output_budgets['generation_4'],
            analyzer.stage_output_budgets['generation_4'],
            analyzer.stage_output_budgets['generation_4'],
            analyzer.stage_output_budgets['generation_4'],
        ]
        self.assertEqual(
            [request['max_tokens'] for request in completions.requests],
            expected_budgets,
        )
        self.assertEqual(analyzer.last_trace.usage['total_tokens'], 840)
        self.assertEqual(
            len(analyzer.last_trace.stage_traces),
            4,
        )
        self.assertTrue(
            all(
                stage.status == 'success'
                for stage in analyzer.last_trace.stage_traces
            )
        )
        generation_events = [
            event for event in events
            if (
                event.stage.startswith('generation_')
                and event.state == 'success'
            )
        ]
        self.assertEqual(len(generation_events), 4)
        self.assertEqual(events[-1].stage, 'final')
        self.assertEqual(events[-1].state, 'success')

    def test_stage_budgets_vary_and_thinking_stays_enabled(self):
        analyzer, completions = _build_analyzer(
            _successful_outcomes()
        )

        analyzer.analyze('product document')

        budgets = analyzer.stage_output_budgets
        self.assertLess(
            budgets['generation_1'],
            budgets['generation_4'],
        )
        self.assertEqual(
            budgets['generation_4'],
            max(budgets.values()),
        )
        for request in completions.requests:
            self.assertNotIn('extra_body', request)
            self.assertNotIn('reasoning_effort', request)
            self.assertEqual(
                request['response_format'],
                {'type': 'json_object'},
            )
            self.assertEqual(request['model'], 'deepseek-flash')
        for stage_trace in analyzer.last_trace.stage_traces:
            self.assertEqual(
                stage_trace.request_thinking_config['mode'],
                'model_default',
            )
            self.assertTrue(
                stage_trace.request_thinking_config['reasoning_enabled']
            )
            self.assertEqual(
                stage_trace.request_max_tokens,
                analyzer.stage_output_budgets[stage_trace.key],
            )

    def test_output_limit_is_capped_by_model_registry(self):
        completions = _FakeCompletions([])
        client = SimpleNamespace(
            chat=SimpleNamespace(completions=completions)
        )
        analyzer = AIAnalyzer(
            client=client,
            model='deepseek-flash',
            max_output_tokens=999999,
        )

        self.assertEqual(analyzer.model_output_limit, 384 * 1024)
        self.assertEqual(analyzer.max_output_tokens, 384 * 1024)

    def test_truncation_grows_budget_before_giving_up(self):
        partial_json = '{product_summary: 概述, core_features: [A'
        truncated_response = _build_response(
            partial_json,
            finish_reason='length',
            completion_tokens=10240,
            reasoning_tokens=9000,
        )
        analyzer, completions = _build_analyzer(
            [truncated_response, truncated_response],
            max_truncation_recoveries=1,
        )

        with self.assertRaises(AIOutputTruncatedError) as context:
            analyzer.analyze('产品文档')

        self.assertEqual(len(completions.requests), 2)
        self.assertGreater(
            completions.requests[1]['max_tokens'],
            completions.requests[0]['max_tokens'],
        )
        stage_trace = analyzer.last_trace.stage_traces[0]
        self.assertTrue(stage_trace.truncated)
        self.assertTrue(analyzer.last_trace.truncated)
        self.assertEqual(stage_trace.finish_reason, 'length')
        self.assertEqual(
            stage_trace.usage['completion_tokens_details'][
                'reasoning_tokens'
            ],
            18000,
        )
        self.assertIsNotNone(stage_trace.json_error)
        self.assertEqual(
            context.exception.error_code,
            'ai_output_truncated',
        )

    def test_invalid_json_retries_with_narrower_scope(self):
        outcomes = [_build_response('{invalid json')]
        outcomes.extend(_successful_outcomes())
        analyzer, completions = _build_analyzer(
            outcomes,
            max_validation_retries=1,
        )

        analysis = analyzer.analyze('产品文档')

        self.assertEqual(analysis.product_summary, '访客管理系统')
        self.assertEqual(len(completions.requests), 8)
        self.assertIn(
            '缩小单次输出范围',
            completions.requests[1]['messages'][0]['content'],
        )
        stage_trace = analyzer.last_trace.stage_traces[0]
        self.assertEqual(stage_trace.attempts, 2)
        self.assertEqual(stage_trace.retry_count, 1)
        self.assertEqual(stage_trace.status, 'success')

    def test_empty_truncated_response_is_treated_as_truncation(self):
        truncated = _build_response(
            None,
            finish_reason='length',
            completion_tokens=10240,
            reasoning_tokens=10000,
        )
        analyzer, completions = _build_analyzer(
            [truncated, truncated],
            max_truncation_recoveries=1,
        )

        with self.assertRaises(AIOutputTruncatedError):
            analyzer.analyze('产品文档')

        self.assertEqual(len(completions.requests), 2)
        self.assertGreater(
            completions.requests[1]['max_tokens'],
            completions.requests[0]['max_tokens'],
        )

    def test_invalid_json_splits_scope_after_retry_fails(self):
        document = (
            '# 模块A\n' + '访客预约流程说明。' * 120
            + '\n\n# 模块B\n' + '访客审核流程说明。' * 120
        )
        payload_a = {
            'product_summary': '访客管理系统',
            'core_features': ['访客预约'],
        }
        payload_b = {
            'product_summary': '访客管理系统',
            'core_features': ['访客审核'],
        }
        outcomes = [
            _build_response('{invalid json'),
            _build_response('{invalid json'),
            _build_response(json.dumps(payload_a, ensure_ascii=False)),
            _build_response(json.dumps(payload_b, ensure_ascii=False)),
        ]
        outcomes.extend(_successful_outcomes()[1:])
        analyzer, completions = _build_analyzer(
            outcomes,
            max_validation_retries=1,
        )

        analysis = analyzer.analyze(document)

        self.assertEqual(
            analysis.core_features,
            ['访客预约', '访客审核'],
        )
        self.assertEqual(len(completions.requests), 10)
        self.assertNotEqual(
            completions.requests[2]['messages'][1]['content'],
            completions.requests[3]['messages'][1]['content'],
        )

    def test_max_tokens_rejection_halves_budget(self):
        outcomes = [_bad_request('max_tokens is too large')]
        outcomes.extend(_successful_outcomes())
        analyzer, completions = _build_analyzer(outcomes)

        analysis = analyzer.analyze('产品文档')

        self.assertEqual(analysis.product_summary, '访客管理系统')
        self.assertEqual(len(completions.requests), 8)
        self.assertLess(
            completions.requests[1]['max_tokens'],
            completions.requests[0]['max_tokens'],
        )

    def test_max_tokens_rejection_uses_reported_limit(self):
        outcomes = [
            _bad_request(
                'This model supports at most 8192 completion tokens'
            )
        ]
        outcomes.extend(_successful_outcomes())
        analyzer, completions = _build_analyzer(outcomes)

        analyzer.analyze('产品文档')

        self.assertEqual(completions.requests[0]['max_tokens'], 10240)
        self.assertEqual(completions.requests[1]['max_tokens'], 8192)

    def test_stage4_splits_by_modules_for_large_feature_sets(self):
        spec = next(
            item for item in STAGE_SPECS
            if item.key == 'generation_4'
        )
        features = [f'模块{i}' for i in range(1, 9)]
        subtasks = _build_subtasks(
            spec=spec,
            completed_stage_payloads={
                'generation_1': {'core_features': features},
            },
            document_content='产品文档',
        )

        self.assertEqual(len(subtasks), 9)
        self.assertEqual(len(subtasks[0].modules), 4)
        self.assertIn('_modules1', subtasks[0].key)
        self.assertIn('_modules2', subtasks[-2].key)
        self.assertEqual(
            subtasks[-1].key,
            'generation_4_cross_module',
        )

    def test_stage4_keeps_four_subtasks_for_small_feature_sets(self):
        spec = next(
            item for item in STAGE_SPECS
            if item.key == 'generation_4'
        )
        subtasks = _build_subtasks(
            spec=spec,
            completed_stage_payloads={
                'generation_1': {
                    'core_features': ['访客预约', '访客审核'],
                },
            },
            document_content='产品文档',
        )

        self.assertEqual(len(subtasks), 4)
        self.assertEqual(
            subtasks[0].modules,
            ('访客预约', '访客审核'),
        )

    def test_stage_failure_stops_following_stages(self):
        outcomes = [
            _successful_outcomes()[0],
            _build_response(None, response_id='resp_stage_2'),
        ]
        analyzer, completions = _build_analyzer(
            outcomes,
            max_retries=2,
            max_truncation_recoveries=0,
        )

        with self.assertRaises(AIEmptyResponseError):
            analyzer.analyze('产品文档')

        self.assertEqual(len(completions.requests), 2)
        self.assertEqual(
            analyzer.last_trace.stage_traces[0].status,
            'success',
        )
        self.assertEqual(
            analyzer.last_trace.stage_traces[1].status,
            'error',
        )
        self.assertEqual(
            analyzer.last_trace.stage_traces[2].status,
            'pending',
        )

    def test_json_format_error_preserves_raw_content_and_position(self):
        analyzer, completions = _build_analyzer(
            [_build_response('{invalid json')],
            max_retries=2,
        )

        with self.assertRaises(AIJSONParseError) as context:
            analyzer.analyze('产品文档')

        self.assertEqual(len(completions.requests), 1)
        self.assertIsInstance(
            context.exception.__cause__,
            json.JSONDecodeError,
        )
        stage_trace = analyzer.last_trace.stage_traces[0]
        self.assertEqual(stage_trace.raw_content, '{invalid json')
        self.assertIsNotNone(stage_trace.json_error)
        self.assertIn('第 1 行', stage_trace.json_error)

    def test_final_pydantic_validation_failure(self):
        analyzer, _ = _build_analyzer([])
        payloads = _stage_payloads()
        payloads[0]['product_summary'] = None
        stage_payloads = {
            'generation_1': payloads[0],
            'generation_2': payloads[1],
            'generation_3': payloads[2],
            'generation_4': payloads[3],
        }
        trace = AIAnalysisTrace(model='deepseek-flash')

        with self.assertRaises(AIValidationError) as context:
            analyzer._merge_stage_results(stage_payloads, trace)

        self.assertTrue(trace.validation_field_errors)
        self.assertEqual(context.exception.stage, 'final')

    def test_transient_api_error_is_retried(self):
        connection_error = APIConnectionError(
            request=httpx.Request(
                'POST',
                'https://api.deepseek.com/chat/completions',
            )
        )
        outcomes = [connection_error] + _successful_outcomes()
        analyzer, completions = _build_analyzer(
            outcomes,
            max_retries=1,
        )

        analysis = analyzer.analyze('产品文档')

        self.assertEqual(analysis.product_summary, '访客管理系统')
        self.assertEqual(len(completions.requests), 8)
        self.assertEqual(
            analyzer.last_trace.stage_traces[0].attempts,
            2,
        )


if __name__ == '__main__':
    unittest.main()
