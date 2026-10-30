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
    STAGE_SPECS,
    _build_subtasks,
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


def _stage_payloads():
    return [
        {
            'product_summary': '访客管理系统',
            'core_features': ['访客预约', '访客审核'],
        },
        {
            'business_rules': ['预约需审核', '黑名单人员禁止签到'],
        },
        {
            'potential_risks': ['并发审核冲突', '重复签到'],
        },
        {
            'ambiguous_requirements': ['通知方式待确认'],
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

    def test_staged_generation_merges_all_fields(self):
        analyzer, completions = _build_analyzer(
            _successful_outcomes()
        )
        events = []

        analysis = analyzer.analyze('产品文档', on_event=events.append)

        self.assertEqual(analysis.product_summary, '访客管理系统')
        self.assertEqual(analysis.core_features[0], '访客预约')
        self.assertEqual(analysis.business_rules[0], '预约需审核')
        self.assertEqual(analysis.potential_risks[0], '并发审核冲突')
        self.assertEqual(
            analysis.ambiguous_requirements[0],
            '通知方式待确认',
        )
        self.assertEqual(len(completions.requests), 6)
        expected_budgets = [
            analyzer.stage_output_budgets['generation_1'],
            analyzer.stage_output_budgets['generation_2'],
            analyzer.stage_output_budgets['generation_3'],
            analyzer.stage_output_budgets['generation_4'],
            analyzer.stage_output_budgets['generation_4'],
            analyzer.stage_output_budgets['generation_4'],
        ]
        self.assertEqual(
            [request['max_tokens'] for request in completions.requests],
            expected_budgets,
        )
        self.assertEqual(analyzer.last_trace.usage['total_tokens'], 720)
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
        self.assertEqual(len(completions.requests), 7)
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
        self.assertEqual(len(completions.requests), 9)
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
        self.assertEqual(len(completions.requests), 7)
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

        self.assertEqual(len(subtasks), 7)
        self.assertEqual(len(subtasks[0].modules), 4)
        self.assertIn('_modules1', subtasks[0].key)
        self.assertIn('_modules2', subtasks[-2].key)
        self.assertEqual(
            subtasks[-1].key,
            'generation_4_cross_module',
        )

    def test_stage4_keeps_three_subtasks_for_small_feature_sets(self):
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

        self.assertEqual(len(subtasks), 3)
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
        self.assertEqual(len(completions.requests), 7)
        self.assertEqual(
            analyzer.last_trace.stage_traces[0].attempts,
            2,
        )


if __name__ == '__main__':
    unittest.main()
