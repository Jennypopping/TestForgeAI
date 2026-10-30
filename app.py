import hashlib
import logging
import os
import tempfile
import time
import traceback

import streamlit as st
from dotenv import load_dotenv

from services.ai_analyzer import AIAnalysisError, AIAnalyzer
from services.analysis_ui import (
    AnalysisProgress,
    render_analysis_report,
    render_debug_panel,
)
from services.document_parser import DocumentParser

load_dotenv()

logging.basicConfig(
    level=getattr(
        logging,
        os.getenv('LOG_LEVEL', 'INFO').upper(),
        logging.INFO,
    ),
    format='%(asctime)s %(levelname)s %(name)s - %(message)s',
)

st.set_page_config(
    page_title='TestForge AI',
    page_icon='🧪',
    layout='wide',
)

st.title('🧪 TestForge AI')
st.caption('AI-Powered Requirement Analysis & Test Generation')

st.divider()

st.subheader('产品文档智能分析')


ANALYSIS_RESULT_STATE_KEY = 'analysis_result'
ANALYSIS_TRACE_STATE_KEY = 'analysis_trace'
ANALYSIS_FILE_STATE_KEY = 'analysis_file_signature'


def _uploaded_file_signature(uploaded_file):
    if uploaded_file is None:
        return None

    content_digest = hashlib.sha256(uploaded_file.getvalue()).hexdigest()
    return f'{uploaded_file.name}:{uploaded_file.size}:{content_digest}'


def _clear_analysis_state():
    for key in (
        ANALYSIS_RESULT_STATE_KEY,
        ANALYSIS_TRACE_STATE_KEY,
        ANALYSIS_FILE_STATE_KEY,
    ):
        st.session_state.pop(key, None)


uploaded_file = st.file_uploader(
    '上传产品需求文档',
    type=['pdf', 'docx', 'md', 'txt'],
)

if uploaded_file:

    st.info(f'当前文件：{uploaded_file.name}')

    current_file_signature = _uploaded_file_signature(uploaded_file)
    stored_file_signature = st.session_state.get(ANALYSIS_FILE_STATE_KEY)
    if (
        stored_file_signature is not None
        and stored_file_signature != current_file_signature
    ):
        _clear_analysis_state()

    if st.button('开始 AI 分析', type='primary'):

        if not os.getenv('DEEPSEEK_API_KEY'):
            st.error('请先配置 DeepSeek API Key')
            st.stop()

        temp_path = None
        analyzer = None
        status = st.status('AI 分析进行中...', expanded=True)
        progress = AnalysisProgress(status)

        try:
            suffix = os.path.splitext(uploaded_file.name)[1]

            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=suffix,
            ) as temp_file:
                temp_file.write(uploaded_file.getvalue())
                temp_path = temp_file.name

            progress.update(
                'document_parse',
                'running',
                f'正在解析 {uploaded_file.name}...',
            )
            parse_started = time.perf_counter()
            try:
                parser = DocumentParser()
                document_content = parser.parse(temp_path)
            except Exception:
                parse_elapsed = time.perf_counter() - parse_started
                progress.update(
                    'document_parse',
                    'error',
                    (
                        f'文件 {uploaded_file.name} 解析失败，'
                        f'耗时 {parse_elapsed:.2f} 秒'
                    ),
                )
                raise

            parse_elapsed = time.perf_counter() - parse_started
            progress.update(
                'document_parse',
                'success',
                (
                    f'文件 {uploaded_file.name}，'
                    f'{len(document_content):,} 字符，'
                    f'耗时 {parse_elapsed:.2f} 秒'
                ),
            )

            with st.expander('查看 Docling 原始解析结果'):
                st.markdown(document_content)

            analyzer = AIAnalyzer()
            analysis = analyzer.analyze(
                document_content,
                on_event=progress.on_event,
            )

            status.update(
                label='AI 分析完成',
                state='complete',
                expanded=False,
            )

            st.session_state[ANALYSIS_RESULT_STATE_KEY] = analysis
            st.session_state[ANALYSIS_TRACE_STATE_KEY] = analyzer.last_trace
            st.session_state[ANALYSIS_FILE_STATE_KEY] = current_file_signature

        except AIAnalysisError as exc:
            progress.update(
                'final',
                'error',
                f'AI 分析失败：{exc}',
            )
            status.update(
                label='AI 分析失败',
                state='error',
                expanded=True,
            )
            st.error(f'AI 分析失败：{exc}')
            render_debug_panel(exc.trace, traceback.format_exc())

        except Exception as exc:
            progress.update(
                'final',
                'error',
                f'{type(exc).__name__}: {exc}',
            )
            status.update(
                label='运行失败',
                state='error',
                expanded=True,
            )
            st.error(f'运行失败：{type(exc).__name__}: {exc}')
            if analyzer is not None:
                render_debug_panel(
                    analyzer.last_trace,
                    traceback.format_exc(),
                )

        finally:
            if temp_path and os.path.exists(temp_path):
                os.remove(temp_path)

    analysis = st.session_state.get(ANALYSIS_RESULT_STATE_KEY)
    if analysis is not None:
        st.download_button(
            label='下载 AI 分析结果 JSON',
            data=analysis.model_dump_json(indent=2),
            file_name='product_analysis.json',
            mime='application/json',
            on_click='ignore',
        )
        render_analysis_report(analysis)
        render_debug_panel(
            st.session_state.get(ANALYSIS_TRACE_STATE_KEY)
        )
