from __future__ import annotations

import logging
import os

import streamlit as st
from dotenv import load_dotenv

from services.database import DatabaseInitializationError, initialize_database
from services.ui_components import inject_button_styles


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
    layout='wide',
)
inject_button_styles()

try:
    initialize_database()
except DatabaseInitializationError as exc:
    st.error(f'数据库初始化失败：{exc}')
    st.stop()

pages = [
    st.Page(
        'views/new_analysis.py',
        title='新建分析',
        default=True,
    ),
    st.Page(
        'views/report_management.py',
        title='报告管理',
    ),
    st.Page(
        'views/report_detail.py',
        title='报告详情',
    ),
    st.Page(
        'views/test_case_management.py',
        title='测试用例管理',
    ),
]

st.sidebar.caption('AI-Powered Requirement Analysis')
selected_page = st.navigation(pages, position='sidebar')
selected_page.run()
