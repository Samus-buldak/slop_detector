"""Сигнальные модули (multi-slayer feature extraction).

Каждый сигнал — чистая функция, возвращающая dict признаков + список
объяснений (explanations), пригодных для показа аналитику.
"""
from .text_stylometry import text_stylometry_signals      # noqa: F401
from .ai_liar import ai_liar_score                         # noqa: F401
from .html_structure import html_structure_signals         # noqa: F401
from .url_signals import url_signals                       # noqa: F401
from .header_signals import header_signals                 # noqa: F401
