"""LLM Evaluation Harness — benchmark prompts and models, offline."""

from .harness import compare_runs, paired_bootstrap, run_evaluation
from .models import EchoModel, HttpModel, KeywordModel, OracleModel, RandomModel
from .report import leaderboard, leaderboard_table, markdown_report

__all__ = [
    "run_evaluation", "compare_runs", "paired_bootstrap",
    "OracleModel", "KeywordModel", "RandomModel", "EchoModel", "HttpModel",
    "leaderboard", "leaderboard_table", "markdown_report",
]
__version__ = "1.0.0"
