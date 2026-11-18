"""Tests for reports and the CLI."""

from evalharness.harness import run_evaluation
from evalharness.models import EchoModel, KeywordModel, OracleModel
from evalharness.report import leaderboard_table, markdown_report, results_to_csv
from evalharness.cli import main


def _results():
    return [run_evaluation(model, "zero-shot")
            for model in (OracleModel(), KeywordModel(), EchoModel())]


def test_leaderboard_table_names_the_winner_first():
    table = leaderboard_table(_results())
    lines = [line for line in table.splitlines() if line.startswith("|")]
    assert "oracle" in lines[2]
    assert "keyword" in lines[3]


def test_markdown_report_has_all_sections():
    report = markdown_report(_results())
    assert "# LLM Evaluation Report" in report
    assert "## Leaderboard" in report
    assert "## Per-category scores" in report
    assert "## Weakest items" in report


def test_csv_has_one_row_per_run_plus_header():
    rows = results_to_csv(_results()).strip().splitlines()
    assert len(rows) == 4
    assert rows[0].startswith("model,template,score")


def test_cli_prompts_and_models_listings(capsys):
    assert main(["prompts"]) == 0
    assert "zero-shot" in capsys.readouterr().out
    assert main(["models"]) == 0
    assert "keyword" in capsys.readouterr().out


def test_cli_run_prints_leaderboard(capsys):
    assert main(["run", "--models", "oracle", "keyword", "--templates", "zero-shot"]) == 0
    out = capsys.readouterr().out
    assert "oracle/zero-shot" in out
    assert "Leaderboard" not in out  # table header uses Rank, not the word
    assert "| Rank |" in out
