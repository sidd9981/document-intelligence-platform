"""
Unit tests for the prompt registry.

Uses a temp directory so no real prompt files are needed.
"""

from __future__ import annotations

import pytest
from pathlib import Path

from finsight.mlops.prompt_registry import (
    _parse_version,
    _find_latest_version,
    load_prompt,
    render_prompt,
    clear_cache,
)


@pytest.fixture(autouse=True)
def reset_cache():
    clear_cache()
    yield
    clear_cache()


@pytest.fixture
def prompts_dir(tmp_path):
    return tmp_path


def test_parse_version_simple():
    name, version = _parse_version("synthesis_v1.txt")
    assert name == "synthesis"
    assert version == 1


def test_parse_version_multi_word():
    name, version = _parse_version("faithfulness_judge_v3.txt")
    assert name == "faithfulness_judge"
    assert version == 3


def test_parse_version_invalid_raises():
    with pytest.raises(ValueError):
        _parse_version("synthesis.txt")


def test_find_latest_version_picks_highest(prompts_dir):
    (prompts_dir / "synthesis_v1.txt").write_text("v1 template")
    (prompts_dir / "synthesis_v2.txt").write_text("v2 template")
    (prompts_dir / "synthesis_v3.txt").write_text("v3 template")

    path, version_str = _find_latest_version("synthesis", prompts_dir)
    assert version_str == "synthesis_v3"
    assert path.name == "synthesis_v3.txt"


def test_find_latest_version_raises_when_missing(prompts_dir):
    with pytest.raises(FileNotFoundError):
        _find_latest_version("nonexistent", prompts_dir)


def test_load_prompt_returns_template_and_version(prompts_dir):
    (prompts_dir / "synthesis_v1.txt").write_text("Answer: {question}")

    template, version = load_prompt("synthesis", prompts_dir=prompts_dir)
    assert "Answer:" in template
    assert version == "synthesis_v1"


def test_load_prompt_is_cached(prompts_dir):
    (prompts_dir / "synthesis_v1.txt").write_text("template text")

    result1 = load_prompt("synthesis", prompts_dir=prompts_dir)
    result2 = load_prompt("synthesis", prompts_dir=prompts_dir)
    assert result1 == result2


def test_load_prompt_picks_latest_version(prompts_dir):
    (prompts_dir / "synthesis_v1.txt").write_text("old template")
    (prompts_dir / "synthesis_v2.txt").write_text("new template")

    template, version = load_prompt("synthesis", prompts_dir=prompts_dir)
    assert template == "new template"
    assert version == "synthesis_v2"


def test_render_prompt_substitutes_variables():
    template = "Question: {question}\nContext: {context}"
    result = render_prompt(template, question="What is revenue?", context="Revenue was $383B.")
    assert result == "Question: What is revenue?\nContext: Revenue was $383B."


def test_render_prompt_raises_on_missing_variable():
    template = "Question: {question}\nContext: {context}"
    with pytest.raises(KeyError):
        render_prompt(template, question="What is revenue?")


def test_load_prompt_raises_when_no_files(prompts_dir):
    with pytest.raises(FileNotFoundError):
        load_prompt("missing_prompt", prompts_dir=prompts_dir)