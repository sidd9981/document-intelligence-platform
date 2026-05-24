"""
Prompt registry.

Loads versioned prompt templates from the prompts/ directory.
Every prompt is a plain text file with {placeholders} for variables.

Prompts are loaded once at startup and cached in memory. The version
string is extracted from the filename — synthesis_v2.txt has version
synthesis_v2. This version is stamped on every Langfuse trace so
faithfulness regressions can be correlated to specific prompt changes.

Adding a new prompt version means creating a new file. The registry
always loads the highest version number for each prompt name.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

from finsight.telemetry.tracing import get_tracer

logger = logging.getLogger(__name__)
tracer = get_tracer(__name__)

PROMPTS_DIR = Path(__file__).parent.parent.parent / "prompts"

_cache: dict[str, tuple[str, str]] = {}


def _parse_version(filename: str) -> tuple[str, int]:
    """Extract prompt name and version number from a filename.

    synthesis_v2.txt -> ("synthesis", 2)
    faithfulness_judge_v1.txt -> ("faithfulness_judge", 1)
    """
    match = re.match(r"^(.+)_v(\d+)\.txt$", filename)
    if not match:
        raise ValueError(f"prompt filename {filename!r} does not match pattern <name>_v<N>.txt")
    return match.group(1), int(match.group(2))


def _find_latest_version(prompt_name: str, prompts_dir: Path) -> tuple[Path, str]:
    """Find the highest-versioned file for a given prompt name.

    Returns the file path and the version string e.g. synthesis_v2.
    """
    candidates = list(prompts_dir.glob(f"{prompt_name}_v*.txt"))
    if not candidates:
        raise FileNotFoundError(
            f"no prompt files found for {prompt_name!r} in {prompts_dir}"
        )

    versioned = []
    for path in candidates:
        try:
            _, version_num = _parse_version(path.name)
            versioned.append((version_num, path))
        except ValueError:
            continue

    versioned.sort(key=lambda x: x[0], reverse=True)
    best_path = versioned[0][1]
    _, version_num = _parse_version(best_path.name)
    version_str = f"{prompt_name}_v{version_num}"
    return best_path, version_str


def load_prompt(prompt_name: str, prompts_dir: Path | None = None) -> tuple[str, str]:
    """Load the latest version of a named prompt template.

    Cached after first load. Returns (template_text, version_string).

    Args:
        prompt_name: Base name without version or extension,
                     e.g. "synthesis", "faithfulness_judge".
        prompts_dir: Override the default prompts directory. Used in tests.

    Returns:
        Tuple of (template text with {placeholders}, version string).

    Raises:
        FileNotFoundError: If no versioned file exists for prompt_name.
    """
    cache_key = f"{prompts_dir or PROMPTS_DIR}:{prompt_name}"

    if cache_key in _cache:
        return _cache[cache_key]

    directory = prompts_dir or PROMPTS_DIR

    with tracer.start_as_current_span("prompt_registry.load") as span:
        span.set_attribute("prompt_name", prompt_name)

        path, version_str = _find_latest_version(prompt_name, directory)
        template = path.read_text(encoding="utf-8")

        _cache[cache_key] = (template, version_str)

        logger.info("loaded prompt %s from %s", version_str, path)
        span.set_attribute("version", version_str)

        return template, version_str


def render_prompt(template: str, **kwargs) -> str:
    """Render a prompt template with the given variables.

    Args:
        template: Prompt text with {placeholder} variables.
        **kwargs: Variable values to substitute.

    Returns:
        Rendered prompt string.

    Raises:
        KeyError: If a required placeholder is missing from kwargs.
    """
    return template.format(**kwargs)


def clear_cache() -> None:
    """Clear the prompt cache. Used in tests."""
    _cache.clear()