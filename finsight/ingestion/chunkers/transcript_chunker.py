"""
Chunker for earnings call transcripts.

Splits by speaker turn. Each turn becomes one chunk with a speaker
metadata header prepended. The header is what lets the synthesis agent
attribute statements to specific executives across multiple years.

Target format: speaker name followed by colon on its own line, then
turn content until the next speaker line. This matches Seeking Alpha
and Finnhub transcript formats.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from datetime import date

from finsight.services.llm import count_tokens

CHUNK_MIN_TOKENS = 20

SPEAKER_LINE_RE = re.compile(r"^([A-Z][A-Za-z\s\.\-']+):\s*$")


@dataclass
class TranscriptChunk:
    chunk_id: str
    content: str
    token_count: int
    chunk_index: int
    speaker: str
    speaker_role: str


def chunk_transcript(
    text: str,
    doc_id: str,
    company_name: str,
    call_date: date,
    speaker_roles: dict[str, str] | None = None,
) -> list[TranscriptChunk]:
    """Split a transcript into one chunk per speaker turn.

    Args:
        text: Raw transcript text.
        doc_id: Used to generate stable chunk IDs.
        company_name: Canonical company name, included in the speaker header.
        call_date: Date of the earnings call, included in the speaker header.
        speaker_roles: Optional mapping of speaker name to role, e.g.
                       {"Tim Cook": "CEO", "Luca Maestri": "CFO"}.
                       Speakers not in this dict get role "Unknown".

    Returns:
        List of TranscriptChunk objects in order of appearance.
    """
    if speaker_roles is None:
        speaker_roles = {}

    turns = _split_into_turns(text)

    chunks: list[TranscriptChunk] = []
    chunk_index = 0
    pending_speaker = ""
    pending_lines: list[str] = []

    for speaker, lines in turns:
        content = " ".join(lines).strip()
        if not content:
            continue

        token_count = count_tokens(content)
        if token_count < CHUNK_MIN_TOKENS:
            if pending_lines and pending_speaker == speaker:
                pending_lines.extend(lines)
                continue
            elif not pending_lines:
                pending_speaker = speaker
                pending_lines = lines
                continue

        if pending_lines:
            merged_content = " ".join(pending_lines).strip()
            merged_tokens = count_tokens(merged_content)
            if merged_tokens >= CHUNK_MIN_TOKENS:
                role = speaker_roles.get(pending_speaker, "Unknown")
                header = _speaker_header(pending_speaker, role, company_name, call_date)
                full_content = f"{header}\n\n{merged_content}"
                chunks.append(TranscriptChunk(
                    chunk_id=_chunk_id(doc_id, chunk_index),
                    content=full_content,
                    token_count=count_tokens(full_content),
                    chunk_index=chunk_index,
                    speaker=pending_speaker,
                    speaker_role=role,
                ))
                chunk_index += 1
            pending_speaker = ""
            pending_lines = []

        role = speaker_roles.get(speaker, "Unknown")
        header = _speaker_header(speaker, role, company_name, call_date)
        full_content = f"{header}\n\n{content}"

        chunks.append(TranscriptChunk(
            chunk_id=_chunk_id(doc_id, chunk_index),
            content=full_content,
            token_count=count_tokens(full_content),
            chunk_index=chunk_index,
            speaker=speaker,
            speaker_role=role,
        ))
        chunk_index += 1

    return chunks


def _split_into_turns(text: str) -> list[tuple[str, list[str]]]:
    """Parse transcript text into (speaker, lines) pairs.

    Speaker lines are identified by the SPEAKER_LINE_RE pattern:
    a name in title case followed by a colon on its own line.
    Everything between two speaker lines belongs to the first speaker.

    Text before the first speaker line is assigned to "Operator"
    since transcripts typically start with operator instructions.
    """
    lines = text.splitlines()
    turns: list[tuple[str, list[str]]] = []
    current_speaker = "Operator"
    current_lines: list[str] = []

    for line in lines:
        match = SPEAKER_LINE_RE.match(line.strip())
        if match:
            if current_lines:
                turns.append((current_speaker, current_lines))
            current_speaker = match.group(1).strip()
            current_lines = []
        else:
            stripped = line.strip()
            if stripped:
                current_lines.append(stripped)

    if current_lines:
        turns.append((current_speaker, current_lines))

    return turns


def _speaker_header(speaker: str, role: str, company: str, call_date: date) -> str:
    return f"[Speaker: {speaker}, Role: {role}, Company: {company}, Date: {call_date.isoformat()}]"


def _chunk_id(doc_id: str, chunk_index: int) -> str:
    raw = f"{doc_id}:{chunk_index}"
    return hashlib.sha256(raw.encode()).hexdigest()[:32]