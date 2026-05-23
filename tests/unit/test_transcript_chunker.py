"""
Unit tests for the transcript chunker.

No services needed. All tests run against in-memory text.
"""

import tiktoken
import pytest
from datetime import date

from finsight.services import llm
from finsight.ingestion.chunkers.transcript_chunker import (
    TranscriptChunk,
    chunk_transcript,
    _split_into_turns,
    _speaker_header,
)


@pytest.fixture(autouse=True)
def init_tokenizer():
    llm._tokenizer = tiktoken.get_encoding("cl100k_base")
    yield
    llm._tokenizer = None


SAMPLE_TRANSCRIPT = """Operator:
Welcome to Apple's Q4 2023 earnings call. I will now turn the call over to your host.

Tim Cook:
Thank you. Good afternoon everyone. We are pleased to report another strong quarter
with revenue of $89.5 billion. Our services business continues to grow and we remain
focused on innovation across all our product lines.

Luca Maestri:
Thank you Tim. Our gross margin was 45.2 percent for the quarter, up from 42.3 percent
a year ago. We generated operating cash flow of $21.6 billion and returned over
$24 billion to shareholders during the quarter.

Analyst:
Thank you for the detail. Can you speak to the demand environment in China and
whether you are seeing any impact from local competition?

Tim Cook:
China remains an incredibly important market for us. We saw some pressure in the
quarter but remain confident in our long-term position there. We continue to invest
in the region and our team there is doing fantastic work.
"""

CALL_DATE = date(2023, 10, 26)
SPEAKER_ROLES = {"Tim Cook": "CEO", "Luca Maestri": "CFO", "Operator": "Operator"}


def test_chunk_transcript_returns_chunks():
    chunks = chunk_transcript(SAMPLE_TRANSCRIPT, "doc-001", "Apple Inc.", CALL_DATE)
    assert len(chunks) >= 1


def test_all_chunks_have_required_fields():
    chunks = chunk_transcript(SAMPLE_TRANSCRIPT, "doc-002", "Apple Inc.", CALL_DATE)
    for chunk in chunks:
        assert chunk.chunk_id
        assert chunk.content
        assert chunk.token_count > 0
        assert chunk.speaker


def test_speaker_header_appears_in_chunk_content():
    chunks = chunk_transcript(
        SAMPLE_TRANSCRIPT, "doc-003", "Apple Inc.", CALL_DATE, SPEAKER_ROLES
    )
    tim_chunks = [c for c in chunks if c.speaker == "Tim Cook"]
    assert len(tim_chunks) >= 1
    for chunk in tim_chunks:
        assert "Tim Cook" in chunk.content
        assert "CEO" in chunk.content
        assert "Apple Inc." in chunk.content
        assert "2023-10-26" in chunk.content


def test_speaker_roles_are_assigned():
    chunks = chunk_transcript(
        SAMPLE_TRANSCRIPT, "doc-004", "Apple Inc.", CALL_DATE, SPEAKER_ROLES
    )
    luca_chunks = [c for c in chunks if c.speaker == "Luca Maestri"]
    assert len(luca_chunks) >= 1
    assert luca_chunks[0].speaker_role == "CFO"


def test_unknown_speaker_gets_unknown_role():
    chunks = chunk_transcript(
        SAMPLE_TRANSCRIPT, "doc-005", "Apple Inc.", CALL_DATE, SPEAKER_ROLES
    )
    analyst_chunks = [c for c in chunks if c.speaker == "Analyst"]
    assert len(analyst_chunks) >= 1
    assert analyst_chunks[0].speaker_role == "Unknown"


def test_chunk_ids_are_deterministic():
    chunks_a = chunk_transcript(SAMPLE_TRANSCRIPT, "doc-006", "Apple Inc.", CALL_DATE)
    chunks_b = chunk_transcript(SAMPLE_TRANSCRIPT, "doc-006", "Apple Inc.", CALL_DATE)
    assert [c.chunk_id for c in chunks_a] == [c.chunk_id for c in chunks_b]


def test_different_doc_ids_produce_different_chunk_ids():
    chunks_a = chunk_transcript(SAMPLE_TRANSCRIPT, "doc-a", "Apple Inc.", CALL_DATE)
    chunks_b = chunk_transcript(SAMPLE_TRANSCRIPT, "doc-b", "Apple Inc.", CALL_DATE)
    ids_a = {c.chunk_id for c in chunks_a}
    ids_b = {c.chunk_id for c in chunks_b}
    assert ids_a.isdisjoint(ids_b)


def test_split_into_turns_detects_speakers():
    turns = _split_into_turns(SAMPLE_TRANSCRIPT)
    speakers = [t[0] for t in turns]
    assert "Tim Cook" in speakers
    assert "Luca Maestri" in speakers


def test_split_into_turns_groups_lines_by_speaker():
    turns = _split_into_turns(SAMPLE_TRANSCRIPT)
    tim_turns = [t for t in turns if t[0] == "Tim Cook"]
    assert len(tim_turns) == 2
    first_turn_text = " ".join(tim_turns[0][1])
    assert "89.5 billion" in first_turn_text


def test_speaker_header_format():
    header = _speaker_header("Tim Cook", "CEO", "Apple Inc.", CALL_DATE)
    assert header == "[Speaker: Tim Cook, Role: CEO, Company: Apple Inc., Date: 2023-10-26]"