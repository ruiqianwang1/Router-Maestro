"""Tests for mid-stream upstream disconnect diagnostics.

An upstream connection cut *after* a stream opened means the response was
truncated. Previously this landed in the generic transport error path, so the
logs showed a bare transport failure with no sign that a response had been
amputated partway through -- which made silent tunnel disconnects very hard to
diagnose. These tests pin the distinct classification and logging.

The classification lives in the transport rather than in each provider stream
method, so every stream path gets it by construction; the wiring tests at the
bottom pin that.
"""

from __future__ import annotations

import logging

import httpx
import pytest

from router_maestro.providers.base import ProviderError, ProviderFailureKind
from router_maestro.providers.copilot_support.auth_session import CopilotAuthSession
from router_maestro.providers.copilot_support.transport import CopilotTransport
from router_maestro.providers.h2_keepalive import (
    StreamWatch,
    is_mid_stream_disconnect,
    log_truncated_stream,
    raise_mid_stream_disconnect,
)

KEEPALIVE_LOGGER = "router_maestro.providers.h2_keepalive"


def _watch(chunks: int = 0, *, terminal: bool = False) -> StreamWatch:
    watch = StreamWatch()
    for _ in range(chunks):
        watch.record_chunk()
    watch.saw_terminal = terminal
    return watch


class TestDisconnectClassification:
    @pytest.mark.parametrize(
        "error",
        [
            httpx.RemoteProtocolError("Server disconnected"),
            httpx.ReadError("connection reset"),
            httpx.NetworkError("network down"),
        ],
    )
    def test_transport_cuts_are_recognized(self, error):
        assert is_mid_stream_disconnect(error) is True

    @pytest.mark.parametrize(
        "error",
        [
            httpx.ConnectTimeout("timed out"),
            httpx.ReadTimeout("read timed out"),
            httpx.PoolTimeout("pool timed out"),
            httpx.InvalidURL("bad url"),
        ],
    )
    def test_other_errors_are_not_misclassified(self, error):
        """Timeouts have their own handler and must not be reported as disconnects."""
        assert is_mid_stream_disconnect(error) is False


class TestMidStreamDisconnectRaise:
    def test_raises_retryable_transport_error(self):
        error = httpx.RemoteProtocolError("Server disconnected")

        with pytest.raises(ProviderError) as excinfo:
            raise_mid_stream_disconnect(
                "Copilot",
                error,
                provider="github-copilot",
                model="claude-opus-4",
                watch=_watch(12),
            )

        raised = excinfo.value
        assert raised.kind is ProviderFailureKind.TRANSPORT
        assert raised.status_code == 502
        assert raised.retryable is True
        assert raised.provider == "github-copilot"
        assert raised.model == "claude-opus-4"
        assert raised.cause is error

    def test_message_is_distinct_from_generic_transport_failure(self):
        """The downstream error must say the stream was cut, not just 'transport failed'."""
        with pytest.raises(ProviderError) as excinfo:
            raise_mid_stream_disconnect(
                "Copilot",
                httpx.RemoteProtocolError("Server disconnected"),
                watch=_watch(),
            )
        assert "disconnected mid-response" in excinfo.value.safe_message

    def test_logs_chunk_count_and_timings(self, caplog):
        """Timings are what let a user recognize a fixed-interval idle reclaim."""
        with caplog.at_level(logging.ERROR, logger=KEEPALIVE_LOGGER):
            with pytest.raises(ProviderError):
                raise_mid_stream_disconnect(
                    "Copilot",
                    httpx.RemoteProtocolError("Server disconnected"),
                    watch=_watch(7),
                )

        assert len(caplog.records) == 1
        message = caplog.records[0].getMessage()
        assert "7 chunk(s)" in message
        assert "into the stream" in message
        assert "since the last chunk" in message
        assert "truncated" in message
        # The log must point at the actual cause, not leave the user guessing.
        assert "intermediary" in message
        assert "keepalive" in message

    def test_reports_a_zero_chunk_cut(self, caplog):
        """The keepalive's own failure mode cuts the stream before any chunk lands.

        A long reasoning turn can go minutes without emitting an SSE byte, so
        the disconnect this feature exists to prevent commonly has no chunks at
        all. Reporting must not depend on chunks having arrived.
        """
        with caplog.at_level(logging.ERROR, logger=KEEPALIVE_LOGGER):
            with pytest.raises(ProviderError):
                raise_mid_stream_disconnect(
                    "Copilot",
                    httpx.RemoteProtocolError("Server disconnected"),
                    watch=_watch(0),
                )
        assert "0 chunk(s)" in caplog.records[0].getMessage()

    def test_does_not_log_silently(self, caplog):
        """The original failure mode logged nothing at all; guard against regressing."""
        with caplog.at_level(logging.DEBUG, logger=KEEPALIVE_LOGGER):
            with pytest.raises(ProviderError):
                raise_mid_stream_disconnect(
                    "Copilot",
                    httpx.RemoteProtocolError("Server disconnected"),
                )
        assert caplog.records, "mid-stream disconnect produced no log output"
        assert caplog.records[0].levelno >= logging.ERROR


class TestTruncatedStreamLog:
    def test_warns_when_stream_ends_without_terminal_event(self, caplog):
        """A clean EOF with no terminal marker is still a truncated response."""
        with caplog.at_level(logging.WARNING, logger=KEEPALIVE_LOGGER):
            log_truncated_stream("Copilot", model="claude-opus-4", watch=_watch(3))

        assert len(caplog.records) == 1
        message = caplog.records[0].getMessage()
        assert "3 chunk(s)" in message
        assert "truncated" in message
        assert "claude-opus-4" in message


class TestStreamWatch:
    def test_records_chunks(self):
        watch = StreamWatch()
        assert watch.chunks == 0
        watch.record_chunk()
        watch.record_chunk()
        assert watch.chunks == 2

    def test_idle_never_exceeds_elapsed(self):
        watch = StreamWatch()
        watch.record_chunk()
        assert watch.idle <= watch.elapsed


def _transport() -> CopilotTransport:
    auth = CopilotAuthSession.__new__(CopilotAuthSession)
    auth.cached_token = "tok"
    auth.api_base = "https://api.example.invalid"
    auth.provider_name = "github-copilot"
    return CopilotTransport(auth)


class _FakeStreamResponse:
    """Minimal stand-in for the httpx.Response the transport hands to callers."""

    def __init__(self, lines: list[str], *, raise_at_end: BaseException | None = None) -> None:
        self.status_code = 200
        self.headers = {}
        self._lines = lines
        self._raise_at_end = raise_at_end

    async def aiter_lines(self):
        for line in self._lines:
            yield line
        if self._raise_at_end is not None:
            raise self._raise_at_end


class TestTransportInstrumentation:
    """The transport counts chunks and classifies cuts for *every* stream path.

    Putting this in the transport rather than in each provider stream method is
    what keeps chat and Responses covered without either one opting in.
    """

    async def _consume(self, transport, response, **kwargs):
        """Drive stream_with_auth_retry against a canned response."""
        import contextlib

        @contextlib.asynccontextmanager
        async def fake_stream(*args, **kw):
            yield response

        lines = []
        async with transport.stream_with_auth_retry(
            "/chat/completions",
            json={},
            headers_kwargs={},
            model="claude-opus-4",
            get_client=lambda: _FakeClient(fake_stream),
            get_headers=lambda **kw: {},
            recycle_client=_noop,
            refresh_for_auth_status=_never,
            raise_auth_failure=lambda *a, **kw: None,
            **kwargs,
        ) as streamed:
            async for line in streamed.aiter_lines():
                lines.append(line)
        return lines

    @pytest.mark.asyncio
    async def test_counts_chunks_and_recognizes_a_terminal_marker(self, caplog):
        transport = _transport()
        response = _FakeStreamResponse(['data: {"x":1}', "data: [DONE]"])

        with caplog.at_level(logging.WARNING, logger=KEEPALIVE_LOGGER):
            lines = await self._consume(transport, response)

        assert lines == ['data: {"x":1}', "data: [DONE]"]
        assert not caplog.records, "a stream with [DONE] must not warn about truncation"

    @pytest.mark.asyncio
    async def test_warns_when_a_stream_ends_without_a_terminal_marker(self, caplog):
        transport = _transport()
        response = _FakeStreamResponse(['data: {"x":1}'])

        with caplog.at_level(logging.WARNING, logger=KEEPALIVE_LOGGER):
            await self._consume(transport, response)

        assert any("truncated" in r.getMessage() for r in caplog.records)

    @pytest.mark.asyncio
    async def test_responses_terminal_event_counts_as_terminal(self, caplog):
        """The Responses API ends with an event, not `[DONE]`."""
        transport = _transport()
        response = _FakeStreamResponse(["event: response.completed", 'data: {"x":1}'])

        with caplog.at_level(logging.WARNING, logger=KEEPALIVE_LOGGER):
            await self._consume(transport, response)

        assert not caplog.records

    @pytest.mark.asyncio
    async def test_mid_stream_cut_is_classified(self, caplog):
        transport = _transport()
        response = _FakeStreamResponse(
            ['data: {"x":1}'],
            raise_at_end=httpx.RemoteProtocolError("Server disconnected"),
        )

        with caplog.at_level(logging.ERROR, logger=KEEPALIVE_LOGGER):
            with pytest.raises(ProviderError) as excinfo:
                await self._consume(transport, response)

        assert "disconnected mid-response" in excinfo.value.safe_message
        assert excinfo.value.kind is ProviderFailureKind.TRANSPORT
        assert "1 chunk(s)" in caplog.records[0].getMessage()

    @pytest.mark.asyncio
    async def test_silent_cut_before_any_chunk_is_classified(self):
        """The case this feature exists for: cut during the silent reasoning window."""
        transport = _transport()
        response = _FakeStreamResponse(
            [], raise_at_end=httpx.RemoteProtocolError("Server disconnected")
        )

        with pytest.raises(ProviderError) as excinfo:
            await self._consume(transport, response)

        assert "disconnected mid-response" in excinfo.value.safe_message

    @pytest.mark.asyncio
    async def test_non_transport_errors_pass_through_unchanged(self):
        """A caller's own exception must not be relabelled as a disconnect."""
        transport = _transport()
        response = _FakeStreamResponse([], raise_at_end=ValueError("caller bug"))

        with pytest.raises(ValueError, match="caller bug"):
            await self._consume(transport, response)


class _FakeClient:
    def __init__(self, stream_cm):
        self.stream = stream_cm


async def _noop() -> None:
    return None


async def _never(path: str, status: int) -> bool:
    return False
