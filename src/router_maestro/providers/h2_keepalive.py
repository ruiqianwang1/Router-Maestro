"""HTTP/2 PING keepalive for upstream connections.

Why this exists
---------------
When Router-Maestro reaches an upstream API through a local tunnel client
(v2rayN/Xray, a corporate proxy, a VPN), intermediaries on the path may reclaim
connections they consider idle. Some do it without sending a GOAWAY or a TLS
close_notify: the socket is simply cut, and the local side only learns about it
by reading EOF.

Long reasoning turns make this easy to hit. With a high reasoning effort and a
large thinking budget, an upstream can spend several minutes producing tokens
before flushing the first SSE event. The HTTP/2 stream is wide open the whole
time but carries zero bytes, which is indistinguishable from an abandoned
connection to anything in the middle.

TCP keepalive cannot solve this. The proxy's socket terminates at the local
tunnel client (e.g. 127.0.0.1:10808), so keepalive probes loop back on the
loopback interface and never traverse the tunnel. Refreshing an intermediary's
idle timer requires real bytes along the *whole* path.

An HTTP/2 PING is the right instrument: it is a connection-level control frame
(RFC 9113 §6.7), so it produces bytes on the wire without touching any stream's
data flow. It cannot corrupt the response body the way injecting bytes into the
HTTP framing would.

Implementation notes
--------------------
httpcore 1.x has no PING support of its own -- its ``keepalive_expiry`` only
governs pool eviction and emits no frames -- so the frame is written directly
through the connection's own h2 state machine.

Two details keep this safe:

* Writes go through httpcore's ``_write_lock``. h2 state mutation and the
  ``data_to_send()`` drain are not atomic, so writing without that lock could
  interleave with a request's frames and corrupt the connection.
* PING ACKs need no handling here. httpcore's event dispatch only routes
  ``ResponseReceived``/``DataReceived``/``StreamEnded``/``StreamReset``/
  ``RemoteSettingsChanged``/``ConnectionTerminated``; a ``PingAckReceived``
  matches no branch and is discarded, so ACKs never reach response streams.

Because httpcore internals are private, every access is defensive: if a future
version renames these attributes the keepalive disables itself and logs once,
rather than breaking requests.

This module also owns the diagnostics for the failure it prevents -- see
``raise_mid_stream_disconnect``. Keeping both here means the whole feature lives
in one file that upstream does not have, so it survives rebases untouched.
"""

from __future__ import annotations

import asyncio
import contextlib
import time
from typing import Any, NoReturn

import httpx

from router_maestro.utils import get_logger

logger = get_logger("providers.h2_keepalive")

# PING payloads must be exactly 8 bytes (RFC 9113 section 6.7).
_PING_PAYLOAD = b"rm-alive"

# Seconds between PING frames. Must stay comfortably below the shortest idle
# reclaim window on the path (observed windows start around 240-270s).
KEEPALIVE_INTERVAL = 60.0


def _iter_h2_connections(client: Any) -> list[Any]:
    """Return live HTTP/2 connection objects owned by an httpx client.

    Walks httpx -> httpcore pool -> connection -> protocol-specific connection.
    Handles direct connections and SOCKS/HTTP proxy connections alike, since
    both expose the negotiated protocol object as ``_connection``.
    """
    transport = getattr(client, "_transport", None)
    pool = getattr(transport, "_pool", None)
    if pool is None:
        return []
    try:
        connections = list(pool.connections)
    except Exception:  # pragma: no cover - defensive against httpcore changes
        return []

    found = []
    for connection in connections:
        # AsyncHTTPConnection / AsyncSocks5Connection wrap the real protocol
        # object; an HTTP/1.1 connection has no _h2_state and is skipped.
        inner = getattr(connection, "_connection", None)
        if inner is None:
            continue
        if not hasattr(inner, "_h2_state") or not hasattr(inner, "_write_lock"):
            continue
        if not hasattr(inner, "_network_stream"):
            continue
        found.append(inner)
    return found


async def _ping_connection(connection: Any, *, write_timeout: float) -> bool:
    """Send a single PING frame on one HTTP/2 connection.

    Returns True if a frame was written. Never raises: a connection that is
    already dead is exactly the condition the caller is trying to detect, and
    the request's own read path will surface it with proper context.
    """
    # A connection that already saw GOAWAY, or has a stored write error, must
    # not be touched -- writing would raise and add nothing.
    if getattr(connection, "_connection_terminated", None) is not None:
        return False
    if getattr(connection, "_write_exception", None) is not None:
        return False

    try:
        async with connection._write_lock:
            h2_state = connection._h2_state
            h2_state.ping(_PING_PAYLOAD)
            data = h2_state.data_to_send()
            if not data:
                return False
            await connection._network_stream.write(data, {"write": write_timeout})
        return True
    except Exception as error:
        # Downgrade to debug: the stream's own read path reports the failure
        # with request context, so an error here would just be duplicate noise.
        logger.debug("h2 keepalive ping failed (%s)", type(error).__name__)
        return False


class H2KeepaliveTask:
    """Periodically PING an httpx client's idle HTTP/2 connections.

    Owns one background task per client. Safe to start and stop repeatedly;
    starting an already-running task is a no-op.
    """

    __slots__ = ("_client", "_interval", "_task", "_write_timeout", "_warned_unsupported")

    def __init__(self, client: Any, *, interval: float, write_timeout: float = 10.0) -> None:
        self._client = client
        self._interval = interval
        self._write_timeout = write_timeout
        self._task: asyncio.Task | None = None
        self._warned_unsupported = False

    @property
    def running(self) -> bool:
        """Return whether the background ping loop is active."""
        return self._task is not None and not self._task.done()

    def start(self) -> None:
        """Start the ping loop if it is not already running."""
        if self.running:
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            # No event loop (sync context, e.g. some CLI paths). Keepalive is
            # only meaningful for in-flight async streams, so skip silently.
            return
        self._task = loop.create_task(self._run(), name="h2-keepalive")

    async def stop(self) -> None:
        """Cancel the ping loop and wait for it to unwind."""
        task = self._task
        self._task = None
        if task is None or task.done():
            return
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await task

    async def _run(self) -> None:
        while True:
            await asyncio.sleep(self._interval)
            client = self._client
            if client is None or getattr(client, "is_closed", False):
                return
            try:
                connections = _iter_h2_connections(client)
            except Exception:  # pragma: no cover - defensive
                connections = []

            if not connections:
                # Either no HTTP/2 connection is open yet, or httpcore's
                # internals moved. Both are non-fatal; warn once for the latter
                # so a silent regression is visible in logs.
                if not self._warned_unsupported and self._pool_has_connections(client):
                    self._warned_unsupported = True
                    logger.warning(
                        "h2 keepalive found no HTTP/2 connections to ping; "
                        "upstream may be HTTP/1.1 or httpcore internals changed"
                    )
                continue

            sent = 0
            for connection in connections:
                if await _ping_connection(connection, write_timeout=self._write_timeout):
                    sent += 1
            if sent:
                logger.debug("h2 keepalive sent %d ping(s)", sent)

    @staticmethod
    def _pool_has_connections(client: Any) -> bool:
        transport = getattr(client, "_transport", None)
        pool = getattr(transport, "_pool", None)
        if pool is None:
            return False
        try:
            return bool(list(pool.connections))
        except Exception:  # pragma: no cover - defensive
            return False


def is_mid_stream_disconnect(error: BaseException) -> bool:
    """Return whether an error indicates the upstream cut an open stream.

    `RemoteProtocolError` is what httpcore raises when a read returns EOF with
    frames still outstanding -- the exact signature of a connection torn down
    without a GOAWAY. `ReadError`/`NetworkError` cover the same condition
    surfaced at the socket layer.
    """
    return isinstance(
        error,
        (httpx.RemoteProtocolError, httpx.ReadError, httpx.NetworkError),
    )


class StreamWatch:
    """Track elapsed and idle time across a streamed upstream response.

    A stream cut by an idle-connection reclaim is indistinguishable from a
    generic transport error unless you know *when* it died. The idle duration
    is the tell: a disconnect at a suspiciously round idle interval is an
    intermediary's timer firing, not the upstream failing.

    Note that ``chunks`` is not a truncation signal on its own. The failure this
    module addresses hits hardest when the upstream is still reasoning and has
    emitted nothing at all, so a cut stream commonly has zero chunks. Only the
    caller knows whether the response stream had opened.
    """

    __slots__ = ("_started_at", "_last_chunk_at", "chunks", "saw_terminal")

    def __init__(self) -> None:
        now = time.monotonic()
        self._started_at = now
        self._last_chunk_at = now
        self.chunks = 0
        self.saw_terminal = False

    def record_chunk(self) -> None:
        """Note that a chunk reached the consumer, resetting the idle clock."""
        self.chunks += 1
        self._last_chunk_at = time.monotonic()

    @property
    def elapsed(self) -> float:
        """Seconds since the stream opened."""
        return time.monotonic() - self._started_at

    @property
    def idle(self) -> float:
        """Seconds since the last chunk (or since open, if none arrived)."""
        return time.monotonic() - self._last_chunk_at


def raise_mid_stream_disconnect(
    label: str,
    error: BaseException,
    *,
    provider: str | None = None,
    model: str | None = None,
    watch: StreamWatch | None = None,
) -> NoReturn:
    """Raise a ProviderError for a stream cut after the response had opened.

    This is deliberately separate from the generic transport error path. An
    upstream connection dropped mid-stream is not an ordinary transport
    failure: it means the response was truncated, and it is the signature of an
    intermediary reclaiming a connection it considered idle (which sends no
    close frame, so the local side only sees EOF).

    Folding this into the generic handler is what made the failure mode hard to
    diagnose -- the logs showed a bare transport error with no indication that a
    stream had been amputated partway through.

    Left retryable: the router only retries while it is still pulling the first
    chunk, i.e. before anything reached the client. A cut during the long silent
    reasoning window -- the exact case this module exists for -- lands there, and
    retrying it is safe. Once chunks have been delivered the router has already
    handed the stream off, so this flag no longer causes a retry.
    """
    from router_maestro.providers.base import ProviderError, ProviderFailureKind

    detail = f"after {watch.chunks if watch else 0} chunk(s)"
    if watch is not None:
        detail += f", {watch.elapsed:.1f}s into the stream"
        detail += f", {watch.idle:.1f}s since the last chunk"
    logger.error(
        "%s stream disconnected by upstream %s (%s); "
        "response is truncated. A silent mid-stream disconnect usually means an "
        "intermediary (tunnel/proxy/CDN) reclaimed the connection while it was idle -- "
        "check the h2 keepalive interval if this recurs on long reasoning turns.",
        label,
        detail,
        type(error).__name__,
    )
    raise ProviderError(
        f"{label} stream disconnected mid-response ({type(error).__name__})",
        status_code=502,
        retryable=True,
        kind=ProviderFailureKind.TRANSPORT,
        provider=provider or label,
        model=model,
        cause=error,
    ) from error


def log_truncated_stream(
    label: str,
    *,
    model: str | None = None,
    watch: StreamWatch | None = None,
) -> None:
    """Log a stream that ended cleanly but without a terminal event.

    Distinguishes an amputated stream from a normal completion. An upstream
    that closes without a finish reason has truncated the response, even though
    the transport reports an orderly shutdown.
    """
    logger.warning(
        "%s stream ended without a terminal event after %d chunk(s)%s "
        "(model=%s); response is truncated, not complete.",
        label,
        watch.chunks if watch else 0,
        f", {watch.elapsed:.1f}s" if watch is not None else "",
        model,
    )
