"""Tests for HTTP/2 PING keepalive on upstream connections.

Context: intermediaries between the proxy and the upstream API (tunnel clients,
CDNs, corporate proxies) can reclaim connections they consider idle without
sending a close frame. Long reasoning turns produce no SSE bytes for minutes,
so the connection looks idle and gets cut mid-stream. TCP keepalive cannot fix
this when the socket terminates at a local tunnel client, so the proxy emits
HTTP/2 PING frames to keep real bytes flowing end-to-end.
"""

from __future__ import annotations

import asyncio

import h2.config
import h2.connection
import pytest

from router_maestro.providers.h2_keepalive import (
    H2KeepaliveTask,
    _iter_h2_connections,
    _ping_connection,
)

# HTTP/2 frame header: 3-byte length, 1-byte type. PING is type 0x06.
PING_FRAME_TYPE = 0x06


class FakeNetworkStream:
    """Capture bytes written to the wire."""

    def __init__(self, fail: bool = False) -> None:
        self.writes: list[bytes] = []
        self.fail = fail

    async def write(self, data: bytes, timeout=None) -> None:
        if self.fail:
            raise ConnectionResetError("connection reset")
        self.writes.append(data)


class FakeH2Connection:
    """Stand-in for httpcore's AsyncHTTP2Connection with the attributes we touch."""

    def __init__(self, *, terminated=None, write_exception=None, fail_write=False) -> None:
        self._h2_state = h2.connection.H2Connection(
            config=h2.config.H2Configuration(client_side=True)
        )
        self._h2_state.initiate_connection()
        self._h2_state.data_to_send()  # drain the preamble
        self._network_stream = FakeNetworkStream(fail=fail_write)
        self._write_lock = asyncio.Lock()
        self._connection_terminated = terminated
        self._write_exception = write_exception


class FakeHTTP11Connection:
    """An HTTP/1.1 connection exposes no h2 state and must be skipped."""


class FakePoolConnection:
    def __init__(self, inner) -> None:
        self._connection = inner


class FakePool:
    def __init__(self, connections) -> None:
        self.connections = connections


class FakeTransport:
    def __init__(self, pool) -> None:
        self._pool = pool


class FakeClient:
    def __init__(self, connections, is_closed: bool = False) -> None:
        self._transport = FakeTransport(FakePool(connections))
        self.is_closed = is_closed


def _contains_ping_frame(data: bytes) -> bool:
    """Return whether a PING frame appears in a serialized frame stream."""
    offset = 0
    while offset + 9 <= len(data):
        length = int.from_bytes(data[offset : offset + 3], "big")
        frame_type = data[offset + 3]
        if frame_type == PING_FRAME_TYPE:
            return True
        offset += 9 + length
    return False


class TestConnectionDiscovery:
    def test_finds_http2_connections(self):
        h2_conn = FakeH2Connection()
        client = FakeClient([FakePoolConnection(h2_conn)])
        assert _iter_h2_connections(client) == [h2_conn]

    def test_skips_http11_connections(self):
        client = FakeClient([FakePoolConnection(FakeHTTP11Connection())])
        assert _iter_h2_connections(client) == []

    def test_skips_unestablished_connections(self):
        client = FakeClient([FakePoolConnection(None)])
        assert _iter_h2_connections(client) == []

    def test_mixed_pool_returns_only_http2(self):
        h2_conn = FakeH2Connection()
        client = FakeClient(
            [
                FakePoolConnection(FakeHTTP11Connection()),
                FakePoolConnection(h2_conn),
                FakePoolConnection(None),
            ]
        )
        assert _iter_h2_connections(client) == [h2_conn]

    def test_non_httpx_client_is_tolerated(self):
        """A client without httpcore internals must not raise."""

        class Opaque:
            pass

        assert _iter_h2_connections(Opaque()) == []


class TestPingConnection:
    @pytest.mark.asyncio
    async def test_writes_a_real_ping_frame(self):
        conn = FakeH2Connection()
        assert await _ping_connection(conn, write_timeout=5.0) is True
        assert len(conn._network_stream.writes) == 1
        assert _contains_ping_frame(conn._network_stream.writes[0])

    @pytest.mark.asyncio
    async def test_skips_terminated_connection(self):
        """A connection that received GOAWAY must not be written to."""
        conn = FakeH2Connection(terminated=object())
        assert await _ping_connection(conn, write_timeout=5.0) is False
        assert conn._network_stream.writes == []

    @pytest.mark.asyncio
    async def test_skips_connection_with_stored_write_error(self):
        conn = FakeH2Connection(write_exception=ConnectionError("dead"))
        assert await _ping_connection(conn, write_timeout=5.0) is False
        assert conn._network_stream.writes == []

    @pytest.mark.asyncio
    async def test_write_failure_is_swallowed(self):
        """A dead connection is the condition being detected, not an error to raise.

        The request's own read path reports the failure with proper context, so
        raising here would only produce duplicate, context-free noise.
        """
        conn = FakeH2Connection(fail_write=True)
        assert await _ping_connection(conn, write_timeout=5.0) is False

    @pytest.mark.asyncio
    async def test_holds_write_lock_while_writing(self):
        """h2 state mutation and data_to_send() must be atomic w.r.t. other writers.

        Without the lock, a ping could interleave with a request's frames and
        corrupt the connection.
        """
        conn = FakeH2Connection()
        await conn._write_lock.acquire()

        task = asyncio.create_task(_ping_connection(conn, write_timeout=5.0))
        await asyncio.sleep(0.05)
        assert conn._network_stream.writes == [], "ping wrote while lock was held"

        conn._write_lock.release()
        assert await task is True
        assert len(conn._network_stream.writes) == 1


class TestKeepaliveTask:
    @pytest.mark.asyncio
    async def test_pings_on_interval(self):
        conn = FakeH2Connection()
        client = FakeClient([FakePoolConnection(conn)])
        task = H2KeepaliveTask(client, interval=0.05)
        task.start()
        try:
            await asyncio.sleep(0.17)
        finally:
            await task.stop()
        assert len(conn._network_stream.writes) >= 2

    @pytest.mark.asyncio
    async def test_stop_cancels_the_loop(self):
        conn = FakeH2Connection()
        client = FakeClient([FakePoolConnection(conn)])
        task = H2KeepaliveTask(client, interval=0.05)
        task.start()
        await asyncio.sleep(0.12)
        await task.stop()
        assert not task.running

        count = len(conn._network_stream.writes)
        await asyncio.sleep(0.15)
        assert len(conn._network_stream.writes) == count, "pings continued after stop"

    @pytest.mark.asyncio
    async def test_start_is_idempotent(self):
        client = FakeClient([FakePoolConnection(FakeH2Connection())])
        task = H2KeepaliveTask(client, interval=0.05)
        task.start()
        first = task._task
        task.start()
        try:
            assert task._task is first
        finally:
            await task.stop()

    @pytest.mark.asyncio
    async def test_stops_when_client_closes(self):
        conn = FakeH2Connection()
        client = FakeClient([FakePoolConnection(conn)])
        task = H2KeepaliveTask(client, interval=0.05)
        task.start()
        await asyncio.sleep(0.08)
        client.is_closed = True
        await asyncio.sleep(0.12)
        assert not task.running
        await task.stop()

    @pytest.mark.asyncio
    async def test_http11_only_pool_is_harmless(self):
        """Non-tunnel/HTTP-1.1 setups must be unaffected -- no crash, no pings."""
        client = FakeClient([FakePoolConnection(FakeHTTP11Connection())])
        task = H2KeepaliveTask(client, interval=0.05)
        task.start()
        try:
            await asyncio.sleep(0.12)
            assert task.running, "loop died on an HTTP/1.1 pool"
        finally:
            await task.stop()

    @pytest.mark.asyncio
    async def test_stop_without_start_is_safe(self):
        task = H2KeepaliveTask(FakeClient([]), interval=0.05)
        await task.stop()
        assert not task.running

    def test_start_without_event_loop_is_a_noop(self):
        """Sync contexts (some CLI paths) have no loop; keepalive just skips."""
        task = H2KeepaliveTask(FakeClient([]), interval=0.05)
        task.start()
        assert not task.running

    @pytest.mark.asyncio
    async def test_one_dead_connection_does_not_stop_the_others(self):
        dead = FakeH2Connection(fail_write=True)
        alive = FakeH2Connection()
        client = FakeClient([FakePoolConnection(dead), FakePoolConnection(alive)])
        task = H2KeepaliveTask(client, interval=0.05)
        task.start()
        try:
            await asyncio.sleep(0.12)
        finally:
            await task.stop()
        assert len(alive._network_stream.writes) >= 1
