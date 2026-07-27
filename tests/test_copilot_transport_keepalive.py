"""Keepalive lifecycle contracts for CopilotTransport.

The keepalive task must be bound to the lifetime of the pooled HTTP/2 client:
started when a client is created, stopped whenever that client goes away.
Leaking a task past its client would keep pinging a dead pool forever.
"""

from __future__ import annotations

import asyncio

import pytest

from router_maestro.providers.copilot_support.auth_session import CopilotAuthSession
from router_maestro.providers.copilot_support.transport import CopilotTransport
from router_maestro.providers.h2_keepalive import KEEPALIVE_INTERVAL


def _transport() -> CopilotTransport:
    auth = CopilotAuthSession.__new__(CopilotAuthSession)
    auth.cached_token = "tok"
    auth.api_base = "https://api.example.invalid"
    return CopilotTransport(auth)


class TestKeepaliveLifecycle:
    @pytest.mark.asyncio
    async def test_starts_with_the_client(self):
        transport = _transport()
        try:
            transport.get_client()
            assert transport.keepalive is not None
            assert transport.keepalive.running
        finally:
            await transport.close()

    @pytest.mark.asyncio
    async def test_close_stops_the_task(self):
        transport = _transport()
        transport.get_client()
        keepalive = transport.keepalive

        await transport.close()

        assert transport.keepalive is None
        assert not keepalive.running

    @pytest.mark.asyncio
    async def test_recycle_stops_the_old_task_and_starts_a_new_one(self):
        transport = _transport()
        transport.get_client()
        first = transport.keepalive

        await transport.recycle_client()
        assert not first.running, "old keepalive outlived its client"

        try:
            transport.get_client()
            assert transport.keepalive is not None
            assert transport.keepalive is not first
            assert transport.keepalive.running
        finally:
            await transport.close()

    @pytest.mark.asyncio
    async def test_age_based_recycle_replaces_the_task(self):
        """The 300s client rotation must not leave a task pinging a closed pool."""
        transport = _transport()
        transport.get_client()
        first = transport.keepalive

        # Force the age check to trip on the next get_client().
        transport.client_created_at -= transport.client_max_age + 1
        transport.get_client()
        try:
            assert transport.keepalive is not first
            assert transport.keepalive.running
            await asyncio.sleep(0)  # let the deferred stop() run
            assert not first.running
        finally:
            await transport.close()

    @pytest.mark.asyncio
    async def test_uses_the_module_interval(self):
        transport = _transport()
        try:
            transport.get_client()
            assert transport.keepalive._interval == KEEPALIVE_INTERVAL
        finally:
            await transport.close()

    def test_interval_stays_below_observed_reclaim_windows(self):
        """Pings are useless if they fire slower than the intermediary's timer.

        Observed idle-reclaim windows on affected paths start around 240s, so
        the interval needs real headroom below that -- not merely to be under it.
        """
        assert 0 < KEEPALIVE_INTERVAL <= 120.0

    @pytest.mark.asyncio
    async def test_close_without_client_is_safe(self):
        transport = _transport()
        await transport.close()
        assert transport.keepalive is None

    @pytest.mark.asyncio
    async def test_client_is_still_http2(self):
        """Keepalive is worthless if the client stops negotiating HTTP/2."""
        transport = _transport()
        try:
            client = transport.get_client()
            # httpx exposes the negotiated pool; h2 must be in the ALPN set.
            pool = client._transport._pool
            assert pool._http2 is True
        finally:
            await transport.close()
