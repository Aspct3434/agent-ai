"""Tests for gateway API authentication (AGENT_API_TOKEN) and CORS config.

These exercise the auth helpers and the HTTP middleware directly, without
booting the full app lifespan (which needs MCP servers / DBs), mirroring the
import-only style of test_gateway_unit.py. ``_API_TOKEN`` is a module global the
middleware reads at request time, so monkeypatching it flips auth on/off.
"""
from __future__ import annotations

import sys
from pathlib import Path
from urllib.parse import parse_qs

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pytest
from starlette.requests import Request
from starlette.responses import PlainTextResponse

import gateway
from proxy_auth import signed_proxy_query


def _make_request(
    path: str = "/api/status",
    method: str = "GET",
    headers: dict | None = None,
    query: str = "",
    client: tuple[str, int] = ("203.0.113.10", 4242),
):
    raw = [(k.lower().encode(), v.encode()) for k, v in (headers or {}).items()]
    scope = {
        "type": "http",
        "method": method,
        "path": path,
        "headers": raw,
        "query_string": query.encode(),
        "client": client,
    }
    return Request(scope)


class _FakeWS:
    def __init__(self, query: dict | None = None, headers: dict | None = None):
        self.query_params = query or {}
        self.headers = headers or {}


# ---------------------------------------------------------------------------
# Token extraction
# ---------------------------------------------------------------------------

class TestTokenExtraction:
    def test_bearer_header(self):
        assert gateway._bearer_or_api_key("Bearer abc123", None) == "abc123"

    def test_bearer_is_case_insensitive_on_scheme(self):
        assert gateway._bearer_or_api_key("bearer abc123", None) == "abc123"

    def test_x_api_key_header(self):
        assert gateway._bearer_or_api_key(None, "  k-99 ") == "k-99"

    def test_bearer_wins_over_api_key(self):
        assert gateway._bearer_or_api_key("Bearer fromauth", "fromkey") == "fromauth"

    def test_none_when_absent(self):
        assert gateway._bearer_or_api_key(None, None) is None

    def test_http_token_reads_request_headers(self):
        req = _make_request(headers={"Authorization": "Bearer xyz"})
        assert gateway._http_token(req) == "xyz"

    def test_ws_token_prefers_query_param(self):
        ws = _FakeWS(query={"token": "q-tok"}, headers={"authorization": "Bearer h-tok"})
        assert gateway._ws_token(ws) == "q-tok"

    def test_ws_token_falls_back_to_header(self):
        ws = _FakeWS(headers={"x-api-key": "h-tok"})
        assert gateway._ws_token(ws) == "h-tok"


# ---------------------------------------------------------------------------
# _token_ok
# ---------------------------------------------------------------------------

class TestTokenOk:
    def test_missing_configured_token_never_authenticates(self, monkeypatch):
        monkeypatch.setattr(gateway, "_API_TOKEN", "")
        assert gateway._token_ok(None) is False
        assert gateway._token_ok("anything") is False

    def test_correct_token_accepted(self, monkeypatch):
        monkeypatch.setattr(gateway, "_API_TOKEN", "secret")
        assert gateway._token_ok("secret") is True

    def test_wrong_token_rejected(self, monkeypatch):
        monkeypatch.setattr(gateway, "_API_TOKEN", "secret")
        assert gateway._token_ok("nope") is False

    def test_missing_token_rejected_when_required(self, monkeypatch):
        monkeypatch.setattr(gateway, "_API_TOKEN", "secret")
        assert gateway._token_ok(None) is False


# ---------------------------------------------------------------------------
# HTTP middleware
# ---------------------------------------------------------------------------

class TestAuthMiddleware:
    @pytest.mark.asyncio
    async def test_blocks_when_no_token_configured(self, monkeypatch):
        monkeypatch.setattr(gateway, "_API_TOKEN", "")
        monkeypatch.setattr(gateway, "_ALLOW_INSECURE_NO_AUTH", False)

        async def call_next(_req):  # pragma: no cover - must not be reached
            raise AssertionError("call_next should not run without auth config")

        resp = await gateway._auth_middleware(_make_request(), call_next)
        assert resp.status_code == 503

    @pytest.mark.asyncio
    async def test_explicit_insecure_mode_allows_no_token(self, monkeypatch):
        monkeypatch.setattr(gateway, "_API_TOKEN", "")
        monkeypatch.setattr(gateway, "_ALLOW_INSECURE_NO_AUTH", True)

        async def call_next(_req):
            return PlainTextResponse("ok")

        resp = await gateway._auth_middleware(_make_request(), call_next)
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_health_always_open(self, monkeypatch):
        monkeypatch.setattr(gateway, "_API_TOKEN", "secret")
        monkeypatch.setattr(gateway, "_ALLOW_INSECURE_NO_AUTH", False)

        async def call_next(_req):
            return PlainTextResponse("ok")

        resp = await gateway._auth_middleware(_make_request(path="/health"), call_next)
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_options_preflight_bypasses_auth(self, monkeypatch):
        monkeypatch.setattr(gateway, "_API_TOKEN", "secret")
        monkeypatch.setattr(gateway, "_ALLOW_INSECURE_NO_AUTH", False)

        async def call_next(_req):
            return PlainTextResponse("ok")

        resp = await gateway._auth_middleware(
            _make_request(method="OPTIONS"), call_next
        )
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_missing_token_returns_401(self, monkeypatch):
        monkeypatch.setattr(gateway, "_API_TOKEN", "secret")
        monkeypatch.setattr(gateway, "_ALLOW_INSECURE_NO_AUTH", False)

        async def call_next(_req):  # pragma: no cover - must not be reached
            raise AssertionError("call_next should not run for unauthorized request")

        resp = await gateway._auth_middleware(_make_request(), call_next)
        assert resp.status_code == 401
        assert resp.headers.get("www-authenticate") == "Bearer"

    @pytest.mark.asyncio
    async def test_valid_token_passes(self, monkeypatch):
        monkeypatch.setattr(gateway, "_API_TOKEN", "secret")
        monkeypatch.setattr(gateway, "_ALLOW_INSECURE_NO_AUTH", False)
        called = {"hit": False}

        async def call_next(_req):
            called["hit"] = True
            return PlainTextResponse("ok")

        resp = await gateway._auth_middleware(
            _make_request(headers={"Authorization": "Bearer secret"}), call_next
        )
        assert resp.status_code == 200
        assert called["hit"] is True

    @pytest.mark.asyncio
    async def test_signed_proxy_url_bypasses_api_token_without_exposing_api(self, monkeypatch):
        monkeypatch.setattr(gateway, "_API_TOKEN", "secret")
        monkeypatch.setattr(gateway, "_ALLOW_INSECURE_NO_AUTH", False)
        query = signed_proxy_query(4321)
        called = {"hit": False}

        async def call_next(_req):
            called["hit"] = True
            return PlainTextResponse("ok")

        resp = await gateway._auth_middleware(
            _make_request(path="/proxy/4321/", query=query),
            call_next,
        )

        parsed = parse_qs(query)
        assert resp.status_code == 200
        assert called["hit"] is True
        assert "agent_proxy_4321=" in resp.headers.get("set-cookie", "")
        assert parsed["proxy_token"][0] != "secret"

        cookie = resp.headers["set-cookie"].split(";", 1)[0]
        called["hit"] = False
        asset_resp = await gateway._auth_middleware(
            _make_request(
                path="/proxy/4321/assets/app.css",
                headers={"Cookie": cookie},
            ),
            call_next,
        )
        assert asset_resp.status_code == 200
        assert called["hit"] is True

    @pytest.mark.asyncio
    async def test_signed_proxy_url_is_bound_to_port(self, monkeypatch):
        monkeypatch.setattr(gateway, "_API_TOKEN", "secret")
        monkeypatch.setattr(gateway, "_ALLOW_INSECURE_NO_AUTH", False)
        query = signed_proxy_query(4321)

        async def call_next(_req):  # pragma: no cover
            raise AssertionError("wrong-port proxy signature must not authorize")

        resp = await gateway._auth_middleware(
            _make_request(path="/proxy/4322/", query=query),
            call_next,
        )

        assert resp.status_code == 401

    @pytest.mark.asyncio
    async def test_wrong_token_returns_401(self, monkeypatch):
        monkeypatch.setattr(gateway, "_API_TOKEN", "secret")
        monkeypatch.setattr(gateway, "_ALLOW_INSECURE_NO_AUTH", False)

        async def call_next(_req):  # pragma: no cover
            raise AssertionError("unauthorized request must not reach the route")

        resp = await gateway._auth_middleware(
            _make_request(headers={"X-API-Key": "wrong"}), call_next
        )
        assert resp.status_code == 401


# ---------------------------------------------------------------------------
# Rate limit identity
# ---------------------------------------------------------------------------

class TestRateLimitIdentity:
    def test_http_key_uses_token_not_session(self):
        req = _make_request(headers={"Authorization": "Bearer shared-secret"})
        assert gateway._rate_limit_key_for_http(req).startswith("token:")

    def test_http_key_falls_back_to_client_ip(self):
        req = _make_request(client=("198.51.100.23", 5000))
        assert gateway._rate_limit_key_for_http(req) == "ip:198.51.100.23"


# ---------------------------------------------------------------------------
# CORS origin resolution
# ---------------------------------------------------------------------------

class TestCorsOrigins:
    def test_default_is_localhost_only(self, monkeypatch):
        monkeypatch.delenv("AGENT_CORS_ORIGINS", raising=False)
        origins = gateway._cors_origins()
        assert "http://localhost:5173" in origins
        assert "*" not in origins

    def test_wildcard(self, monkeypatch):
        monkeypatch.setenv("AGENT_CORS_ORIGINS", "*")
        assert gateway._cors_origins() == ["*"]

    def test_explicit_list(self, monkeypatch):
        monkeypatch.setenv("AGENT_CORS_ORIGINS", "https://a.example, https://b.example")
        assert gateway._cors_origins() == ["https://a.example", "https://b.example"]
