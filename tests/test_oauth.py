"""Unit tests for the Codex OAuth (PKCE) manager. No live network — httpx mocked."""
from __future__ import annotations

import base64
import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch
from urllib.parse import parse_qs, urlparse

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from auth import oauth as oauth_mod
from auth.oauth import CodexOAuth, _decode_jwt_claims, _pkce


def _jwt(claims: dict) -> str:
    payload = base64.urlsafe_b64encode(json.dumps(claims).encode()).rstrip(b"=").decode()
    return f"header.{payload}.sig"


def _resp(payload: dict) -> MagicMock:
    m = MagicMock()
    m.json.return_value = payload
    m.raise_for_status.return_value = None
    return m


@pytest.fixture
def mgr(tmp_path):
    return CodexOAuth(store_path=tmp_path / "auth.json")


class TestPkce:
    def test_pair_is_urlsafe_and_distinct(self) -> None:
        verifier, challenge = _pkce()
        assert verifier and challenge and verifier != challenge
        assert "=" not in verifier and "=" not in challenge

    def test_decode_jwt_claims(self) -> None:
        assert _decode_jwt_claims(_jwt({"email": "a@b.co"}))["email"] == "a@b.co"

    def test_decode_jwt_bad_token(self) -> None:
        assert _decode_jwt_claims("not-a-jwt") == {}


class TestAuthorizeUrl:
    def test_contains_pkce_and_state(self, mgr) -> None:
        url = mgr.authorize_url()
        q = parse_qs(urlparse(url).query)
        assert q["response_type"] == ["code"]
        assert q["code_challenge_method"] == ["S256"]
        assert q["code_challenge"] and q["state"]
        # the state's verifier is remembered for the exchange
        assert q["state"][0] in mgr._pending


class TestComplete:
    def test_exchange_mints_key_and_injects(self, mgr, monkeypatch) -> None:
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        url = mgr.authorize_url()
        state = parse_qs(urlparse(url).query)["state"][0]

        calls: list[dict] = []

        def fake_post(_url, data=None, timeout=None):
            calls.append(data)
            if data["grant_type"] == "authorization_code":
                return _resp({
                    "access_token": "at",
                    "refresh_token": "rt",
                    "id_token": _jwt({"email": "u@x.io", "sub": "acc_1"}),
                    "expires_in": 3600,
                })
            return _resp({"access_token": "sk-minted-key"})  # token-exchange

        with patch.object(oauth_mod.httpx, "post", side_effect=fake_post):
            ts = mgr.complete("the-code", state)

        assert ts.api_key == "sk-minted-key"
        assert ts.email == "u@x.io"
        import os
        assert os.environ["OPENAI_API_KEY"] == "sk-minted-key"
        # persisted
        assert "sk-minted-key" in (mgr._path.read_text())
        # two token requests: code exchange first, then token-exchange
        assert calls[0]["grant_type"] == "authorization_code"
        assert calls[1]["grant_type"].endswith("token-exchange")

    def test_unknown_state_rejected(self, mgr) -> None:
        with pytest.raises(ValueError):
            mgr.complete("code", "never-issued-state")


class TestRefresh:
    def test_refresh_updates_and_remints(self, tmp_path, monkeypatch) -> None:
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        mgr = CodexOAuth(store_path=tmp_path / "a.json")
        mgr._tokens.refresh_token = "rt0"
        mgr._tokens.api_key = "sk-old"

        def fake_post(_url, data=None, timeout=None):
            if data["grant_type"] == "refresh_token":
                return _resp({
                    "access_token": "at2",
                    "refresh_token": "rt1",
                    "id_token": _jwt({"email": "u@x.io"}),
                    "expires_in": 3600,
                })
            return _resp({"access_token": "sk-new"})

        with patch.object(oauth_mod.httpx, "post", side_effect=fake_post):
            assert mgr.refresh() is True
        assert mgr._tokens.refresh_token == "rt1"
        assert mgr._tokens.api_key == "sk-new"

    def test_refresh_without_token_is_noop(self, mgr) -> None:
        assert mgr.refresh() is False


class TestStatusLogout:
    def test_status_none(self, mgr, monkeypatch) -> None:
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        st = mgr.status()
        assert st["method"] == "none" and st["signed_in"] is False

    def test_status_signed_in(self, mgr) -> None:
        mgr._tokens.api_key = "sk-x"
        mgr._tokens.email = "u@x.io"
        st = mgr.status()
        assert st["method"] == "oauth" and st["signed_in"] and st["email"] == "u@x.io"

    def test_logout_clears_tokens_and_injected_env(self, mgr, monkeypatch) -> None:
        import os
        mgr._tokens.api_key = "sk-x"
        os.environ["OPENAI_API_KEY"] = "sk-x"
        mgr._save()
        mgr.logout()
        assert mgr._tokens.api_key == ""
        assert os.environ.get("OPENAI_API_KEY") != "sk-x"
        assert not mgr._path.exists()

    def test_logout_keeps_unrelated_env_key(self, mgr) -> None:
        import os
        mgr._tokens.api_key = "sk-oauth"
        os.environ["OPENAI_API_KEY"] = "sk-users-own-key"
        mgr.logout()
        # we must not clobber a key we didn't set
        assert os.environ.get("OPENAI_API_KEY") == "sk-users-own-key"
        os.environ.pop("OPENAI_API_KEY", None)
