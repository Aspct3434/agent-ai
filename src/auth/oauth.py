"""OpenAI / Codex OAuth 2.0 (PKCE) sign-in.

Lets a user authenticate with their ChatGPT/OpenAI account instead of pasting
an API key. The flow mirrors the public Codex CLI client:

  1. PKCE authorization-code request to ``auth.openai.com``.
  2. Browser redirects to ``http://localhost:1455/auth/callback`` with a code.
  3. Exchange code -> {access_token, refresh_token, id_token}.
  4. Token-exchange the id_token -> a usable OpenAI API key (``sk-…``).
  5. Persist the tokens and inject the key into ``OPENAI_API_KEY`` so LiteLLM
     uses it transparently. Refresh re-mints the key when it expires.

All endpoint/client values are overridable via env so the same machinery works
for a self-registered OAuth app. Token requests use synchronous httpx so this
module can be driven from both the CLI and (via a worker thread) the gateway.

NOTE: OpenAI's hosted login could not be exercised end-to-end in development;
if the live token-exchange shape differs, only ``_token_request`` payloads
need adjusting — the surrounding flow is standard OAuth 2.0 + PKCE.
"""
from __future__ import annotations

import base64
import hashlib
import json
import logging
import os
import secrets
import threading
import time
import webbrowser
from dataclasses import asdict, dataclass, fields
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlencode, urlparse

import httpx

logger = logging.getLogger(__name__)

# ── Provider config (public Codex client; all overridable) ───────────────────
CLIENT_ID = os.getenv("CODEX_CLIENT_ID", "app_EMoamEEZ73f0CkXaXp7hrann")
ISSUER = os.getenv("CODEX_OAUTH_ISSUER", "https://auth.openai.com").rstrip("/")
AUTH_URL = f"{ISSUER}/oauth/authorize"
TOKEN_URL = f"{ISSUER}/oauth/token"
REDIRECT_URI = os.getenv("CODEX_REDIRECT_URI", "http://localhost:1455/auth/callback")
SCOPE = os.getenv("CODEX_OAUTH_SCOPE", "openid profile email offline_access")
API_KEY_ENV = os.getenv("CODEX_API_KEY_ENV", "OPENAI_API_KEY")
_TOKEN_EXCHANGE_GRANT = "urn:ietf:params:oauth:grant-type:token-exchange"


def _callback_port(uri: str) -> int:
    try:
        return urlparse(uri).port or 1455
    except ValueError:
        return 1455


CALLBACK_PORT = _callback_port(REDIRECT_URI)


def _pkce() -> tuple[str, str]:
    """Return a (verifier, S256 challenge) PKCE pair."""
    verifier = base64.urlsafe_b64encode(secrets.token_bytes(64)).rstrip(b"=").decode()
    digest = hashlib.sha256(verifier.encode()).digest()
    challenge = base64.urlsafe_b64encode(digest).rstrip(b"=").decode()
    return verifier, challenge


def _decode_jwt_claims(token: str) -> dict[str, Any]:
    """Read (unverified) JWT payload claims — only for display (email/account)."""
    try:
        payload = token.split(".")[1]
        payload += "=" * (-len(payload) % 4)
        data = json.loads(base64.urlsafe_b64decode(payload))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


@dataclass
class TokenSet:
    api_key: str = ""
    access_token: str = ""
    refresh_token: str = ""
    id_token: str = ""
    account_id: str = ""
    email: str = ""
    obtained_at: float = 0.0
    expires_in: int = 0

    def expired(self) -> bool:
        if not self.expires_in:
            return False
        return time.time() > self.obtained_at + self.expires_in - 60


class CodexOAuth:
    """Manages the Codex OAuth token lifecycle + persistence + env injection."""

    def __init__(self, store_path: str | Path | None = None) -> None:
        default = Path.home() / ".agent-ai" / "codex_auth.json"
        self._path = Path(store_path or os.getenv("AGENT_AUTH_FILE") or default)
        self._pending: dict[str, str] = {}  # state -> code_verifier
        self._lock = threading.Lock()
        self._tokens = self._load()
        if self._tokens.api_key:
            self._inject()

    # ── persistence ──────────────────────────────────────────────────────
    def _load(self) -> TokenSet:
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
            valid = {f.name for f in fields(TokenSet)}
            return TokenSet(**{k: v for k, v in data.items() if k in valid})
        except Exception:
            return TokenSet()

    def _save(self) -> None:
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._path.write_text(json.dumps(asdict(self._tokens), indent=2), encoding="utf-8")
        except OSError as exc:
            logger.warning("Could not persist OAuth tokens: %s", exc)

    def _inject(self) -> None:
        if self._tokens.api_key:
            os.environ[API_KEY_ENV] = self._tokens.api_key

    # ── flow ─────────────────────────────────────────────────────────────
    def authorize_url(self) -> str:
        """Build the authorize URL and remember its PKCE verifier by state."""
        verifier, challenge = _pkce()
        state = secrets.token_urlsafe(24)
        with self._lock:
            self._pending[state] = verifier
        params = {
            "response_type": "code",
            "client_id": CLIENT_ID,
            "redirect_uri": REDIRECT_URI,
            "scope": SCOPE,
            "code_challenge": challenge,
            "code_challenge_method": "S256",
            "id_token_add_organizations": "true",
            "state": state,
        }
        return f"{AUTH_URL}?{urlencode(params)}"

    def _token_request(self, data: dict[str, str]) -> dict[str, Any]:
        resp = httpx.post(TOKEN_URL, data=data, timeout=30.0)
        resp.raise_for_status()
        body: dict[str, Any] = resp.json()
        return body

    def _mint_api_key(self, id_token: str) -> str:
        body = self._token_request({
            "grant_type": _TOKEN_EXCHANGE_GRANT,
            "client_id": CLIENT_ID,
            "requested_token": "openai-api-key",
            "subject_token": id_token,
            "subject_token_type": "urn:ietf:params:oauth:token-type:id_token",
        })
        return str(body.get("access_token", ""))

    def complete(self, code: str, state: str) -> TokenSet:
        """Exchange an auth code for tokens, mint an API key, persist + inject."""
        with self._lock:
            verifier = self._pending.pop(state, None)
        if verifier is None:
            raise ValueError("Unknown or expired OAuth state")

        body = self._token_request({
            "grant_type": "authorization_code",
            "code": code,
            "redirect_uri": REDIRECT_URI,
            "client_id": CLIENT_ID,
            "code_verifier": verifier,
        })
        claims = _decode_jwt_claims(str(body.get("id_token", "")))
        ts = TokenSet(
            access_token=str(body.get("access_token", "")),
            refresh_token=str(body.get("refresh_token", "")),
            id_token=str(body.get("id_token", "")),
            email=str(claims.get("email", "")),
            account_id=str(claims.get("sub", "")),
            obtained_at=time.time(),
            expires_in=int(body.get("expires_in", 0) or 0),
        )
        ts.api_key = self._mint_api_key(ts.id_token) if ts.id_token else ts.access_token
        self._tokens = ts
        self._save()
        self._inject()
        logger.info("Codex OAuth sign-in complete (%s)", ts.email or "account")
        return ts

    def refresh(self) -> bool:
        if not self._tokens.refresh_token:
            return False
        try:
            body = self._token_request({
                "grant_type": "refresh_token",
                "client_id": CLIENT_ID,
                "refresh_token": self._tokens.refresh_token,
                "scope": SCOPE,
            })
        except Exception as exc:
            logger.warning("Codex OAuth refresh failed: %s", exc)
            return False
        self._tokens.access_token = str(body.get("access_token", self._tokens.access_token))
        if body.get("refresh_token"):
            self._tokens.refresh_token = str(body["refresh_token"])
        if body.get("id_token"):
            self._tokens.id_token = str(body["id_token"])
        self._tokens.obtained_at = time.time()
        self._tokens.expires_in = int(body.get("expires_in", 0) or 0)
        if self._tokens.id_token:
            try:
                self._tokens.api_key = self._mint_api_key(self._tokens.id_token)
            except Exception as exc:
                logger.warning("Codex API-key re-mint failed: %s", exc)
        self._save()
        self._inject()
        return True

    def ensure_fresh(self) -> None:
        if self._tokens.api_key and self._tokens.expired():
            self.refresh()

    def status(self) -> dict[str, Any]:
        signed_in = bool(self._tokens.api_key)
        method = "oauth" if signed_in else ("api_key" if os.getenv(API_KEY_ENV) else "none")
        expires_at = (
            self._tokens.obtained_at + self._tokens.expires_in
            if self._tokens.expires_in
            else None
        )
        return {
            "method": method,
            "signed_in": signed_in,
            "email": self._tokens.email,
            "account_id": self._tokens.account_id,
            "expires_at": expires_at,
            "client_id": CLIENT_ID,
        }

    def logout(self) -> None:
        old_key = self._tokens.api_key
        self._tokens = TokenSet()
        try:
            self._path.unlink(missing_ok=True)
        except OSError:
            pass
        # Only clear the env var if it holds the key we injected.
        if old_key and os.environ.get(API_KEY_ENV) == old_key:
            os.environ.pop(API_KEY_ENV, None)


# ── Local redirect listener (binds 127.0.0.1:CALLBACK_PORT for one request) ──


def wait_for_callback(timeout: float = 300.0) -> dict[str, str]:
    """Block until the OAuth provider redirects to the local callback.

    Returns ``{"code", "state", "error"}``. Raises ``TimeoutError`` if nothing
    arrives in time.
    """
    holder: dict[str, str] = {}

    class _Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args: Any) -> None:  # silence stdlib logging
            pass

        def do_GET(self) -> None:
            q = parse_qs(urlparse(self.path).query)
            holder["code"] = (q.get("code") or [""])[0]
            holder["state"] = (q.get("state") or [""])[0]
            holder["error"] = (q.get("error") or [""])[0]
            ok = not holder["error"] and holder["code"]
            msg = (
                "Signed in. You can close this tab and return to agent-ai."
                if ok
                else "Sign-in failed. Close this tab and try again."
            )
            page = (
                "<html><body style=\"font-family:system-ui;background:#09090b;"
                "color:#e4e4e7;display:flex;align-items:center;justify-content:center;"
                f"height:100vh\"><h2>{msg}</h2></body></html>"
            )
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(page.encode("utf-8"))

    server = HTTPServer(("127.0.0.1", CALLBACK_PORT), _Handler)
    server.timeout = timeout
    try:
        server.handle_request()  # serve exactly one request (the callback)
    finally:
        server.server_close()
    if not holder.get("code") and not holder.get("error"):
        raise TimeoutError("No OAuth callback received (timed out or cancelled)")
    return holder


def run_local_login(
    manager: CodexOAuth, *, open_browser: bool = True, timeout: float = 300.0
) -> TokenSet:
    """Full local login: open the browser, await the callback, complete exchange."""
    url = manager.authorize_url()
    if open_browser:
        try:
            webbrowser.open(url)
        except Exception:
            pass
    else:
        print(f"Open this URL to sign in:\n{url}\n")
    result = wait_for_callback(timeout=timeout)
    if result.get("error"):
        raise RuntimeError(f"OAuth error: {result['error']}")
    return manager.complete(result["code"], result["state"])
