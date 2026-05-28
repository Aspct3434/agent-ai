from __future__ import annotations

import hmac
import os
import secrets
import time
from hashlib import sha256
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

_PROCESS_PROXY_SECRET = secrets.token_urlsafe(32)
_DEFAULT_TTL_SECONDS = max(60, int(os.getenv("AGENT_PROXY_URL_TTL_SECONDS", "86400")))

PROXY_EXPIRES_PARAM = "proxy_expires"
PROXY_TOKEN_PARAM = "proxy_token"


def _proxy_secret() -> str:
    """Return the signing secret shared by gateway and tool URL generation."""
    return (
        os.getenv("AGENT_PROXY_SIGNING_SECRET", "").strip()
        or os.getenv("AGENT_API_TOKEN", "").strip()
        or _PROCESS_PROXY_SECRET
    )


def _message(port: int, expires: int) -> bytes:
    return f"proxy-v1:{int(port)}:{int(expires)}".encode()


def sign_proxy_token(port: int, expires: int) -> str:
    return hmac.new(
        _proxy_secret().encode("utf-8"),
        _message(port, expires),
        sha256,
    ).hexdigest()


def verify_proxy_token(port: int, expires: str | int | None, token: str | None) -> bool:
    if not expires or not token:
        return False
    try:
        expires_int = int(expires)
    except (TypeError, ValueError):
        return False
    if expires_int < int(time.time()):
        return False
    expected = sign_proxy_token(port, expires_int)
    return hmac.compare_digest(str(token), expected)


def proxy_cookie_name(port: int) -> str:
    return f"agent_proxy_{int(port)}"


def proxy_cookie_value(port: int, expires: int | None = None) -> str:
    expires_int = int(expires or (time.time() + _DEFAULT_TTL_SECONDS))
    return f"{expires_int}:{sign_proxy_token(port, expires_int)}"


def verify_proxy_cookie(port: int, cookie_value: str | None) -> bool:
    if not cookie_value or ":" not in cookie_value:
        return False
    expires, token = cookie_value.split(":", 1)
    return verify_proxy_token(port, expires, token)


def signed_proxy_query(port: int, expires: int | None = None) -> str:
    expires_int = int(expires or (time.time() + _DEFAULT_TTL_SECONDS))
    return urlencode(
        {
            PROXY_EXPIRES_PARAM: str(expires_int),
            PROXY_TOKEN_PARAM: sign_proxy_token(port, expires_int),
        }
    )


def append_signed_proxy_query(url: str, port: int) -> str:
    split = urlsplit(url)
    query = parse_qsl(split.query, keep_blank_values=True)
    query.extend(parse_qsl(signed_proxy_query(port), keep_blank_values=True))
    return urlunsplit((split.scheme, split.netloc, split.path, urlencode(query), split.fragment))


def strip_proxy_auth_params(query: str) -> str:
    pairs = [
        (key, value)
        for key, value in parse_qsl(query, keep_blank_values=True)
        if key not in {PROXY_EXPIRES_PARAM, PROXY_TOKEN_PARAM}
    ]
    return urlencode(pairs)
