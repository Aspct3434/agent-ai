"""CLI for Codex OAuth sign-in: ``python -m auth <login|status|logout>``.

Run with the src dir importable, e.g.::

    PYTHONPATH=src python -m auth login
"""
from __future__ import annotations

import sys

from auth.oauth import CodexOAuth, run_local_login


def main(argv: list[str] | None = None) -> int:
    args = argv if argv is not None else sys.argv[1:]
    cmd = args[0] if args else "status"
    mgr = CodexOAuth()

    if cmd == "login":
        try:
            tokens = run_local_login(mgr, open_browser="--no-browser" not in args)
        except Exception as exc:
            print(f"Sign-in failed: {exc}", file=sys.stderr)
            return 1
        print(f"Signed in{(' as ' + tokens.email) if tokens.email else ''}. "
              f"OPENAI_API_KEY is now set from your account.")
        return 0

    if cmd == "logout":
        mgr.logout()
        print("Signed out; stored Codex tokens cleared.")
        return 0

    if cmd == "status":
        st = mgr.status()
        if st["signed_in"]:
            print(f"Signed in via OAuth{(' as ' + st['email']) if st['email'] else ''}.")
        elif st["method"] == "api_key":
            print("Not signed in via OAuth; using an API key from the environment.")
        else:
            print("Not authenticated. Run `python -m auth login`.")
        return 0

    print(f"Unknown command {cmd!r}. Use: login | status | logout", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
