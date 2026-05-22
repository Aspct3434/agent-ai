from __future__ import annotations

import functools
import http.server
import json
import os
import socketserver
import sys
import tempfile
import threading
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from tools import ToolManager  # noqa: E402


class _ReusableTCPServer(socketserver.TCPServer):
    allow_reuse_address = True


def run_expose_local_http_service_test() -> None:
    old_public_base = os.environ.get("PUBLIC_BASE_URL")
    try:
        os.environ["PUBLIC_BASE_URL"] = "http://localhost:8000"
        with tempfile.TemporaryDirectory(prefix="expose-http-service-test-") as td:
            root = Path(td)
            (root / "index.html").write_text("service ok", encoding="utf-8")
            handler = functools.partial(
                http.server.SimpleHTTPRequestHandler,
                directory=str(root),
            )
            with _ReusableTCPServer(("127.0.0.1", 0), handler) as server:
                port = int(server.server_address[1])
                thread = threading.Thread(target=server.serve_forever, daemon=True)
                thread.start()

                manager = ToolManager()
                result = json.loads(
                    manager.expose_local_http_service(port, name="test-service")
                )

                assert result["exposed"] is True
                assert result["connectable"] is True
                assert result["port"] == port
                assert result["url"] == f"http://localhost:8000/proxy/{port}/"
                print(json.dumps(result, indent=2))
                print("EXPOSE LOCAL HTTP SERVICE CHECKS PASSED")

                server.shutdown()
                thread.join(timeout=5)
    finally:
        if old_public_base is None:
            os.environ.pop("PUBLIC_BASE_URL", None)
        else:
            os.environ["PUBLIC_BASE_URL"] = old_public_base


def test_expose_local_http_service_returns_proxy_url() -> None:
    run_expose_local_http_service_test()


if __name__ == "__main__":
    run_expose_local_http_service_test()
