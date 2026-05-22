from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from tools import ToolManager


def run_publish_static_site_test() -> None:
    temp_dir = Path(tempfile.mkdtemp(prefix="publish-static-site-test-"))
    old_published_dir = os.environ.get("PUBLISHED_SITES_DIR")
    old_public_base = os.environ.get("PUBLIC_BASE_URL")

    try:
        source = temp_dir / "egypt-football"
        source.mkdir()
        (source / "index.html").write_text(
            "<h1>Egypt Football</h1>",
            encoding="utf-8",
        )
        (source / "styles.css").write_text(
            "body { font-family: sans-serif; }",
            encoding="utf-8",
        )

        published_root = temp_dir / "published"
        os.environ["PUBLISHED_SITES_DIR"] = str(published_root)
        os.environ["PUBLIC_BASE_URL"] = "http://localhost:8000"

        manager = ToolManager()
        result = json.loads(
            manager.publish_static_site(str(source), slug="egypt-football-history")
        )

        assert result["published"] is True
        assert result["index_exists"] is True
        assert result["url"] == "http://localhost:8000/sites/egypt-football-history/"
        assert (published_root / "egypt-football-history" / "index.html").is_file()
        assert "styles.css" in result["files"]

        print(json.dumps(result, indent=2))
        print("PUBLISH STATIC SITE CHECKS PASSED")
    finally:
        if old_published_dir is None:
            os.environ.pop("PUBLISHED_SITES_DIR", None)
        else:
            os.environ["PUBLISHED_SITES_DIR"] = old_published_dir

        if old_public_base is None:
            os.environ.pop("PUBLIC_BASE_URL", None)
        else:
            os.environ["PUBLIC_BASE_URL"] = old_public_base

        shutil.rmtree(temp_dir, ignore_errors=True)


def test_publish_static_site_returns_backend_url() -> None:
    run_publish_static_site_test()


if __name__ == "__main__":
    run_publish_static_site_test()
