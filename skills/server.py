from __future__ import annotations

import importlib.util
import logging
import sys
from pathlib import Path

import anyio
from mcp.server.fastmcp import FastMCP

SKILLS_DIR = Path(__file__).parent
# Allow skills to import the _skill decorator via `from _skill import skill`
sys.path.insert(0, str(SKILLS_DIR))

logger = logging.getLogger(__name__)
mcp = FastMCP("skills")


def _load_skills() -> None:
    for skill_path in sorted(SKILLS_DIR.glob("*.py")):
        if skill_path.name.startswith("_") or skill_path.name == "server.py":
            continue
        try:
            spec = importlib.util.spec_from_file_location(
                f"_skills.{skill_path.stem}", skill_path
            )
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
        except Exception:
            logger.exception("Failed to load skill module: %s", skill_path.name)
            continue

        for attr_name in dir(module):
            obj = getattr(module, attr_name)
            if callable(obj) and getattr(obj, "_is_skill", False):
                mcp.add_tool(
                    obj,
                    name=getattr(obj, "_skill_name", attr_name),
                    description=getattr(obj, "_skill_description", None),
                )
                logger.info("Registered skill: %s (from %s)", attr_name, skill_path.name)


_load_skills()

if __name__ == "__main__":
    anyio.run(mcp.run_stdio_async)
