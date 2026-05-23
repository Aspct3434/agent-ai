"""Regression tests for the recurring "build a React app" failure.

The failure mode: a user asks for an interactive React (Vite/Next/Vue) site,
Node.js is not on PATH, the non-root agent cannot apt-get install it, and the
agent then (a) burns iterations retrying apt-get and (b) silently substitutes a
different stack (e.g. a Flask page) while claiming the React task is done.

The permanent fix has four parts, validated here:

  1. Node + npm + npx are baked into the Docker image  → Dockerfile (structural)
  2. node/npm/npx appear in tools._PROBED_RUNTIMES       → the env tool reports them
  3. get_system_environment surfaces them in 'runtimes'  → snapshot check
  4. SYSTEM_DIRECTIVE forbids silent stack substitution  → directive check
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(_SRC))

from tools import (  # noqa: E402
    _PROBED_RUNTIMES,
    GET_SYSTEM_ENVIRONMENT_TOOL,
    _collect_system_environment,
)

# ---------------------------------------------------------------------------
# Part 2 — the Node toolchain must be probed by get_system_environment
# ---------------------------------------------------------------------------


class TestNodeToolchainProbed:
    REQUIRED = {"node", "npm", "npx"}

    def test_node_toolchain_names_present(self):
        missing = self.REQUIRED - set(_PROBED_RUNTIMES)
        assert not missing, (
            f"These Node toolchain names are missing from _PROBED_RUNTIMES: {missing}. "
            "Without them the agent can't see node/npm/npx in get_system_environment "
            "and will waste iterations trying to apt-get install a runtime that is "
            "already baked into the image."
        )

    def test_no_duplicates_in_probed_runtimes(self):
        seen: set[str] = set()
        dupes: list[str] = []
        for name in _PROBED_RUNTIMES:
            if name in seen:
                dupes.append(name)
            seen.add(name)
        assert not dupes, f"Duplicate entries in _PROBED_RUNTIMES: {dupes}"


# ---------------------------------------------------------------------------
# Part 3 — get_system_environment surfaces them in the runtimes block
# ---------------------------------------------------------------------------


class TestNodeInEnvironmentSnapshot:
    def test_runtimes_block_includes_node_toolchain(self):
        data = json.loads(_collect_system_environment())
        runtimes = data["runtimes"]
        for name in ("node", "npm", "npx"):
            assert name in runtimes, f"'{name}' missing from runtimes block"
            assert isinstance(runtimes[name], bool)

    def test_runtimes_match_shutil_which(self):
        data = json.loads(_collect_system_environment())
        runtimes = data["runtimes"]
        for name in ("node", "npm", "npx"):
            expected = shutil.which(name) is not None
            assert runtimes[name] == expected, (
                f"runtimes['{name}'] is {runtimes[name]!r} but "
                f"shutil.which('{name}') says {expected!r}"
            )

    def test_non_root_warning_does_not_recommend_nvm(self):
        """The stale advice to install Node via nvm must be gone.

        When the agent is non-root without sudo, the env warning used to tell it
        to use nvm for a per-user Node install — but Node is now pre-installed,
        so that advice sent the agent down a pointless install path.
        """
        data = json.loads(_collect_system_environment())
        warning = data.get("user", {}).get("warning", "")
        if warning:  # only present when non-root AND no sudo
            assert "nvm" not in warning.lower(), (
                "Env warning still recommends nvm; Node is pre-installed now."
            )


# ---------------------------------------------------------------------------
# Part 4 — the tool description mentions the Node toolchain
# ---------------------------------------------------------------------------


class TestEnvToolDescription:
    DESC = GET_SYSTEM_ENVIRONMENT_TOOL["description"].lower()

    def test_description_mentions_node(self):
        assert "node" in self.DESC

    def test_description_mentions_npm_or_npx(self):
        assert "npm" in self.DESC or "npx" in self.DESC


# ---------------------------------------------------------------------------
# Part 5 — SYSTEM_DIRECTIVE forbids silent requirement substitution and states
#          the runtimes are pre-installed
# ---------------------------------------------------------------------------


class TestSystemDirectiveAntiSubstitution:
    @staticmethod
    def _directive() -> str:
        from agent import SYSTEM_DIRECTIVE

        return SYSTEM_DIRECTIVE.lower()

    def test_directive_forbids_silent_substitution(self):
        directive = self._directive()
        assert "substitut" in directive, (
            "SYSTEM_DIRECTIVE must explicitly forbid silently substituting a "
            "different tech stack for the one the user requested."
        )

    def test_directive_mentions_react(self):
        # The canonical example of the failure should be called out by name.
        assert "react" in self._directive()

    def test_directive_states_runtimes_preinstalled(self):
        directive = self._directive()
        assert "pre-installed" in directive or "already installed" in directive or (
            "already" in directive and "installed" in directive
        ), (
            "SYSTEM_DIRECTIVE must tell the model the standard runtimes "
            "(Node/npm, Python, Rust) are already installed so it stops trying "
            "to apt-get install them."
        )

    def test_directive_mentions_node_toolchain(self):
        directive = self._directive()
        assert "npm" in directive or "npx" in directive or "node.js" in directive


# ---------------------------------------------------------------------------
# Part 1 — Dockerfile bakes Node in (structural check; can't run docker here)
# ---------------------------------------------------------------------------


class TestDockerfileInstallsNode:
    @staticmethod
    def _dockerfile_text() -> str:
        path = Path(__file__).resolve().parents[1] / "Dockerfile"
        return path.read_text(encoding="utf-8")

    def test_dockerfile_downloads_node(self):
        text = self._dockerfile_text()
        assert "nodejs.org/dist" in text, (
            "Dockerfile must download the official Node.js binary so node/npm/npx "
            "are available to the non-root agent without an apt-get install."
        )

    def test_dockerfile_verifies_node_install(self):
        text = self._dockerfile_text()
        assert "node --version" in text and "npm --version" in text, (
            "Dockerfile should run 'node --version' and 'npm --version' so a broken "
            "Node install fails the build instead of failing silently at runtime."
        )

    def test_dockerfile_sets_node_version(self):
        text = self._dockerfile_text()
        assert "NODE_VERSION=" in text, (
            "Pin a NODE_VERSION in the Dockerfile for reproducible builds."
        )
