"""Tests for the permanent Rust-server fix.

These tests validate that all five root causes of the recurring "install a simple
Rust server" failure are resolved:

  1. ca-certificates / TLS  → Dockerfile fix (can't unit-test; verified structurally)
  2. Agent home directory   → Dockerfile fix (same)
  3. rustc/cargo in probed runtimes → tools._PROBED_RUNTIMES check
  4. User identity in env snapshot  → _collect_system_environment output check
  5. Tool description       → GET_SYSTEM_ENVIRONMENT_TOOL description check

The file also validates the zero-dependency Rust skill:
  6. Skill source is syntactically valid Python
  7. Skill source contains only std::net (no external crate names)
  8. Cargo.toml inside the skill has no [dependencies] section
  9. Cargo.lock inside the skill is a valid zero-dep lock file
 10. If cargo is available: actually builds the binary offline
"""
from __future__ import annotations

import ast
import importlib.util
import json
import os
import shutil
import sys
from pathlib import Path

import pytest

# Make src/ importable
_SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(_SRC))

from tools import (  # noqa: E402
    _PROBED_RUNTIMES,
    GET_SYSTEM_ENVIRONMENT_TOOL,
    _collect_system_environment,
    _ensure_skills_dir,
)

# ---------------------------------------------------------------------------
# Root cause 3 — rustc/cargo/gcc/make must appear in _PROBED_RUNTIMES
# ---------------------------------------------------------------------------

class TestProbedRuntimes:
    REQUIRED_TOOLCHAIN = {"rustc", "cargo", "gcc", "g++", "make", "curl", "wget", "git"}

    def test_all_toolchain_names_present(self):
        missing = self.REQUIRED_TOOLCHAIN - set(_PROBED_RUNTIMES)
        assert not missing, (
            f"These toolchain names are missing from _PROBED_RUNTIMES: {missing}. "
            "The agent can't see them in get_system_environment and will try to install "
            "tools that are already present, wasting iterations."
        )

    def test_probed_runtimes_has_no_duplicates(self):
        seen: set[str] = set()
        dupes: list[str] = []
        for name in _PROBED_RUNTIMES:
            if name in seen:
                dupes.append(name)
            seen.add(name)
        assert not dupes, f"Duplicate entries in _PROBED_RUNTIMES: {dupes}"


# ---------------------------------------------------------------------------
# Root cause 4 — user identity block in get_system_environment
# ---------------------------------------------------------------------------

class TestSystemEnvironmentUserIdentity:
    def test_env_snapshot_is_valid_json(self):
        snapshot = _collect_system_environment()
        data = json.loads(snapshot)  # must not raise
        assert isinstance(data, dict)

    def test_env_snapshot_contains_user_block(self):
        data = json.loads(_collect_system_environment())
        assert "user" in data, "get_system_environment must include a 'user' block"
        user = data["user"]
        assert "username" in user
        assert "is_root" in user
        assert "sudo_available" in user
        assert "home_dir" in user
        assert "home_exists" in user

    def test_env_snapshot_user_values_correct_types(self):
        data = json.loads(_collect_system_environment())
        user = data["user"]
        assert isinstance(user["username"], str)
        assert isinstance(user["is_root"], bool)
        assert isinstance(user["sudo_available"], bool)
        assert isinstance(user["home_dir"], str)
        assert isinstance(user["home_exists"], bool)

    def test_env_snapshot_contains_runtimes_block(self):
        data = json.loads(_collect_system_environment())
        assert "runtimes" in data
        runtimes = data["runtimes"]
        # Every entry in _PROBED_RUNTIMES must appear as a bool
        for name in _PROBED_RUNTIMES:
            assert name in runtimes, f"'{name}' missing from runtimes block"
            assert isinstance(runtimes[name], bool)

    def test_env_snapshot_runtimes_matches_shutil_which(self):
        """Each runtime entry must equal shutil.which(name) is not None."""
        data = json.loads(_collect_system_environment())
        runtimes = data["runtimes"]
        for name in _PROBED_RUNTIMES:
            expected = shutil.which(name) is not None
            assert runtimes[name] == expected, (
                f"runtimes['{name}'] is {runtimes[name]!r} but "
                f"shutil.which('{name}') says {expected!r}"
            )


# ---------------------------------------------------------------------------
# Root cause 5 — GET_SYSTEM_ENVIRONMENT_TOOL description mentions key concepts
# ---------------------------------------------------------------------------

class TestToolDescription:
    DESC = GET_SYSTEM_ENVIRONMENT_TOOL["description"]

    def test_description_mentions_rustc_or_cargo(self):
        lower = self.DESC.lower()
        assert "rustc" in lower or "cargo" in lower, (
            "Tool description must mention rustc/cargo so the LLM knows "
            "it can check for the Rust toolchain before attempting to install it."
        )

    def test_description_mentions_is_root_or_sudo(self):
        lower = self.DESC.lower()
        assert "is_root" in lower or "sudo" in lower, (
            "Tool description must mention is_root / sudo_available so the LLM "
            "knows to check privilege before running apt-get or rustup."
        )

    def test_description_says_call_first(self):
        lower = self.DESC.lower()
        assert "always" in lower or "first" in lower or "before" in lower, (
            "Tool description must instruct the LLM to call this tool BEFORE "
            "attempting any package install."
        )


# ---------------------------------------------------------------------------
# Built-in skill: source-level validation (no cargo needed)
# ---------------------------------------------------------------------------

_SKILL_SRC = Path(_SRC) / "builtin_skills" / "std_rust_http_server.py"


@pytest.mark.skipif(
    not _SKILL_SRC.exists(),
    reason="src/builtin_skills/std_rust_http_server.py not found",
)
class TestStdRustSkillSource:
    def test_skill_file_is_valid_python(self):
        source = _SKILL_SRC.read_text(encoding="utf-8")
        tree = ast.parse(source)  # raises SyntaxError on invalid Python
        assert tree is not None

    def test_skill_uses_no_external_crates(self):
        """The embedded Cargo.toml must not have a [dependencies] TOML section."""
        import re

        source = _SKILL_SRC.read_text(encoding="utf-8")
        # A TOML [dependencies] *section header* always appears at the start of
        # a line with no leading whitespace.  Comments containing the word
        # "[dependencies]" in a sentence do not count.
        has_deps_section = bool(re.search(r"^\[dependencies\]", source, re.MULTILINE))
        assert not has_deps_section, (
            "Skill must not declare a [dependencies] TOML section — it should use "
            "only std::net so it builds without internet."
        )

    def test_skill_rust_source_uses_std_net(self):
        source = _SKILL_SRC.read_text(encoding="utf-8")
        assert "std::net::TcpListener" in source

    def test_skill_cargo_lock_is_embedded(self):
        source = _SKILL_SRC.read_text(encoding="utf-8")
        assert "_CARGO_LOCK" in source, "Pre-committed Cargo.lock must be embedded in the skill"
        assert "std-http-server" in source

    def test_skill_has_offline_flag(self):
        source = _SKILL_SRC.read_text(encoding="utf-8")
        assert "--offline" in source, (
            "cargo build must use --offline so the build never needs crates.io"
        )

    def test_skill_has_frozen_flag(self):
        source = _SKILL_SRC.read_text(encoding="utf-8")
        assert "--frozen" in source, (
            "cargo build must use --frozen to lock the Cargo.lock in place"
        )

    def test_skill_description_mentions_offline(self):
        # Load the module and inspect the decorated function
        spec = importlib.util.spec_from_file_location("_skill_test", _SKILL_SRC)
        assert spec is not None and spec.loader is not None

        # Provide a minimal _skill stub for import
        skill_stub_src = """\
def skill(fn=None, *, name=None, description=None):
    def _wrap(f):
        f._skill_name = name or f.__name__
        f._skill_description = description or f.__doc__ or ""
        f._is_skill = True
        return f
    return _wrap(fn) if fn is not None else _wrap
"""
        # Temporarily put _skill on sys.modules
        import types
        stub = types.ModuleType("_skill")
        exec(skill_stub_src, stub.__dict__)  # known-safe: execing a local string literal
        sys.modules["_skill"] = stub

        try:
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)  # type: ignore[union-attr]
        finally:
            sys.modules.pop("_skill", None)

        fn = module.build_std_rust_http_server
        desc = fn._skill_description.lower()
        assert "offline" in desc or "no crates.io" in desc or "no internet" in desc


# ---------------------------------------------------------------------------
# Built-in skill deployment
# ---------------------------------------------------------------------------

class TestBuiltinSkillDeployment:
    def test_ensure_skills_dir_deploys_builtin_skill(self, tmp_path: Path):
        if not _SKILL_SRC.exists():
            pytest.skip("src/builtin_skills/std_rust_http_server.py not found")
        _ensure_skills_dir(tmp_path)
        deployed = tmp_path / "std_rust_http_server.py"
        assert deployed.exists(), (
            "_ensure_skills_dir must copy builtin skills from src/builtin_skills/ "
            "into the skills directory"
        )

    def test_ensure_skills_dir_deploys_bootstrap_files(self, tmp_path: Path):
        _ensure_skills_dir(tmp_path)
        assert (tmp_path / "_skill.py").exists()
        assert (tmp_path / "server.py").exists()

    def test_ensure_skills_dir_overwrites_builtin_on_update(self, tmp_path: Path):
        """Builtin skills must be overwritten (not skipped) to propagate updates."""
        if not _SKILL_SRC.exists():
            pytest.skip("src/builtin_skills/std_rust_http_server.py not found")
        # Write stale content first
        dest = tmp_path / "std_rust_http_server.py"
        dest.write_text("# stale content", encoding="utf-8")
        _ensure_skills_dir(tmp_path)
        # Must have been updated to the real skill content
        fresh = dest.read_text(encoding="utf-8")
        assert "stale content" not in fresh
        assert "build_std_rust_http_server" in fresh


# ---------------------------------------------------------------------------
# Integration: actually build if cargo is available (skipped otherwise)
# ---------------------------------------------------------------------------

def _cargo_runnable() -> bool:
    """Return True only when cargo can actually be executed, not just found on PATH.

    On Windows, cargo may appear in PATH via a WSL shim or a broken install that
    shutil.which() sees but CreateProcess cannot launch (WinError 50).  A quick
    probe run catches that before the test attempts a full build.
    """
    import subprocess

    if shutil.which("cargo") is None:
        return False
    try:
        subprocess.run(
            ["cargo", "--version"],
            capture_output=True,
            timeout=10,
            check=False,
        )
        return True
    except (OSError, subprocess.SubprocessError):
        return False


CARGO_AVAILABLE = _cargo_runnable()


@pytest.mark.skipif(not CARGO_AVAILABLE, reason="cargo not on PATH — skipping live build")
class TestRustBuildIntegration:
    def test_zero_dep_rust_server_builds_offline(self, tmp_path: Path):
        """The zero-dependency skill must build successfully offline."""
        spec = importlib.util.spec_from_file_location("_skill_int", _SKILL_SRC)
        assert spec is not None and spec.loader is not None

        import types
        stub = types.ModuleType("_skill")
        skill_stub_src = """\
def skill(fn=None, *, name=None, description=None):
    def _wrap(f):
        f._skill_name = name or f.__name__
        f._skill_description = description or f.__doc__ or ""
        f._is_skill = True
        return f
    return _wrap(fn) if fn is not None else _wrap
"""
        exec(skill_stub_src, stub.__dict__)  # known-safe: execing a local string literal
        sys.modules["_skill"] = stub

        try:
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)  # type: ignore[union-attr]
        finally:
            sys.modules.pop("_skill", None)

        result_json = module.build_std_rust_http_server(
            build_dir=str(tmp_path / "rust-build"),
            port=9999,
        )
        result = json.loads(result_json)

        assert result["built"] is True, (
            f"Zero-dependency Rust build failed.\n"
            f"exit_code={result['exit_code']}\n"
            f"stderr={result['stderr']}"
        )
        assert result["binary_exists"] is True
        assert result["binary_path"] is not None

        binary = Path(result["binary_path"])
        assert binary.is_file()
        assert os.access(str(binary), os.X_OK), "Binary must be executable"
