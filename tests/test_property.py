"""Property-based tests using Hypothesis.

These tests verify invariants that must hold for *all* inputs, not just the
hand-crafted examples in the unit tests.  They complement rather than replace
the unit tests: the unit tests prove specific cases; these tests prove
universal properties.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hypothesis import given, settings
from hypothesis import strategies as st

from contract import (
    _normalise_task_contract,
    _normalize_command,
)

# ---------------------------------------------------------------------------
# _normalize_command — universal invariants
# ---------------------------------------------------------------------------

@given(st.text())
def test_normalize_command_always_returns_str(text: str) -> None:
    """Whatever string we pass in, we always get a string back."""
    result = _normalize_command(text)
    assert isinstance(result, str)


@given(st.text())
def test_normalize_command_never_raises(text: str) -> None:
    """_normalize_command must not raise on any text input."""
    _normalize_command(text)  # should not raise


@given(st.none())
def test_normalize_command_none_is_empty(none_val: None) -> None:
    """None input returns the empty string (documented contract)."""
    assert _normalize_command(none_val) == ""


@given(st.text(alphabet=st.characters(whitelist_categories=("Zs",))))
def test_normalize_command_whitespace_only_is_empty(ws: str) -> None:
    """Whitespace-only input normalises to the empty string."""
    assert _normalize_command(ws) == ""


@given(st.text(), st.text())
def test_normalize_command_idempotent(text: str, _ignored: str) -> None:
    """Normalising an already-normalised string returns the same string."""
    once = _normalize_command(text)
    twice = _normalize_command(once)
    assert once == twice


# ---------------------------------------------------------------------------
# _normalise_task_contract — universal invariants
# ---------------------------------------------------------------------------

_VALID_MODES = st.sampled_from(["answer", "execute"])
_VALID_EVIDENCE = st.lists(
    st.sampled_from(
        ["filesystem_artifact",
         "running_http_service", "running_tcp_service", "database_mutation",
         "command_output", "none"]
    ),
    min_size=1,
    max_size=4,
)


@given(st.dictionaries(st.text(), st.text() | st.lists(st.text()) | st.integers()))
@settings(max_examples=200)
def test_normalise_task_contract_never_raises(d: dict) -> None:
    """_normalise_task_contract must not raise on any dict input.

    The function returns (populated_dict, None) on success and
    ({}, error_str) on failure — it never raises.
    """
    contract, err = _normalise_task_contract(d)
    # The function returns an empty dict on error; a populated dict on success.
    assert isinstance(contract, dict)
    if err is None:
        # Success: contract must have the required keys
        assert "mode" in contract
        assert "summary" in contract
    else:
        # Failure: contract is the empty sentinel dict
        assert contract == {}
        assert isinstance(err, str) and err


# Strategy for non-whitespace-only summaries (stripped length >= 1)
_NON_BLANK_TEXT = st.text(min_size=1, max_size=200).filter(lambda s: s.strip())


@given(
    _VALID_MODES,
    _NON_BLANK_TEXT,
    st.lists(_NON_BLANK_TEXT, min_size=1, max_size=5),
    _VALID_EVIDENCE,
)
@settings(max_examples=300)
def test_normalise_task_contract_valid_inputs_always_succeed(
    mode: str, summary: str, criteria: list[str], evidence: list[str]
) -> None:
    """Well-formed inputs always produce a contract with no error."""
    # execute mode requires at least one non-'none' evidence type
    if mode == "execute":
        evidence = [e for e in evidence if e != "none"] or ["filesystem_artifact"]

    contract, err = _normalise_task_contract(
        {
            "mode": mode,
            "summary": summary,
            "success_criteria": criteria,
            "evidence_requirements": evidence,
        }
    )
    assert err is None, f"Unexpected error for valid input: {err!r}"
    assert contract["mode"] == mode
    # Deduplication: no repeats in the returned evidence list
    seen: set[str] = set()
    for req in contract["evidence_requirements"]:
        assert req not in seen, f"Duplicate evidence requirement: {req!r}"
        seen.add(req)


@given(st.text())
def test_normalise_task_contract_invalid_mode_always_errors(mode: str) -> None:
    """Any mode that is not 'answer' or 'execute' must produce an error."""
    if mode in {"answer", "execute"}:
        return
    _, err = _normalise_task_contract(
        {
            "mode": mode,
            "summary": "do something",
            "success_criteria": ["done"],
            "evidence_requirements": ["none"],
        }
    )
    assert err is not None
    assert "mode" in err


# ---------------------------------------------------------------------------
# Security gate — regex must never raise on arbitrary command strings
# ---------------------------------------------------------------------------

def test_security_gate_import() -> None:
    """The security gate regex patterns can be imported cleanly."""
    from tools import ToolManager  # noqa: F401


@given(st.text(max_size=2000))
@settings(max_examples=500)
def test_security_gate_never_raises(command: str) -> None:
    """The security gate must not raise (e.g. catastrophic backtracking) on any string."""
    import re
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    import tools as tools_module
    # Access the compiled patterns directly via the module's _BLOCK_PATTERNS list
    block_patterns = getattr(tools_module, "_BLOCK_PATTERNS", [])
    for pattern in block_patterns:
        if isinstance(pattern, re.Pattern):
            pattern.search(command)  # must not raise or hang
