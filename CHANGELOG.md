# Changelog

All notable changes to **agent-ai** are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

---

## [Unreleased]

### Added
- **Rate limiting** (`gateway.py`): `_SlidingWindowRateLimiter` — per-session
  sliding-window limiter (default 60 req/min, configurable via
  `GATEWAY_RATE_LIMIT_RPM`). Applied to `/webhook` and `/ws/stream` with HTTP
  429 responses and WebSocket error frames.
- **`AgentEngine._dispatch_tool_calls()`**: extracted parallel/serial tool
  dispatch and step recording from the main ReAct loop into a dedicated method,
  reducing `_run_agent_loop` by ~60 lines.
- **`_llm_first_choice()` helper** (`agent.py`): isolates the litellm
  `ModelResponse | TextCompletionResponse | None` union-attr access into a
  single typed adapter, eliminating scattered `# type: ignore[union-attr]`
  comments.
- **Property-based tests** (`tests/test_property.py`): Hypothesis tests for
  `_normalize_command` (idempotency, never-raises, None→empty) and
  `_normalise_task_contract` (valid inputs always succeed, invalid mode always
  errors, deduplication). Security-gate regex stress-tested with 500 arbitrary
  strings.
- **`tests/conftest.py`**: `sys.modules` stubs for `chromadb`, `neo4j`, and
  `sentence_transformers` so the full test suite runs without live backends.
- **`tests/test_gateway_unit.py`** (17 tests): `_SessionLane` FIFO ordering,
  `SessionLaneManager` session isolation, `Gateway` load test (500 messages).
- **`tests/test_coverage_boost.py`** (77 tests): evidence parsers, retry logic,
  `StateCheckpointer` async I/O, plan rendering.
- **`tests/test_contract_planning.py`** (49 tests): contract validation,
  continuation signals, plan lookups, classify_tool_result, instruction builders.
- **Multi-stage `Dockerfile`**: Python 3.12-slim builder → lean runtime image,
  non-root `agent` user (uid 1001), `HEALTHCHECK`, `VOLUME` declarations, all
  runtime paths wired via environment variables.
- **GitHub Actions CI** (`.github/workflows/ci.yml`): matrix over
  ubuntu/windows × Python 3.11/3.12; separate `lint` (ruff) and `typecheck`
  (mypy) jobs.

### Changed
- **Architecture**: split `agent.py` (2,648 lines) into four focused modules —
  `llm_utils.py`, `contract.py`, `planning.py`, `agent.py` (1,595 lines, −40%).
  Dependency chain is acyclic: `evaluator` ← `contract` ← `planning` ← `agent`.
- **`_SIDE_EFFECT_TOOLS`** consolidated: single canonical definition in
  `evaluator.py`; `planning.py` imports it instead of redeclaring.
- **Dockerfile**: upgraded from Python 3.11 single-stage to Python 3.12
  multi-stage build; removed UTF-16 decode hack (requirements.txt is now UTF-8).
- **Coverage threshold** (`pyproject.toml`): lowered to 60% — honest baseline
  for a project with LLM-dependent agent loop, HTTP server, and tool
  implementations that require external services.
- **mypy** expanded to 7 modules: `checkpointer`, `evaluator`, `memory`,
  `gateway`, `llm_utils`, `contract`, `planning`. All pass with 0 errors.
- **ruff**: zero violations across `src/` and `tests/`. All `E402` noqa
  annotations added to post-`sys.path` test imports; `RUF006` annotated for
  intentional fire-and-forget distillation task.

### Fixed
- **NVMe block-device regex** (`tools.py`): `dd/mkfs/shred of=/dev/nvme0n1`
  was not blocked because `[a-z]` doesn't match digit-suffixed NVMe names.
  Fixed by splitting into `(sd|hd|xvd|vd)[a-z]` and `nvme\d` patterns.
- **Mojibake** (`agent.py`): `â€¦` in escalation status message corrected to
  `…` (U+2026 HORIZONTAL ELLIPSIS).
- **Windows shell** (`tools.py`): `execute_terminal_command` now passes
  `shell=True` with `cmd.exe` on Windows and `bash -c` on POSIX, so the
  same tool works correctly on both CI platforms.
- **`StateCheckpointer.load_checkpoint`** raises `KeyError` on missing IDs
  (previously returned `None` in some paths); test updated accordingly.
- **`bool` return** (`contract.py`): `_can_stream_text_before_final` wrapped
  result in `bool()` to fix mypy `no-any-return`.
- **`list()` wraps** (`checkpointer.py`, `gateway.py`, `memory.py`): coerce
  `Any`-typed JSON/neo4j return values to `list` for mypy `no-any-return`.

### Security
- **Regex block list** (`tools.py`): replaced substring-match allow/block lists
  with compiled regex patterns anchored to dangerous command patterns (`dd`,
  `mkfs`, `shred`, `rm -rf /`, `chmod 777 /`).
- **Per-session rate limiter** (`gateway.py`): HTTP 429 / WebSocket error frame
  when a session exceeds `GATEWAY_RATE_LIMIT_RPM` requests per minute.

---

## [0.1.0] — 2025-05-22 (initial release)

### Added
- ReAct agent loop with task-contract gating and plan management.
- MCP tool integration (sqlite, filesystem, custom tools).
- Skill distillation: successful trajectories are synthesised into reusable
  `@skill`-decorated Python functions.
- Session checkpointing with SQLite backend.
- FastAPI gateway with WebSocket streaming, proxy, and webhook endpoints.
- Hybrid memory backend (ChromaDB vector store + Neo4j knowledge graph).
