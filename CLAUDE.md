# Agent AI — Project Instructions

## graphify Knowledge Graph

A knowledge graph of this codebase lives in `graphify-out/graph.json` (3,206 nodes, 7,748 edges, 190 communities). **Always query it before answering questions about architecture, module relationships, or code structure.**

### Query the graph

Before answering any question about how this codebase works, run a BFS query:

```powershell
$py = Get-Content graphify-out\.graphify_python
& $py -c "
import json, sys
import networkx as nx
from networkx.readwrite import json_graph
from pathlib import Path

data = json.loads(Path('graphify-out/graph.json').read_text(encoding='utf-8'))
G = json_graph.node_link_graph(data, edges='links')
question = 'QUESTION'
terms = [t.lower() for t in question.split() if len(t) > 3]
scored = sorted([(sum(1 for t in terms if t in G.nodes[n].get('label','').lower()), n) for n in G.nodes()], reverse=True)
start_nodes = [nid for _, nid in scored[:3] if _ > 0]
subgraph_nodes = set(start_nodes)
frontier = set(start_nodes)
for _ in range(3):
    next_frontier = set()
    for n in frontier:
        for nb in G.neighbors(n):
            if nb not in subgraph_nodes:
                next_frontier.add(nb)
    subgraph_nodes.update(next_frontier)
    frontier = next_frontier
for nid in sorted(subgraph_nodes, key=lambda n: sum(1 for t in terms if t in G.nodes[n].get('label','').lower()), reverse=True)[:40]:
    d = G.nodes[nid]
    print(f'NODE {d.get(\"label\",nid)} [{d.get(\"source_file\",\"\")}]')
for u, v in G.edges():
    if u in subgraph_nodes and v in subgraph_nodes:
        e = G[u][v]
        print(f'EDGE {G.nodes[u].get(\"label\",u)} --{e.get(\"relation\",\"\")}-> {G.nodes[v].get(\"label\",v)} [{e.get(\"confidence\",\"\")}]')
"
```

Replace `QUESTION` with key terms from the user's question.

### God nodes (always relevant)

These nodes connect everything — check them when tracing any cross-module question:

| Node | Edges | File |
|------|-------|------|
| `AgentEngine` | 154 | `src/agent.py` |
| `ToolManager` | 125 | `src/tools.py` |
| `ExecutionStep` | 103 | `src/evaluator.py` |
| `SkillRegistry` | 103 | `src/evaluator.py` |
| `StateCheckpointer` | 95 | `src/checkpointer.py` |
| `EvolutionEngine` | 86 | `src/evolution.py` |
| `NormalizedMessage` | 83 | `src/agent.py` |

### Community map (top clusters)

| ID | Label | Size |
|----|-------|------|
| 0 | Scheduler & Data Models | 86 |
| 1 | Gateway Auth & HTTP Routing | 81 |
| 2 | Terminal UI (TUI) | 73 |
| 3 | FastAPI Gateway Core | 70 |
| 4 | Skill Evaluator & Distiller | 69 |
| 5 | Task Contract Engine | 65 |
| 7 | Cron Scheduler & Jobs | 62 |
| 8 | Agent Engine Core | 61 |
| 9 | Proxy Auth & HMAC | 59 |
| 10 | Messaging Adapters | 54 |
| 11 | Task Graph Engine | 53 |
| 13 | Browser & HTTP Sandbox | 51 |
| 15 | Skill Evolution Engine | 48 |
| 16 | State Checkpointer | 47 |
| 18 | Tool Routing & Toolsets | 43 |
| 19 | LLM Utils & Streaming | 42 |

### Update the graph when files change

**After editing any source file**, run an incremental graph update so the graph stays current:

```powershell
$py = Get-Content graphify-out\.graphify_python
@"
import json
from graphify.detect import detect_incremental, save_manifest
from pathlib import Path
result = detect_incremental(Path('.'))
new_total = result.get('new_total', 0)
deleted = list(result.get('deleted_files', []))
Path('graphify-out/.graphify_incremental.json').write_text(json.dumps(result, ensure_ascii=False), encoding='utf-8')
if new_total == 0 and not deleted:
    print('Graph up to date - no changes detected')
else:
    print(f'{new_total} changed, {len(deleted)} deleted')
"@ | & $py
```

If changes are detected, run `/graphify . --update` to re-extract and rebuild.

**Trigger**: run the update check after any tool call that writes, edits, or deletes a file in `src/`, `tests/`, `control-panel/src/`, or `scripts/`.
