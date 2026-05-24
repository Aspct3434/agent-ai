# Sub-agent delegation guide

Use `delegate_task` to spin up specialized sub-agents for focused work.

## When to delegate

- **researcher**: gathering facts, reading docs, comparing options — any pure-information task with no side effects
- **coder**: writing, refactoring, or reviewing code — when you need high-quality implementation
- **auditor**: security review, correctness check, finding bugs — before declaring a codebase done
- **planner**: breaking a complex goal into ordered, verifiable steps

## Parallel delegation

Pass `tasks` as a list and `mode: "parallel"` to run multiple sub-agents concurrently:

```json
{
  "tasks": [
    {"agent_type": "researcher", "task_description": "Find the best Python HTTP library"},
    {"agent_type": "researcher", "task_description": "Find deployment options for FastAPI"}
  ],
  "mode": "parallel"
}
```

## Rules

- Never delegate a task that requires interactive user input
- Researcher agents must not make filesystem changes
- Coder and auditor agents work on isolated temporary workspaces
- Always incorporate sub-agent results before giving your final answer
