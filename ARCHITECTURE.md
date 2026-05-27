# Agent AI Architecture

Here is the high-level architecture diagram for the Agent AI framework. It visualizes the flow of data from external interfaces (UI, Adapters), through the asynchronous FastAPI gateway, and into the core ReAct Agent loop, which coordinates with memory stores, tooling, and execution sandboxes.

```mermaid
graph TD
    %% External Interfaces
    UI[Control Panel UI <br/>React + Tailwind]
    Adapters[Messaging Adapters <br/>Telegram, Discord, Slack, Email]
    TUI[Terminal UI]

    %% Gateway & Concurrency
    subgraph Gateway Layer
        API[FastAPI Gateway]
        WS[WebSocket Stream]
        Queue[Session FIFO Queue]
        
        API --- WS
        WS --> Queue
    end

    UI --> API
    Adapters --> API
    TUI --> WS

    %% Core Agent Engine
    subgraph Core Agent Engine
        Agent[Agent Engine <br/>ReAct Loop]
        Contract[Task Contract System <br/>Evidence Gating]
        Plan[Plan Management <br/>Action Ledger]
        Eval[Skill Distiller <br/>Evaluator]
        
        Agent <--> Contract
        Agent <--> Plan
        Agent --> Eval
    end

    Queue --> Agent

    %% State & Memory
    subgraph State & Memory
        Checkpoint[(SQLite Checkpointer <br/>State & History)]
        Chroma[(ChromaDB <br/>Semantic Memory)]
        Neo4j[(Neo4j <br/>Graph Memory)]
        
        Agent <--> Checkpoint
        Agent <--> Chroma
        Agent <--> Neo4j
    end

    %% External Services
    LLM((LLM Provider <br/>LiteLLM))
    Agent <--> LLM

    %% Tooling & Execution
    subgraph Tooling & Execution Sandbox
        Tools[Tool Manager]
        MCP[MCP Servers]
        Sandbox[Terminal Sandbox <br/>Docker / Host / Serverless]
        Skills[Evolved Skills <br/>Auto-Maker]
        
        Agent --> Tools
        Tools --> MCP
        Tools --> Sandbox
        Tools --> Skills
        Eval -.->|Synthesizes & Validates| Skills
    end
```
