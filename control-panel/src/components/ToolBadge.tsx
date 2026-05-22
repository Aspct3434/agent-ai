import { useState } from "react";
import { ChevronDown, Wrench } from "lucide-react";

// ---------------------------------------------------------------------------
// JSON syntax tokeniser
// ---------------------------------------------------------------------------

type TokenKind = "key" | "string" | "number" | "literal" | "punct" | "space";

interface Token {
  kind: TokenKind;
  text: string;
}

// Each alternative is a capture group; order matters.
// Group 1 – object key  (quoted string immediately followed by colon)
// Group 2 – string value
// Group 3 – boolean / null literal
// Group 4 – number
// Group 5 – structural punctuation
// Group 6 – everything else (whitespace, newlines, …)
const TOKEN_RE =
  /("(?:[^"\\]|\\.)*"\s*:)|("(?:[^"\\]|\\.)*")|(true|false|null)|(-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)|([{}[\],])|(\s+|.)/g;

function tokenize(json: string): Token[] {
  const tokens: Token[] = [];
  TOKEN_RE.lastIndex = 0;
  let m: RegExpExecArray | null;
  while ((m = TOKEN_RE.exec(json)) !== null) {
    if      (m[1]) tokens.push({ kind: "key",     text: m[1] });
    else if (m[2]) tokens.push({ kind: "string",  text: m[2] });
    else if (m[3]) tokens.push({ kind: "literal", text: m[3] });
    else if (m[4]) tokens.push({ kind: "number",  text: m[4] });
    else if (m[5]) tokens.push({ kind: "punct",   text: m[5] });
    else           tokens.push({ kind: "space",   text: m[6] });
  }
  return tokens;
}

const TOKEN_COLOR: Record<TokenKind, string> = {
  key:     "text-sky-400",
  string:  "text-emerald-400",
  number:  "text-orange-400",
  literal: "text-violet-400",
  punct:   "text-zinc-500",
  space:   "",
};

// ---------------------------------------------------------------------------
// Sub-component: highlighted <pre> block
// ---------------------------------------------------------------------------

function HighlightedJSON({ value }: { value: Record<string, unknown> }) {
  const json = JSON.stringify(value, null, 2);
  const tokens = tokenize(json);
  return (
    <pre className="m-0 overflow-x-auto whitespace-pre text-xs leading-relaxed">
      {tokens.map((tok, i) => (
        <span key={i} className={TOKEN_COLOR[tok.kind]}>
          {tok.text}
        </span>
      ))}
    </pre>
  );
}

// ---------------------------------------------------------------------------
// ToolBadge
// ---------------------------------------------------------------------------

export interface ToolBadgeProps {
  toolName: string;
  params: Record<string, unknown>;
}

export function ToolBadge({ toolName, params }: ToolBadgeProps) {
  const [open, setOpen] = useState(true);

  return (
    <div className="w-full overflow-hidden rounded-lg border border-zinc-700 font-mono shadow-lg">
      {/* Header */}
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        className="flex w-full cursor-pointer items-center gap-2 bg-zinc-800 px-3 py-2 text-left transition-colors hover:bg-zinc-750"
        aria-expanded={open}
      >
        <Wrench size={14} className="shrink-0 text-zinc-400" />
        <span className="flex-1 truncate text-xs font-semibold tracking-wide text-zinc-200">
          {toolName}
        </span>
        <ChevronDown
          size={14}
          className={`shrink-0 text-zinc-400 transition-transform duration-200 ${
            open ? "rotate-180" : "rotate-0"
          }`}
        />
      </button>

      {/* Collapsible body — grid trick for smooth height animation */}
      <div
        className={`grid transition-all duration-200 ease-in-out ${
          open ? "grid-rows-[1fr]" : "grid-rows-[0fr]"
        }`}
      >
        <div className="overflow-hidden">
          <div className="bg-zinc-900 px-4 py-3 text-zinc-300">
            <HighlightedJSON value={params} />
          </div>
        </div>
      </div>
    </div>
  );
}

export default ToolBadge;
