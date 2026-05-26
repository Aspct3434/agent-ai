"""Render the agent's GitHub-flavored Markdown as Telegram-flavored HTML.

The agent is instructed to answer in GitHub-flavored Markdown, but the
Telegram Bot API renders message text literally unless ``parse_mode`` is
set -- so ``**bold**``, ``# headings``, ``` ```code fences``` ```,
``[links](url)`` and ``- bullets`` show up as raw characters.  Telegram
supports a small *HTML* subset (``b i u s code pre a blockquote``) which is
far more robust than ``MarkdownV2`` (no fragile escaping of ``. ! - ( )``),
so this module converts the Markdown the agent emits into that subset.

Public API:
    ``markdown_to_telegram_html(md)``        -> one HTML string
    ``render_telegram_html_chunks(md, n)``   -> HTML strings each <= n chars,
                                                with balanced tags per chunk
    ``html_to_plain(html)``                  -> tags stripped (safety fallback)

Telegram does NOT support HTML headings, lists, tables, or horizontal
rules, so those are mapped to the closest supported construct: headings to
bold lines, bullets to ``•``, tables/hr to monospaced ``<pre>`` blocks.
"""
from __future__ import annotations

import html
import re

# Telegram's hard limit on a single outgoing message (characters).
TG_MSG_LIMIT = 4096

_FENCE = re.compile(r"^\s*(`{3,}|~{3,})\s*([\w+.\-]*)\s*$")
_HEADING = re.compile(r"^\s{0,3}(#{1,6})\s+(.*?)\s*#*\s*$")
_HR = re.compile(r"^\s{0,3}([-*_])(?:\s*\1){2,}\s*$")
_BULLET = re.compile(r"^(\s*)[-*+]\s+(.*)$")
_ORDERED = re.compile(r"^(\s*)(\d+)[.)]\s+(.*)$")
_BLOCKQUOTE = re.compile(r"^\s{0,3}>\s?(.*)$")
_TABLE_SEP = re.compile(r"^\s*\|?\s*:?-{1,}:?\s*(\|\s*:?-{1,}:?\s*)*\|?\s*$")

_BOLD = re.compile(r"\*\*(.+?)\*\*|__(.+?)__", re.DOTALL)
_STRIKE = re.compile(r"~~(.+?)~~", re.DOTALL)
# Single-marker emphasis, guarded so it never fires mid-identifier
# (e.g. snake_case file names or ``2 * 3``).
_ITALIC_STAR = re.compile(r"(?<![\w*])\*(?!\s)([^*\n]+?)(?<!\s)\*(?![\w*])")
_ITALIC_USCORE = re.compile(r"(?<![\w_])_(?!\s)([^_\n]+?)(?<!\s)_(?![\w_])")

_INLINE_CODE = re.compile(r"`([^`\n]+)`")
_LINK = re.compile(r"\[([^\]]+)\]\((\S+?)\)")
_TAG = re.compile(r"<[^>]+>")
_PLACEHOLDER = re.compile(r"\x00(\d+)\x00")


def _render_inline(text: str) -> str:
    """Convert inline Markdown in *text* to Telegram HTML, escaping the rest."""
    stash: list[str] = []

    def _hold(fragment: str) -> str:
        stash.append(fragment)
        return f"\x00{len(stash) - 1}\x00"

    # Protect inline code and links before escaping/emphasis can corrupt them.
    text = _INLINE_CODE.sub(
        lambda m: _hold(f"<code>{html.escape(m.group(1))}</code>"), text
    )
    text = _LINK.sub(
        lambda m: _hold(
            f'<a href="{html.escape(m.group(2), quote=True)}">{html.escape(m.group(1))}</a>'
        ),
        text,
    )

    # Escape literal text; Markdown markers (* _ ~) are not HTML-special so
    # they survive and the emphasis passes below can still see them.
    text = html.escape(text)

    text = _BOLD.sub(lambda m: f"<b>{m.group(1) or m.group(2)}</b>", text)
    text = _STRIKE.sub(lambda m: f"<s>{m.group(1)}</s>", text)
    text = _ITALIC_STAR.sub(lambda m: f"<i>{m.group(1)}</i>", text)
    text = _ITALIC_USCORE.sub(lambda m: f"<i>{m.group(1)}</i>", text)

    return _PLACEHOLDER.sub(lambda m: stash[int(m.group(1))], text)


def _render_table(rows: list[str]) -> str:
    """Render a Markdown table as an aligned monospace ``<pre>`` block."""

    def cells(line: str) -> list[str]:
        line = line.strip()
        if line.startswith("|"):
            line = line[1:]
        if line.endswith("|"):
            line = line[:-1]
        return [c.strip() for c in line.split("|")]

    body = [cells(r) for i, r in enumerate(rows) if i != 1]  # row 1 is the --- sep
    width = max((len(r) for r in body), default=0)
    cols = [
        max((len(r[c]) for r in body if c < len(r)), default=0) for c in range(width)
    ]
    lines: list[str] = []
    for ri, row in enumerate(body):
        padded = " | ".join(
            (row[c] if c < len(row) else "").ljust(cols[c]) for c in range(width)
        )
        lines.append(padded.rstrip())
        if ri == 0:  # underline the header row
            lines.append("-+-".join("-" * cols[c] for c in range(width)))
    return f"<pre>{html.escape(chr(10).join(lines))}</pre>"


def markdown_to_blocks(md: str) -> list[str]:
    """Convert Markdown to a list of self-contained, tag-balanced HTML blocks.

    Each block is safe to send on its own; the chunker joins blocks with
    newlines without ever splitting an HTML tag.
    """
    lines = md.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    blocks: list[str] = []
    i, n = 0, len(lines)

    while i < n:
        line = lines[i]

        fence = _FENCE.match(line)
        if fence:
            marker = fence.group(1)[0]
            lang = fence.group(2)
            close = re.compile(rf"^\s*{re.escape(marker)}{{3,}}\s*$")
            body: list[str] = []
            i += 1
            while i < n and not close.match(lines[i]):
                body.append(lines[i])
                i += 1
            i += 1  # consume the closing fence (or run off the end)
            code = html.escape("\n".join(body))
            if lang:
                blocks.append(
                    f'<pre><code class="language-{html.escape(lang)}">{code}</code></pre>'
                )
            else:
                blocks.append(f"<pre>{code}</pre>")
            continue

        if "|" in line and i + 1 < n and _TABLE_SEP.match(lines[i + 1]):
            rows = [line]
            i += 1
            while i < n and "|" in lines[i] and lines[i].strip():
                rows.append(lines[i])
                i += 1
            blocks.append(_render_table(rows))
            continue

        if _HR.match(line):
            blocks.append("──────────")
            i += 1
            continue

        heading = _HEADING.match(line)
        if heading:
            blocks.append(f"<b>{_render_inline(heading.group(2))}</b>")
            i += 1
            continue

        quote = _BLOCKQUOTE.match(line)
        if quote:
            quoted: list[str] = []
            while i < n and (qm := _BLOCKQUOTE.match(lines[i])):
                quoted.append(_render_inline(qm.group(1)))
                i += 1
            blocks.append("<blockquote>" + "\n".join(quoted) + "</blockquote>")
            continue

        bullet = _BULLET.match(line)
        if bullet:
            indent = " " * len(bullet.group(1))
            blocks.append(f"{indent}• {_render_inline(bullet.group(2))}")
            i += 1
            continue

        ordered = _ORDERED.match(line)
        if ordered:
            indent = " " * len(ordered.group(1))
            blocks.append(
                f"{indent}{ordered.group(2)}. {_render_inline(ordered.group(3))}"
            )
            i += 1
            continue

        # Plain text line (blank lines become empty blocks to preserve spacing).
        blocks.append(_render_inline(line) if line.strip() else "")
        i += 1

    return blocks


def markdown_to_telegram_html(md: str) -> str:
    """Convert Markdown to a single Telegram-HTML string."""
    return "\n".join(markdown_to_blocks(str(md)))


def _split_block(block: str, limit: int) -> list[str]:
    """Split one oversized block into <=limit pieces while keeping tags valid."""
    if block.startswith("<pre>"):
        inner = block[len("<pre>") :]
        if inner.endswith("</pre>"):
            inner = inner[: -len("</pre>")]
        code_open = re.match(r"^<code[^>]*>", inner)
        if code_open and inner.endswith("</code>"):
            inner = inner[code_open.end() : -len("</code>")]
        budget = max(1, limit - len("<pre></pre>"))
        pieces: list[str] = []
        cur = ""
        for raw in inner.split("\n"):
            candidate = raw if not cur else f"{cur}\n{raw}"
            if len(candidate) <= budget:
                cur = candidate
                continue
            if cur:
                pieces.append(cur)
                cur = ""
            while len(raw) > budget:
                pieces.append(raw[:budget])
                raw = raw[budget:]
            cur = raw
        if cur:
            pieces.append(cur)
        return [f"<pre>{p}</pre>" for p in pieces] or [block]

    # A non-<pre> block over the limit is rare (a single >4096-char paragraph).
    # Hard-slicing it may break an inline tag; the adapter's plain-text fallback
    # recovers if Telegram rejects the markup.
    return [block[i : i + limit] for i in range(0, len(block), limit)]


def render_telegram_html_chunks(md: str, limit: int = TG_MSG_LIMIT) -> list[str]:
    """Render Markdown to Telegram-HTML chunks, each <= *limit* characters.

    Greedily packs whole blocks together so tags are never split across a
    chunk boundary; oversized single blocks are split internally.
    """
    chunks: list[str] = []
    current = ""
    for block in markdown_to_blocks(str(md)):
        if len(block) > limit:
            if current:
                chunks.append(current)
                current = ""
            chunks.extend(_split_block(block, limit))
            continue
        candidate = block if not current else f"{current}\n{block}"
        if len(candidate) <= limit:
            current = candidate
        else:
            chunks.append(current)
            current = block
    if current:
        chunks.append(current)
    return chunks or [""]


def html_to_plain(text: str) -> str:
    """Strip Telegram HTML tags and unescape entities (plain-text fallback)."""
    return html.unescape(_TAG.sub("", text))
