"""Unit tests for the Markdown→Telegram-HTML renderer."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from adapters._telegram_format import (
    TG_MSG_LIMIT,
    html_to_plain,
    markdown_to_telegram_html,
    render_telegram_html_chunks,
)


class TestInlineFormatting:
    def test_bold_double_star(self) -> None:
        assert markdown_to_telegram_html("**hi**") == "<b>hi</b>"

    def test_bold_double_underscore(self) -> None:
        assert markdown_to_telegram_html("__hi__") == "<b>hi</b>"

    def test_italic_star(self) -> None:
        assert markdown_to_telegram_html("an *emphatic* word") == "an <i>emphatic</i> word"

    def test_strikethrough(self) -> None:
        assert markdown_to_telegram_html("~~gone~~") == "<s>gone</s>"

    def test_inline_code(self) -> None:
        assert markdown_to_telegram_html("use `pip install`") == "use <code>pip install</code>"

    def test_link(self) -> None:
        out = markdown_to_telegram_html("[Anthropic](https://anthropic.com)")
        assert out == '<a href="https://anthropic.com">Anthropic</a>'

    def test_snake_case_not_italicized(self) -> None:
        # Underscores inside identifiers must not be treated as emphasis.
        assert markdown_to_telegram_html("see my_file_name.py") == "see my_file_name.py"

    def test_bold_wrapping_link(self) -> None:
        out = markdown_to_telegram_html("**[x](http://e.co)**")
        assert out == '<b><a href="http://e.co">x</a></b>'


class TestEscaping:
    def test_angle_brackets_escaped(self) -> None:
        assert markdown_to_telegram_html("a < b & c > d") == "a &lt; b &amp; c &gt; d"

    def test_code_content_escaped(self) -> None:
        out = markdown_to_telegram_html("`<div>`")
        assert out == "<code>&lt;div&gt;</code>"

    def test_no_raw_markdown_markers_leak(self) -> None:
        out = markdown_to_telegram_html("# Title\n\n**bold** and `code`")
        assert "**" not in out
        assert out.startswith("<b>Title</b>")


class TestBlockFormatting:
    def test_heading_becomes_bold(self) -> None:
        assert markdown_to_telegram_html("## Section") == "<b>Section</b>"

    def test_bullets_become_dots(self) -> None:
        out = markdown_to_telegram_html("- one\n- two")
        assert out == "• one\n• two"

    def test_ordered_list_preserved(self) -> None:
        out = markdown_to_telegram_html("1. first\n2. second")
        assert out == "1. first\n2. second"

    def test_blockquote(self) -> None:
        out = markdown_to_telegram_html("> quoted line")
        assert out == "<blockquote>quoted line</blockquote>"

    def test_fenced_code_block_with_language(self) -> None:
        md = "```python\nprint(1)\n```"
        out = markdown_to_telegram_html(md)
        assert out == '<pre><code class="language-python">print(1)</code></pre>'

    def test_fenced_code_block_escapes_content(self) -> None:
        md = "```\nif a < b & c:\n```"
        out = markdown_to_telegram_html(md)
        assert out == "<pre>if a &lt; b &amp; c:</pre>"

    def test_horizontal_rule(self) -> None:
        assert markdown_to_telegram_html("---") == "──────────"

    def test_table_renders_as_pre(self) -> None:
        md = "| a | b |\n| - | - |\n| 1 | 2 |"
        out = markdown_to_telegram_html(md)
        assert out.startswith("<pre>")
        assert out.endswith("</pre>")
        assert "a" in out and "b" in out and "1" in out and "2" in out


class TestChunking:
    def test_short_message_single_chunk(self) -> None:
        assert render_telegram_html_chunks("**hi**") == ["<b>hi</b>"]

    def test_long_plain_text_split_under_limit(self) -> None:
        chunks = render_telegram_html_chunks("x" * 9000)
        assert len(chunks) == 3
        assert all(len(c) <= TG_MSG_LIMIT for c in chunks)

    def test_blocks_never_split_a_tag(self) -> None:
        md = "\n\n".join(f"**item {i}** body text here" for i in range(400))
        for chunk in render_telegram_html_chunks(md):
            assert len(chunk) <= TG_MSG_LIMIT
            # Every opening <b> in a chunk has a matching close.
            assert chunk.count("<b>") == chunk.count("</b>")

    def test_oversized_code_block_stays_balanced(self) -> None:
        md = "```\n" + ("line of code\n" * 1000) + "```"
        chunks = render_telegram_html_chunks(md)
        assert len(chunks) > 1
        for chunk in chunks:
            assert len(chunk) <= TG_MSG_LIMIT
            assert chunk.count("<pre>") == chunk.count("</pre>")
            assert chunk.startswith("<pre>") and chunk.endswith("</pre>")

    def test_empty_input_yields_one_empty_chunk(self) -> None:
        assert render_telegram_html_chunks("") == [""]


class TestHtmlToPlain:
    def test_strips_tags_and_unescapes(self) -> None:
        html_text = '<b>Bold</b> and <a href="http://e.co">link</a> &amp; more &lt;x&gt;'
        assert html_to_plain(html_text) == "Bold and link & more <x>"
