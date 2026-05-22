from __future__ import annotations


def skill(fn=None, *, name=None, description=None):
    '''Mark a function as an MCP skill so the skills server auto-discovers it.

    Use as a plain decorator or with keyword arguments::

        from _skill import skill

        @skill
        def greet(name: str) -> str:
            "Return a personalised greeting."
            return f"Hello, {name}!"

        @skill(name="shout", description="Return an uppercased string.")
        def shout(text: str) -> str:
            return text.upper()
    '''
    def _decorate(f):
        f._is_skill = True
        f._skill_name = name or f.__name__
        f._skill_description = description or f.__doc__ or ""
        return f

    return _decorate(fn) if fn is not None else _decorate
