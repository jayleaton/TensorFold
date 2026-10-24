"""MiniMax-M3.1 required-call tags against the real xgrammar and its minimax_m3 builtin, on template-rendered calls."""

from __future__ import annotations

import json
import sys

USAGE = ("python -I tools/zig/minimax_grammar_check.py TOKENIZER_JSON CHAT_TEMPLATE FIXTURE: TOKENIZER_JSON and "
         "CHAT_TEMPLATE are MiniMax-M3.1-Flash's; FIXTURE is zig/src/server/fixtures/minimax_required_tags.json. "
         "Needs xgrammar 0.2.8, tokenizers and jinja2.")
NS = "]<]minimax[>["


def template(path: str):
    """The chat template as transformers renders it (sandboxed, trim/lstrip blocks, its tojson)."""
    import jinja2
    import jinja2.ext
    from jinja2.sandbox import ImmutableSandboxedEnvironment

    def tojson(x, ensure_ascii=False, indent=None, separators=None, sort_keys=False):
        return json.dumps(x, ensure_ascii=ensure_ascii, indent=indent, separators=separators, sort_keys=sort_keys)

    def raise_exception(message):
        raise jinja2.exceptions.TemplateError(message)

    env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True, extensions=[jinja2.ext.loopcontrols])
    env.filters["tojson"] = tojson
    env.globals["raise_exception"] = raise_exception
    with open(path, encoding="utf-8") as f:
        return env.from_string(f.read())


def block(tpl, tools: list, sample: dict) -> str:
    """The `NS<tool_call>...NS</tool_call>` block the template writes for the sample's call or calls."""
    calls = sample.get("calls") or [sample]
    tool_calls = [{"type": "function", "function": {"name": c["name"], "arguments": c["arguments"]}} for c in calls]
    text = tpl.render(messages=[{"role": "user", "content": "x"}, {"role": "assistant", "content": "", "tool_calls": tool_calls}],
                      tools=tools, add_generation_prompt=False)
    start = text.rindex(NS + "<tool_call>")
    end = text.index(NS + "</tool_call>", start) + len(NS + "</tool_call>")
    return text[start:end]


def vocab(path: str):
    """tokenizer.json's byte-level vocabulary as xgrammar wants it, its size and the eos id."""
    with open(path, encoding="utf-8") as f:
        tj = json.load(f)
    pieces = dict(tj["model"]["vocab"])
    for t in tj.get("added_tokens", []):
        pieces[t["content"]] = t["id"]
    size = max(pieces.values()) + 1
    encoded = [""] * size
    for piece, i in pieces.items():
        encoded[i] = piece
    eos = pieces["[e~["]
    return encoded, size, eos


def walk(xgr, compiled, ids: list, eos: int) -> bool:
    """True when the matcher takes every id and then the end."""
    m = xgr.GrammarMatcher(compiled)
    for i in ids:
        if not m.accept_token(i):
            return False
    return m.accept_token(eos) and m.is_terminated()


def builtin(xgr, tools: list, parallel: bool):
    """xgrammar's own minimax_m3 tag for the same request (reasoning already closed), or the reason it has none."""
    try:
        return xgr.get_model_structural_tag("minimax_m3", tools=tools, tool_choice="required", reasoning="disabled",
                                            parallel_tool_calls=parallel)
    except Exception as e:  # noqa: BLE001 - the builtin's refusal is the result
        return f"{type(e).__name__}: {e}"


def compiled(compiler, tag):
    """The compiled tag, or why xgrammar refuses it."""
    try:
        return compiler.compile_structural_tag(tag)
    except Exception as e:  # noqa: BLE001 - the refusal is the result
        return f"{type(e).__name__}: {e}"


def main() -> int:
    if len(sys.argv) != 4:
        print(USAGE, file=sys.stderr)
        return 2
    import xgrammar as xgr
    from tokenizers import Tokenizer

    encoded, size, eos = vocab(sys.argv[1])
    tok = Tokenizer.from_file(sys.argv[1])
    info = xgr.TokenizerInfo(encoded, vocab_type=xgr.VocabType.BYTE_LEVEL, vocab_size=size, stop_token_ids=[eos])
    compiler = xgr.GrammarCompiler(info)
    tpl = template(sys.argv[2])
    with open(sys.argv[3], encoding="utf-8") as f:
        cases = json.load(f)
    failed = 0
    from importlib.metadata import version

    print(f"xgrammar {version('xgrammar')}, vocab {size}, eos {eos}")
    for c in cases:
        ours = compiler.compile_structural_tag(c["tag"])
        ref = builtin(xgr, c["tools"], c["parallel"])
        ref_compiled = ref if isinstance(ref, str) else compiled(compiler, ref)
        for want, samples in ((True, c["accept"]), (False, c["reject"])):
            for s in samples:
                try:
                    text = block(tpl, c["tools"], s)
                except Exception as e:  # noqa: BLE001 - a call the template cannot write is reported
                    print(f"FAIL {c['name']}: the template cannot write {s}: {e}")
                    failed += 1
                    continue
                ids = tok.encode(text, add_special_tokens=False).ids
                got = walk(xgr, ours, ids, eos)
                ref_got = "n/a" if isinstance(ref_compiled, str) else walk(xgr, ref_compiled, ids, eos)
                ok = got == want
                failed += not ok
                print(f"{'ok  ' if ok else 'FAIL'} {c['name']}: {'accept' if want else 'reject'} -> ours {got}, "
                      f"builtin {ref_got}, {len(ids)} tokens: {text[:90]!r}")
        if isinstance(ref_compiled, str):
            print(f"note {c['name']}: xgrammar's builtin has no grammar for these tools ({ref_compiled[:160]})")
    print(f"{'FAILED' if failed else 'PASSED'}: {failed} wrong of "
          f"{sum(len(c['accept']) + len(c['reject']) for c in cases)} samples over {len(cases)} tool sets")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
