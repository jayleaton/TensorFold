"""MiniMax-M3.1 serving goldens: OpenAI requests rendered by the chat template in Jinja2, as transformers does."""

from __future__ import annotations

import copy
import hashlib
import json
import sys
from pathlib import Path

import jinja2
import jinja2.ext
from jinja2.sandbox import ImmutableSandboxedEnvironment

USAGE = ("python3 -I tools/zig/minimax_reference.py OUT.jsonl NAME=TEMPLATE.jinja...: each line is one request (OpenAI "
         "messages, tools, thinking, effort) and the prompt the template renders after the serving rules; \"upstream\" "
         "holds the text (or \"upstream_error\" the refusal) with upstream cleaning alone where that differs.")
LATE_SYSTEM = "system"  # what server.zig's probe finds for both MiniMax templates: a later system message renders
TOOLS = [
    {"type": "function", "function": {"name": "get_weather", "description": "Weather for a city (été, 天气)", "parameters": {
        "type": "object", "properties": {"city": {"type": "string"}, "days": {"type": "integer"},
                                         "units": {"type": "string"}}, "required": ["city"]}}},
    {"type": "function", "function": {"name": "search", "description": "Web \"search\"", "parameters": {
        "type": "object", "properties": {"q": {"type": "string"}, "filters": {"type": "object"}}}}},
    {"type": "function", "function": {"name": "ping", "description": "No arguments", "parameters": {
        "type": "object", "properties": {}}}},
]
ARGS_W = {"city": "Paris", "days": 3}
ARGS_S = {"q": "naïve \"quoted\" 東京\nline2", "filters": {"lang": ["en", "fr"], "safe": True, "n": 1.5, "x": None}}
J = json.dumps


def raise_exception(message):
    raise jinja2.exceptions.TemplateError(message)


def tojson(x, ensure_ascii=False, indent=None, separators=None, sort_keys=False):
    return json.dumps(x, ensure_ascii=ensure_ascii, indent=indent, separators=separators, sort_keys=sort_keys)


def environment() -> ImmutableSandboxedEnvironment:
    """transformers' ``_compile_jinja_template`` environment."""
    env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True, extensions=[jinja2.ext.loopcontrols])
    env.filters["tojson"] = tojson
    env.globals["raise_exception"] = raise_exception
    return env


def call(cid: str, name: str, args: object) -> dict:
    return {"id": cid, "type": "function", "function": {"name": name, "arguments": args}}


def tool(cid: str, content: object) -> dict:
    return {"role": "tool", "tool_call_id": cid, "content": content}


def tool_turn(args_w=None, args_s=None, rc="Need weather.", parallel=False, out_of_order=False, content="") -> list:
    """An assistant turn calling get_weather (and search when parallel), then its tool results."""
    calls = [call("call_1", "get_weather", ARGS_W if args_w is None else args_w)]
    results = [tool("call_1", "{\"temp\": 18}")]
    if parallel:
        calls.append(call("call_2", "search", ARGS_S if args_s is None else args_s))
        results.append(tool("call_2", "3 results"))
        if out_of_order:
            results.reverse()
    turn = {"role": "assistant", "content": content, "tool_calls": calls}
    if rc is not None:
        turn["reasoning_content"] = rc
    return [turn] + results


def tool_loop(args_w=None, final=True, **kw) -> list:
    m = [{"role": "system", "content": "You are helpful."}, {"role": "user", "content": "Weather in Paris?"}]
    m += tool_turn(args_w, **kw)
    if final:
        m.append({"role": "assistant", "content": "It is 18C.", "reasoning_content": "Got it."})
    return m


def cases() -> dict:
    """Name -> request: messages, tools, thinking, effort, generation (and skip with its reason)."""
    hi = [{"role": "user", "content": "Hi"}]
    chat = [{"role": "system", "content": "Be brief."}, {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello!", "reasoning_content": "greet"}, {"role": "user", "content": "2+2?"}]
    later = [{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"}]
    two_rounds = tool_loop(J(ARGS_W), final=False) + tool_turn(J({"city": "Lyon"}), rc="Also Lyon.") + [
        {"role": "assistant", "content": "Paris 18C, Lyon 18C.", "reasoning_content": "done"},
        {"role": "user", "content": "Thanks, and Nice?"}]
    three = tool_loop(final=False, parallel=True)[:2] + [
        {"role": "assistant", "content": None, "tool_calls": [call("a", "get_weather", {"city": "A"}),
                                                              call("b", "get_weather", {"city": "B"}),
                                                              call("c", "ping", {})]},
        tool("c", "pong"), tool("a", "A: 10C")]
    unknown = tool_loop(final=False, parallel=True, out_of_order=True)
    unknown[-1] = tool("call_9", "unmatched")
    listed = tool_loop(final=False)[:-1] + [tool("call_1", [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}])]
    alias = tool_loop(final=False)
    del alias[2]["reasoning_content"]
    alias[2]["reasoning"] = "Need weather (echoed as reasoning)."
    alias.append({"role": "assistant", "content": "It is 18C.", "reasoning": "Got it."})
    both = tool_loop()
    both[-1]["reasoning"] = "not this one"
    nested = {"q": "a<b & \"c\"", "filters": {"lang": "en", "deep": {"x": 1, "y": [1, 2.5, "z", True]}, "skip": None},
              "ids": [3, 4, 5], "off": False, "u": "café 中文 😀", "empty_list": [], "empty_map": {}, "none_top": None,
              "list_of_maps": [{"a": 1}, {"b": [True, False]}]}
    c = {
        "plain_single": dict(messages=hi),
        "plain_single_nogen": dict(messages=hi, generation=False),
        "plain_multi": dict(messages=chat),
        "plain_multi_chat": dict(messages=chat, thinking=False),
        "user_content_list": dict(messages=[{"role": "user", "content": [{"type": "text", "text": "a"},
                                                                         {"type": "text", "text": "b"}]}]),
        "system_user": dict(messages=chat[:2]),
        "developer_user": dict(messages=[{"role": "developer", "content": "Dev rules"}] + hi),
        "system_then_developer": dict(messages=[{"role": "system", "content": "Sys"},
                                                {"role": "developer", "content": "Dev"}] + hi),
        "root_role": dict(messages=[{"role": "root", "content": "Root rules"}] + hi,
                          skip="role root: the server accepts the OpenAI roles (system, developer, user, assistant, "
                               "tool); root is MiniMax's own high-priority role"),
        "mid_system": dict(messages=later + [{"role": "system", "content": "later"}, {"role": "user", "content": "c"}]),
        "mid_developer": dict(messages=later + [{"role": "developer", "content": "later"},
                                                {"role": "user", "content": "c"}]),
        "effort_low": dict(messages=hi, effort="low"),
        "effort_xhigh": dict(messages=chat, effort="xhigh"),
        "effort_chat_dropped": dict(messages=hi, thinking=False, effort="high"),
        "tool_pending": dict(messages=tool_loop(final=False), tools=TOOLS),
        "tool_final": dict(messages=tool_loop(), tools=TOOLS),
        "tool_final_chat": dict(messages=tool_loop(), tools=TOOLS, thinking=False),
        "tool_pending_no_tools": dict(messages=tool_loop(final=False)),
        "tool_two_rounds": dict(messages=two_rounds, tools=TOOLS),
        "tool_two_rounds_pending": dict(messages=two_rounds[:-2], tools=TOOLS),
        "parallel_in_order": dict(messages=tool_loop(parallel=True), tools=TOOLS),
        "parallel_out_of_order": dict(messages=tool_loop(J(ARGS_W), args_s=J(ARGS_S), parallel=True, out_of_order=True,
                                                         final=False), tools=TOOLS),
        "parallel_out_of_order_final": dict(messages=tool_loop(parallel=True, out_of_order=True), tools=TOOLS),
        "parallel_partial_out_of_order": dict(messages=three, tools=TOOLS),
        "parallel_unknown_id_kept": dict(messages=unknown, tools=TOOLS),
        "tool_content_list": dict(messages=listed, tools=TOOLS),
        "tool_then_user_text": dict(messages=tool_loop(final=False) + [{"role": "user", "content": "Also Lyon?"}],
                                    tools=TOOLS),
        "tool_then_system": dict(messages=tool_loop(final=False) + [{"role": "system", "content": "Answer in French."}],
                                 tools=TOOLS),
        "args_dict": dict(messages=tool_loop(ARGS_W), tools=TOOLS),
        "args_json_string": dict(messages=tool_loop(J(ARGS_W)), tools=TOOLS),
        "args_compact_string": dict(messages=tool_loop(J(ARGS_W, separators=(",", ":"))), tools=TOOLS),
        "args_double_encoded": dict(messages=tool_loop(J(J(ARGS_W))), tools=TOOLS),
        "args_invalid_json": dict(messages=tool_loop("{city: Paris"), tools=TOOLS),
        "args_array_string": dict(messages=tool_loop("[1, 2]"), tools=TOOLS),
        "args_empty_object_string": dict(messages=tool_loop("{}"), tools=TOOLS),
        "args_empty_dict": dict(messages=tool_loop({}), tools=TOOLS),
        "args_null": dict(messages=_with_args(tool_loop(), None), tools=TOOLS),
        "args_empty_string": dict(messages=tool_loop(""), tools=TOOLS),
        "args_none_values": dict(messages=tool_loop({"city": "Paris", "units": None}), tools=TOOLS),
        "args_nested_unicode": dict(messages=tool_loop(nested), tools=TOOLS),
        "args_nested_string": dict(messages=tool_loop(J(nested, ensure_ascii=True)), tools=TOOLS),
        "content_null_with_calls": dict(messages=tool_loop(content=None, rc=None), tools=TOOLS),
        "content_text_with_calls": dict(messages=tool_loop(content="Checking.", rc=None), tools=TOOLS),
        "reasoning_field_alias": dict(messages=alias, tools=TOOLS),
        "reasoning_content_wins": dict(messages=both, tools=TOOLS),
        "inline_think_content": dict(messages=hi + [{"role": "assistant", "content": "<mm:think>\nr\n</mm:think>\nok"},
                                                    {"role": "user", "content": "more"}]),
        "unicode": dict(messages=[{"role": "system", "content": "é中\U0001f600"}, {"role": "user", "content": "é中\U0001f600"},
                                  {"role": "assistant", "content": "ü", "reasoning_content": "☃"},
                                  {"role": "user", "content": "⁣x"}], tools=TOOLS),
        "tool_without_call_refused": dict(messages=hi + [{"role": "assistant", "content": "plain"},
                                                         tool("call_1", "oops")]),
    }
    return c


def _with_args(messages: list, args: object) -> list:
    messages[2]["tool_calls"][0]["function"]["arguments"] = args
    return messages


def upstream(messages: list) -> list:
    """The documented cleaning before the template: JSON-string arguments parsed, None arguments dropped."""
    def drop_none(v):
        return {k: drop_none(x) for k, x in v.items() if x is not None} if isinstance(v, dict) else v
    out = copy.deepcopy(messages)
    for m in out:
        for c in m.get("tool_calls") or []:
            fn = c.get("function")
            if isinstance(fn, dict) and isinstance(fn.get("arguments"), str):
                try:
                    fn["arguments"] = json.loads(fn["arguments"])
                except ValueError:
                    pass
            if isinstance(fn, dict) and isinstance(fn.get("arguments"), dict):
                fn["arguments"] = drop_none(fn["arguments"])
    return out


def serve(messages: list) -> tuple[list, list]:
    """messages.zig ``forTemplate``: what the server hands the template, and the serving rules that changed it."""
    rules = []
    out, instructions = [], []
    for m in copy.deepcopy(messages):
        c = m.get("content")
        if isinstance(c, list):
            m["content"] = "".join(p["text"] for p in c)
        elif c is None:
            m["content"] = ""
        if m["role"] == "assistant" and m.get("reasoning_content") is None and isinstance(m.get("reasoning"), str):
            m["reasoning_content"] = m["reasoning"]
            rules.append("reasoning_alias")
        if m["role"] in ("system", "developer"):
            if not out:
                instructions.append(m)
                continue
            m["role"] = LATE_SYSTEM
        out.append(m)
    if instructions:
        out.insert(0, dict(instructions[0], role="system", content="\n\n".join(i["content"] for i in instructions)))
        if len(instructions) > 1:
            rules.append("leading_instructions_merged")
    i = 0
    while i < len(out):  # a run of tool results in the calling turn's order, when every id names a distinct call
        j = i
        while j < len(out) and out[j]["role"] == "tool":
            j += 1
        if j == i:
            i += 1
            continue
        prev = out[i - 1] if i > 0 else {}
        ids = [c.get("id") for c in prev.get("tool_calls") or []] if prev.get("role") == "assistant" else []
        ranks = [ids.index(t.get("tool_call_id")) if t.get("tool_call_id") in ids and isinstance(t.get("tool_call_id"), str)
                 else None for t in out[i:j]]
        if ids and None not in ranks and len(set(ranks)) == len(ranks) and ranks != sorted(ranks):
            out[i:j] = [t for _, t in sorted(zip(ranks, out[i:j]), key=lambda p: p[0])]
            rules.append("tool_results_reordered")
        i = j
    for m in out:
        for c in m.get("tool_calls") or []:
            fn = c.get("function")
            if not isinstance(fn, dict):
                continue
            args = fn.get("arguments")
            if isinstance(args, dict):
                continue
            if args is None or (isinstance(args, str) and args.strip(" \t\r\n") == ""):
                fn["arguments"] = {}
                rules.append("empty_arguments")
                continue
            try:
                parsed = json.loads(args) if isinstance(args, str) else args
            except ValueError:
                parsed = None
            if not isinstance(parsed, dict):
                parsed = {"_invalid_arguments": args}
                rules.append("invalid_arguments")
            fn["arguments"] = parsed
    return out, sorted(set(rules))


def render(tpl, adaptive_switch: bool, messages: list, req: dict) -> tuple[str | None, str | None]:
    """``prompt.renderIds``' options through hf_text's context: effort only with thinking, the template's mode word."""
    thinking = req.get("thinking", True)
    kw = dict(messages=messages, add_generation_prompt=req.get("generation", True), enable_thinking=thinking)
    if req.get("tools"):
        kw["tools"] = req["tools"]
    if thinking and req.get("effort") is not None:
        kw["reasoning_effort"] = req["effort"]
    kw["thinking_mode"] = ("enabled" if thinking else "disabled") if adaptive_switch else ("thinking" if thinking else "chat")
    try:
        return tpl.render(**kw), None
    except jinja2.exceptions.TemplateError as e:
        return None, str(e)
    except (AttributeError, TypeError) as e:
        return None, f"{type(e).__name__}: {e}"


def main() -> int:
    if len(sys.argv) < 3 or any("=" not in a for a in sys.argv[2:]):
        print(USAGE)
        return 2
    env = environment()
    lines, notes = [], []
    for spec in sys.argv[2:]:
        label, path = spec.split("=", 1)
        source = Path(path).read_text(encoding="utf-8")
        tpl = env.from_string(source)
        switch = "thinking_mode == \"adaptive\"" in source
        for name, req in cases().items():
            line = {"name": name, "template": label, "messages": req["messages"], "tools": req.get("tools"),
                    "thinking": req.get("thinking", True), "effort": req.get("effort"),
                    "generation": req.get("generation", True)}
            if "skip" in req:
                line["skip"] = req["skip"]
                lines.append(line)
                continue
            served, rules = serve(req["messages"])
            text, error = render(tpl, switch, served, req)
            raw, raw_error = render(tpl, switch, upstream(req["messages"]), req)
            if req.get("effort") is not None and not line["thinking"]:
                rules.append("effort_needs_thinking")
            line.update(rules=rules, expected=text, error=error)
            if (raw, raw_error) != (text, error):
                line.update(upstream=raw, upstream_error=raw_error)
            lines.append(line)
        notes.append(f"{label} {hashlib.sha256(source.encode()).hexdigest()[:12]}")
    Path(sys.argv[1]).write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in lines), encoding="utf-8")
    differ = sum("upstream" in x for x in lines)
    print(f"{len(lines)} requests ({sum('skip' in x for x in lines)} skipped, {sum(bool(x.get('error')) for x in lines)} "
          f"refused, {differ} differ from upstream cleaning alone) from {', '.join(notes)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
