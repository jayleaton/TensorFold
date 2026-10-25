"""Regenerates template_goldens.json: chat_template.jinja rendered by Jinja2 (3.1.x) as transformers renders it."""
# Usage: python3 -I gen_template_goldens.py chat_template.jinja template_goldens.json
import json, sys
import jinja2
import jinja2.ext
from jinja2.sandbox import ImmutableSandboxedEnvironment

def raise_exception(message):
    raise jinja2.exceptions.TemplateError(message)

def tojson(x, ensure_ascii=False, indent=None, separators=None, sort_keys=False):
    return json.dumps(x, ensure_ascii=ensure_ascii, indent=indent, separators=separators, sort_keys=sort_keys)

env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True, extensions=[jinja2.ext.loopcontrols])
env.filters["tojson"] = tojson
env.globals["raise_exception"] = raise_exception
tpl = env.from_string(open(sys.argv[1], encoding="utf-8").read())

WEATHER = {"type": "function", "function": {"name": "get_weather", "description": "Get the weather for a city (été, 天气)",
    "parameters": {"type": "object", "properties": {"city": {"type": "string"}, "unit": {"type": "string", "enum": ["c", "f"]}}, "required": ["city"]}}}
SEARCH = {"type": "function", "function": {"name": "search", "description": "Search \"the\" web",
    "parameters": {"type": "object", "properties": {"q": {"type": "string"}, "filters": {"type": "object"}, "ids": {"type": "array", "items": {"type": "integer"}}}}}}
U = lambda c: {"role": "user", "content": c}
A = lambda c, **kw: dict({"role": "assistant", "content": c}, **kw)
TC = lambda name, args: {"type": "function", "function": {"name": name, "arguments": args}}
TOOL = lambda c: {"role": "tool", "content": c}
MODES = [None, "enabled", "disabled", "adaptive"]

base = []
def case(name, messages, tools=None, ctx=None, gen=True):
    base.append({"name": name, "messages": messages, "tools": tools, "context": ctx or {}, "add_generation_prompt": gen})

case("user_only", [U("Hello")])
case("user_only_nogen", [U("Hello")], gen=False)
case("system_user", [{"role": "system", "content": "Be brief."}, U("Hi")])
case("developer_user", [{"role": "developer", "content": "Dev rules"}, U("Hi")])
case("root_dev_user", [{"role": "root", "content": "Root rules", "current_date": "2026-10-22", "current_location": "Zürich"},
                       {"role": "developer", "content": "Dev rules"}, U("Hi")])
case("root_only_empty_dev", [{"role": "root", "content": "Root rules"}, U("Hi")])
case("empty_system_content", [{"role": "system", "content": ""}, U("Hi")])
case("parts_content", [U([{"type": "text", "text": "look "}, {"type": "image"}, {"type": "video"}, {"type": "frame"}, "tail", {"type": "other"}])])
case("none_content", [U(None)])
case("mid_system", [U("a"), A("b"), {"role": "system", "content": "later"}, U("c")])
case("effort_ctx", [U("x")], ctx={"effort": "high", "template_model_id": "mm3.1"})
case("reasoning_effort_ctx", [U("x")], ctx={"reasoning_effort": "low", "effort": "high"})
case("model_identity_ctx", [{"role": "system", "content": ""}, U("x")], ctx={"model_identity": "I am Bob."})
case("assistant_plain", [U("q"), A("answer"), U("q2")])
case("assistant_reasoning_field", [U("q"), A("answer", reasoning_content="step 1\nstep 2"), U("q2")])
case("assistant_reasoning_empty_field", [U("q"), A("answer", reasoning_content=""), U("q2")])
case("assistant_inline_think", [U("q"), A("<mm:think>\n\nmy thoughts\n</mm:think>\n\nfinal answer"), U("q2")])
case("assistant_inline_close_only", [U("q"), A("\nraw thoughts\n</mm:think>\nfinal"), U("q2")])
case("assistant_empty", [U("q"), A(""), U("q2")])
case("tool_call_single", [U("weather?"), A("", tool_calls=[TC("get_weather", {"city": "Paris", "unit": "c"})])], [WEATHER, SEARCH])
case("tool_call_openai_shape", [U("w"), A("let me check", tool_calls=[TC("get_weather", {"city": "Zürich"})])], [WEATHER])
case("tool_call_flat_shape", [U("w"), A("x", tool_calls=[{"name": "get_weather", "arguments": {"city": "Rome"}}])], [WEATHER])
case("tool_call_nested", [U("s"), A("", tool_calls=[TC("search", {"q": "a<b & \"c\"", "filters": {"lang": "en", "deep": {"x": 1, "y": [1, 2.5, "z"]}, "skip": None, "flag": True},
        "ids": [3, 4, 5], "none_arg": None, "off": False, "pi": 3.14159, "big": 12345678901234567890, "u": "café 中文", "empty_list": [], "empty_map": {}, "list_of_maps": [{"a": 1}, {"b": [True, False]}]})])], [SEARCH])
case("tool_call_multi", [U("both"), A("", tool_calls=[TC("get_weather", {"city": "A"}), TC("search", {"q": "B"})])], [WEATHER, SEARCH])
case("tool_call_no_args", [U("x"), A("", tool_calls=[TC("ping", {})])])
case("tool_result_single", [U("w"), A("", tool_calls=[TC("get_weather", {"city": "P"})]), TOOL("sunny 21C"), U("thanks")], [WEATHER])
case("tool_result_multi", [U("both"), A("", tool_calls=[TC("get_weather", {"city": "A"}), TC("search", {"q": "B"})]), TOOL("res A"), TOOL("res B"), A("done")], [WEATHER, SEARCH])
case("tool_result_parts", [U("w"), A("", tool_calls=[TC("get_weather", {"city": "P"})]),
        TOOL([{"type": "text", "text": "t1"}, {"type": "image"}, {"output": "o1"}, {"type": "video"}, {"type": "frame"}, {"type": "text"}])], [WEATHER])
case("tool_result_no_call_error", [U("w"), A("plain"), TOOL("oops")])
case("tool_result_first_error", [TOOL("oops")])
case("tool_result_then_assistant_think", [U("w"), A("<mm:think>\nr\n</mm:think>\nok", tool_calls=[TC("get_weather", {"city": "P"})]), TOOL("21"), A("It is 21", reasoning_content="used tool")], [WEATHER])
case("unicode_everywhere", [{"role": "system", "content": "é中\U0001f600"}, U("é中\U0001f600"), A("ü", reasoning_content="☃")], [WEATHER])
case("no_messages", [])
case("tools_empty_list", [U("x")], [])

cases = []
for c in base:
    modes = MODES if c["name"] in ("user_only", "system_user", "root_dev_user", "tool_call_single", "assistant_reasoning_field", "tool_result_multi") else [None]
    for m in modes:
        for gen in ([True, False] if c["name"] == "user_only" and m is not None else [c["add_generation_prompt"]]):
            ctx = dict(c["context"])
            if m is not None: ctx["thinking_mode"] = m
            name = c["name"] + ("" if m is None else "." + m) + ("" if gen == c["add_generation_prompt"] or c["name"] != "user_only" else ".nogen")
            if c["name"] == "user_only" and m is not None and not gen: name = c["name"] + "." + m + ".nogen"
            cases.append({"name": name, "messages": c["messages"], "tools": c["tools"], "context": ctx, "add_generation_prompt": gen})

seen = set(); out = []
for c in cases:
    if c["name"] in seen: continue
    seen.add(c["name"])
    kw = dict(c["context"]); kw["messages"] = c["messages"]; kw["add_generation_prompt"] = c["add_generation_prompt"]
    if c["tools"] is not None: kw["tools"] = c["tools"]
    rec = {"name": c["name"], "messages": c["messages"], "tools": c["tools"], "context": c["context"], "add_generation_prompt": c["add_generation_prompt"]}
    try:
        rec["output"] = tpl.render(**kw)
    except jinja2.exceptions.TemplateError as e:
        rec["error"] = str(e)
    out.append(rec)
json.dump(out, open(sys.argv[2], "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(len(out), "cases,", sum("error" in r for r in out), "errors")
