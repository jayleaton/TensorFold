"""MiniMax-M3.1 tool-call goldens: calls rendered by the official chat template, and the arguments read back."""
import json
import sys

import jinja2
import jinja2.ext
from jinja2.sandbox import ImmutableSandboxedEnvironment

NS = "]<]minimax[>["
OPEN, CLOSE = NS + "<tool_call>", NS + "</tool_call>"
USAGE = "usage: python3 -I minimax_tool_calls.py chat_template.jinja minimax_tool_calls.json   (jinja2 3.1.x)"


def raise_exception(message):
    """The template's own error hook, as transformers defines it."""
    raise jinja2.exceptions.TemplateError(message)


def tojson(x, ensure_ascii=False, indent=None, separators=None, sort_keys=False):
    """transformers' tojson filter."""
    return json.dumps(x, ensure_ascii=ensure_ascii, indent=indent, separators=separators, sort_keys=sort_keys)


def fn(name, properties=None, defs=None):
    """An OpenAI function tool; no `properties` declares no schema at all."""
    f = {"name": name, "description": "fixture tool " + name}
    if properties is not None:
        f["parameters"] = {"type": "object", "properties": properties}
        if defs:
            f["parameters"]["$defs"] = defs
    return {"type": "function", "function": f}


ENTRY = {"type": "object", "properties": {"a": {"type": "integer"}, "b": {"type": "array", "items": {"type": "boolean"}}}}
WEATHER = fn("get_weather", {"city": {"type": "string"}, "unit": {"type": "string", "enum": ["c", "f"]}, "days": {"type": "integer"}})
STRS = fn("strs", {"text": {"type": "string"}, "lines": {"type": "string"}, "tags": {"type": "array", "items": {"type": "string"}}})
NUMBER = {"type": "number"}
TYPED = fn("typed", {
    "s": {"type": "string"}, "i": {"type": "integer"}, "f": NUMBER, "f2": NUMBER, "f3": NUMBER, "f4": NUMBER, "big": {"type": "integer"},
    "b": {"type": "boolean"}, "list": {"type": "array", "items": {"type": "integer"}},
    "obj": {"type": "object", "properties": {"lang": {"type": "string"}, "flag": {"type": "boolean"}, "deep": {"type": "object", "properties": {
        "x": {"type": "integer"}, "y": {"type": "array", "items": {"type": ["number", "string"]}}}}}},
    "objs": {"type": "array", "items": {"$ref": "#/$defs/entry"}}, "ref": {"$ref": "#/$defs/entry"},
    "empty_list": {"type": "array"}, "empty_map": {"type": "object"}, "any": {"anyOf": [{"type": "integer"}, {"type": "string"}]},
}, {"entry": ENTRY})
FREE = fn("free")
ALL = [WEATHER, STRS, TYPED, FREE]

CASES = []


def case(name, calls, tools=ALL, prose="", expect=None, lossy=None, skip=None):
    """One assistant turn: `calls` as (tool, arguments); `expect` overrides the arguments read back (with `lossy`)."""
    CASES.append({"name": name, "calls": calls, "tools": tools, "prose": prose, "expect": expect, "lossy": lossy, "skip": skip})


case("weather", [("get_weather", {"city": "Paris", "unit": "c", "days": 3})], prose="Let me check.")
case("strings", [("strs", {"text": "café 中文 \U0001f600 \"quoted\" 'single' <b>bold</b> & amp; a<b && c>d ]<] [>[ \\ end",
                           "lines": "line1\nline2\n\n  indented\ttab\r\n", "tags": ["<x>", "&", "a\"b", "", " "]})])
case("strings_look_typed", [("strs", {"text": "{\"a\": 1}", "lines": "12345", "tags": ["1", "2.5", "true", "null", "None", "[1]"]})])
case("strings_blank", [("strs", {"text": "   ", "lines": "\n", "tags": []})])
case("numbers", [("typed", {"i": 42, "f": 3.14159, "f2": 1e-07, "f3": 2.0, "f4": 1e16, "big": 12345678901234567890})])
case("negative_numbers", [("typed", {"i": -7, "f": -0.5, "big": -98765432109876543210})])
case("bools", [("typed", {"b": True, "obj": {"flag": False}}), ("typed", {"b": False})])
case("nested", [("typed", {"obj": {"lang": "en", "deep": {"x": 1, "y": [1, 2.5, "z"]}, "skip": None, "flag": True}, "s": "after"})])
case("list_of_objects", [("typed", {"objs": [{"a": 1}, {"b": [True, False]}, {"a": 2, "b": []}]})])
case("empties", [("typed", {"empty_list": [], "empty_map": {}, "s": "", "list": []})])
case("none_keys", [("typed", {"s": "x", "i": None, "obj": {"lang": None, "flag": True}})])
case("any_of", [("typed", {"any": 5}), ("typed", {"any": "abc"})])
case("ref", [("typed", {"ref": {"a": 2, "b": [False]}})])
case("unicode_keys", [("free", {"ключ": "значение"})])
case("parallel", [("get_weather", {"city": "A"}), ("typed", {"i": 1}), ("free", {}), ("get_weather", {"city": "B", "unit": "f"})],
     prose="Doing all three.")
case("no_args", [("free", {})])
case("no_schema", [("free", {"n": 5, "ok": True, "l": [1, 2], "m": {"k": "v"}, "s": "x", "e": [], "em": {}})],
     expect=[{"n": "5", "ok": "true", "l": ["1", "2"], "m": {"k": "v"}, "s": "x", "e": "", "em": ""}],
     lossy="no schema: a scalar is its text, an empty list or map renders as nothing")
case("no_schema_item_keys", [("free", {"m": {"item": "x"}})], expect=[{"m": ["x"]}],
     lossy="no schema: a mapping whose keys are all `item` reads as a list")
case("string_for_any_of", [("typed", {"any": "5"})], expect=[{"any": 5}], lossy="anyOf integer|string: the text 5 reads as the integer")
case("float_for_integer", [("typed", {"i": 2.5})], expect=[{"i": "2.5"}], lossy="a float under an integer schema stays its text")
case("none_in_list", [("typed", {"list": [1, None, 3]})], expect=[{"list": [1, "", 3]}],
     lossy="None inside a list renders an empty item, read back as the empty text")
case("text_for_integer", [("typed", {"i": "five", "b": "yes"})])
case("ns_in_string", [("strs", {"text": "a " + NS + " b"})], skip="a string holding the NS token renders markup the reader takes apart")
case("key_not_a_name", [("free", {"a b": 1})], skip="a key that is no XML name renders a tag the reader cannot take back")


def drop_none(v):
    """The arguments as the template renders them: None-valued keys left out at every level."""
    if isinstance(v, dict):
        return {k: drop_none(x) for k, x in v.items() if x is not None}
    if isinstance(v, list):
        return [drop_none(x) for x in v]
    return v


def main():
    """Renders every case's assistant turn and writes the fixture."""
    if len(sys.argv) != 3:
        sys.exit(USAGE)
    env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True, extensions=[jinja2.ext.loopcontrols])
    env.filters["tojson"] = tojson
    env.globals["raise_exception"] = raise_exception
    with open(sys.argv[1], encoding="utf-8") as f:
        tpl = env.from_string(f.read())
    out = []
    for c in CASES:
        calls = [{"type": "function", "function": {"name": n, "arguments": a}} for n, a in c["calls"]]
        messages = [{"role": "user", "content": "go"}, {"role": "assistant", "content": c["prose"], "tool_calls": calls}]
        text = tpl.render(messages=messages, tools=c["tools"], add_generation_prompt=False)
        turn = text[text.rindex("]~b]ai\n"):]
        start, end = turn.index(OPEN), turn.index(CLOSE) + len(CLOSE)
        expect = c["expect"] or [drop_none(a) for _, a in c["calls"]]
        out.append({"name": c["name"], "tools": c["tools"], "reply": (c["prose"] + " " if c["prose"] else "") + turn[start:end],
                    "calls": [{"name": n, "arguments": e} for (n, _), e in zip(c["calls"], expect)], "lossy": c["lossy"], "skip": c["skip"]})
    with open(sys.argv[2], "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
        f.write("\n")
    print(len(out), "cases,", sum(1 for r in out if r["lossy"]), "lossy,", sum(1 for r in out if r["skip"]), "skipped")


if __name__ == "__main__":
    main()
