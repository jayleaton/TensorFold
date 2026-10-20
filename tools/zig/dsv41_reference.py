"""DeepSeek-V4.1 reference goldens: conversations rendered by DeepSeek's own encoding/encoding.py, no serving rules."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

USAGE = ("python -I tools/zig/dsv41_reference.py ENCODING_DIR OUT.jsonl: ENCODING_DIR is the release's encoding/ "
         "(encoding.py, tests/). Lines use gen_template.py's fields; \"skip\" names a release golden no OpenAI request "
         "expresses.")
EFFORTS = {"low": 50, "high": 75, "max": 100}
TOOLS = [
    {"type": "function", "function": {"name": "get_weather", "description": "Weather for a city", "parameters": {
        "type": "object", "properties": {"city": {"type": "string"}, "days": {"type": "integer"},
                                         "units": {"type": "string"}}, "required": ["city"]}}},
    {"type": "function", "function": {"name": "search", "description": "Web search", "parameters": {
        "type": "object", "properties": {"q": {"type": "string"}, "filters": {"type": "object"}}}}},
]
ARGS_W = {"city": "Paris", "days": 3}
ARGS_S = {"q": "naïve \"quoted\" 東京\nline2", "filters": {"lang": ["en", "fr"], "safe": True, "n": 1.5, "x": None}}
J = json.dumps


def call(cid: str, name: str, args: object) -> dict:
    return {"id": cid, "type": "function", "function": {"name": name, "arguments": args}}


def tool_turn(args_w, args_s=None, rc="Need weather.", parallel=False, out_of_order=False) -> list[dict]:
    """An assistant turn calling get_weather (and search when parallel), then its tool results."""
    calls = [call("call_1", "get_weather", args_w)]
    results = [{"role": "tool", "tool_call_id": "call_1", "content": "{\"temp\": 18}"}]
    if parallel:
        calls.append(call("call_2", "search", args_s))
        results.append({"role": "tool", "tool_call_id": "call_2", "content": "3 results"})
        if out_of_order:
            results.reverse()
    return [{"role": "assistant", "content": "", "reasoning_content": rc, "tool_calls": calls}] + results


def tool_loop(args_w, args_s=None, parallel=False, out_of_order=False, final=True) -> list[dict]:
    m = [{"role": "system", "content": "You are helpful."}, {"role": "user", "content": "Weather in Paris?"}]
    m += tool_turn(args_w, args_s, parallel=parallel, out_of_order=out_of_order)
    if final:
        m.append({"role": "assistant", "content": "It is 18C.", "reasoning_content": "Got it."})
    return m


def cases() -> dict[str, dict]:
    """Name -> request: messages, tools, thinking, budget, drop_thinking."""
    hi = [{"role": "user", "content": "Hi"}]
    chat = [{"role": "system", "content": "Be brief."}, {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello!", "reasoning_content": "greet"}, {"role": "user", "content": "2+2?"}]
    two_rounds = tool_loop(J(ARGS_W), final=False) + tool_turn(J({"city": "Lyon"}), rc="Also Lyon.") + [
        {"role": "assistant", "content": "Paris 18C, Lyon 18C.", "reasoning_content": "done"},
        {"role": "user", "content": "Thanks, and Nice?"}]
    listed = tool_loop(ARGS_W, final=False)[:-1] + [{"role": "tool", "tool_call_id": "call_1", "content": [
        {"type": "text", "text": "a"}, {"type": "text", "text": "b"}]}]
    mixed = tool_loop(ARGS_W, final=False) + [{"role": "user", "content": "Also Lyon?"}]
    mid_system = tool_loop(ARGS_W, final=False) + [{"role": "system", "content": "Answer in French."}]
    c = {
        "plain_single_think": dict(messages=hi),
        "plain_multi_think_drop": dict(messages=chat),
        "plain_multi_chat": dict(messages=chat, thinking=False),
        "effort_max": dict(messages=hi, budget=100),
        "effort_int_33": dict(messages=hi, budget=33),
        "mid_system": dict(messages=[{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Yo",
                                     "reasoning_content": "r"}, {"role": "system", "content": "Switch to French."}]),
        "tool_args_str_compact": dict(messages=tool_loop(J(ARGS_W, separators=(",", ":"))), tools=TOOLS),
        "tool_args_str_pending_result": dict(messages=tool_loop(J(ARGS_W), final=False), tools=TOOLS),
        "tool_args_double_encoded": dict(messages=tool_loop(J(J(ARGS_W))), tools=TOOLS),
        "tool_args_str_invalid": dict(messages=tool_loop("{city: Paris"), tools=TOOLS),
        "tool_args_str_empty_obj": dict(messages=tool_loop("{}"), tools=TOOLS),
        "tool_args_str_array": dict(messages=tool_loop("[1, 2]"), tools=TOOLS),
        "tool_args_str_unicode_escapes": dict(messages=tool_loop(J({"city": "Zürich \\ \"x\" \U0001F600", "days": -0.5e3})),
                                              tools=TOOLS),
        "parallel_dict": dict(messages=tool_loop(ARGS_W, ARGS_S, parallel=True), tools=TOOLS),
        "parallel_str_out_of_order": dict(messages=tool_loop(J(ARGS_W), J(ARGS_S, ensure_ascii=True), parallel=True,
                                                             out_of_order=True), tools=TOOLS),
        "multi_round_tool_loop_str": dict(messages=two_rounds, tools=TOOLS),
        "tool_result_list_content": dict(messages=listed, tools=TOOLS),
        "no_tools_tool_loop_pending": dict(messages=tool_loop(ARGS_W, final=False)),
        "no_tools_tool_loop_final": dict(messages=tool_loop(ARGS_W)),
        "no_tools_tool_loop_two_rounds": dict(messages=two_rounds[:-2]),
        "no_tools_parallel_out_of_order": dict(messages=tool_loop(J(ARGS_W), ARGS_S, parallel=True, out_of_order=True,
                                                                  final=False)),
        "no_tools_tool_loop_chat": dict(messages=tool_loop(ARGS_W, final=False), thinking=False),
        "no_tools_results_then_user_text": dict(messages=mixed),
        "no_tools_results_then_system": dict(messages=mid_system),
        "drop_false_tool_loop": dict(messages=tool_loop(J(ARGS_W)), drop_thinking=False),
        "user_content_list": dict(messages=[{"role": "user", "content": [{"type": "text", "text": "a"},
                                                                         {"type": "text", "text": "b"}]}]),
    }
    for mode in (True, False):
        name = "thinking" if mode else "chat"
        c[f"tool_args_dict_{name}"] = dict(messages=tool_loop(ARGS_W), tools=TOOLS, thinking=mode)
        c[f"tool_args_str_{name}"] = dict(messages=tool_loop(J(ARGS_W)), tools=TOOLS, thinking=mode)
    return c


def release_goldens(enc, tests: Path) -> dict[str, dict]:
    """The release's tests/test_input_N.json as requests (``test_encoding.py``: chat unless the case says thinking)."""
    out = {}
    for path in sorted(tests.glob("test_input_*.json")):
        case = enc.load_cases(str(path))[0]
        msgs, tools, skip = case["messages"], None, None
        n = path.stem.rsplit("_", 1)[1]
        for i, m in enumerate(msgs):
            if m.get("tools"):
                if i > 0 or m["role"] != "system":
                    skip = "tools on a message after the first: a request's tools join the first system message"
                tools = m.pop("tools")
            if m.get("task") is not None or "mask" in m:
                skip = "task / mask fields: internal classification prompts, not an OpenAI request"
            if isinstance(m.get("content"), list) and any(b.get("type") != "text" for b in m["content"]):
                skip = "image parts: the serving encoder renders them through its image hook and escapes a typed image token"
        effort = case.get("reasoning_effort")
        req = dict(messages=msgs, tools=tools, thinking=case.get("thinking_mode") == "thinking",
                   budget=75 if effort is None else EFFORTS.get(effort, effort),
                   golden=(tests / f"test_output_{n}.txt").read_text())
        if skip:
            req["skip"] = skip
        out[f"release_golden_{n}"] = req
    return out


def render(enc, req: dict) -> str:
    """encode_messages with the request's tools on a leading system message (a new empty one when there is none)."""
    msgs = copy.deepcopy(req["messages"])
    if req.get("tools"):
        if msgs and msgs[0]["role"] == "system":
            msgs[0]["tools"] = copy.deepcopy(req["tools"])
        else:
            msgs.insert(0, {"role": "system", "content": "", "tools": copy.deepcopy(req["tools"])})
    return enc.encode_messages(msgs, thinking_mode="thinking" if req.get("thinking", True) else "chat",
                               drop_thinking=req.get("drop_thinking", True), reasoning_effort=req.get("budget", 75))


def main() -> int:
    if len(sys.argv) != 3:
        print(USAGE)
        return 2
    enc_dir, out = Path(sys.argv[1]), Path(sys.argv[2])
    spec = importlib.util.spec_from_file_location("dsv41_reference", enc_dir / "encoding.py")
    enc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(enc)
    every = {**cases(), **release_goldens(enc, enc_dir / "tests")}
    lines = []
    for name, req in every.items():
        line = {"name": name, "messages": req["messages"], "tools": req.get("tools"), "thinking": req.get("thinking", True),
                "budget": req.get("budget", 75), "drop_thinking": req.get("drop_thinking", True), "response_format": None,
                "generation": True, "image_text": None, "error": None, "jinja": None}
        if "skip" in req:
            line.update(expected=None, skip=req["skip"])
        else:
            line["expected"] = render(enc, req)
            if "golden" in req and line["expected"] != req["golden"]:
                raise SystemExit(f"{name}: the request form does not reproduce the release's golden output")
        lines.append(json.dumps(line, ensure_ascii=False))
    out.write_text("\n".join(lines) + "\n")
    digest = hashlib.sha256((enc_dir / "encoding.py").read_bytes()).hexdigest()[:12]
    print(f"{len(lines)} cases ({sum('skip' in r for r in every.values())} release goldens skipped) from encoding.py {digest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
