"""Regenerates tokenizer_cases.json with the Python `tokenizers` library (0.23.x) from the checkpoint's tokenizer.json.
Usage: python3 -I gen_tokenizer_cases.py tokenizer.json template_goldens.json tokenizer_cases.json"""
import json, sys
from tokenizers import Tokenizer
tok = Tokenizer.from_file(sys.argv[1])
goldens = json.load(open(sys.argv[2], encoding="utf-8"))
texts = [
    "Hello, world!",
    "  leading and trailing  ",
    "tabs\tand\r\nCRLF\n\n\nblank lines\n",
    "numbers 1 12 123 1234 12345 3.14159 1e-5 0x1F",
    "It's we're they've I'm you'll she'd DON'T O'Neil",
    "def f(x):\n    return x**2  # square\n\n\tif x:\n\t\tpass\n",
    "caf\u00e9 vs cafe\u0301 (NFC folds the second), \u00dcber, na\u00efve, \u0130stanbul",
    "\u4e2d\u6587\u6d4b\u8bd5\uff0c\u65e5\u672c\u8a9e\u3001\ud55c\uad6d\uc5b4 \u0e44\u0e17\u0e22 \u0627\u0644\u0639\u0631\u0628\u064a\u0629 \u0420\u0443\u0441\u0441\u043a\u0438\u0439",
    "emoji \U0001f600\U0001f680 \u2603 \u2764\ufe0f \U0001f468\u200d\U0001f469\u200d\U0001f467",
    "url https://example.com/a/b?c=d&e=f#g and path /usr/local/bin///x",
    "]~b]user\nhi[e~[\n]~b]ai\n",
    "]~!b[]~b]system\nS[e~[\n",
    "<mm:think>reasoning</mm:think>answer",
    "]<]minimax[>[<tool_call>\n]<]minimax[>[<invoke name=\"f\">]<]minimax[>[<a>1]<]minimax[>[</a>]<]minimax[>[</invoke>\n]<]minimax[>[</tool_call>",
    "<fim_prefix>x<fim_middle>y<fim_suffix>z",
    "adjacent specials ]~b]]~b][e~[[e~[ end",
    "x" * 300,
    "ab " * 100,
    "",
    " ",
    "\n",
    "<mm:think",
    "]<]minimax[>",
]
for g in goldens:
    if g["name"] in ("user_only.adaptive", "tool_call_nested", "tool_result_multi.enabled", "assistant_reasoning_field.disabled", "unicode_everywhere"):
        texts.append(g["output"])
cases = []
for t in texts:
    e = tok.encode(t, add_special_tokens=False)
    cases.append({"text": t, "ids": e.ids, "decoded": tok.decode(e.ids, skip_special_tokens=False),
                  "decoded_skip": tok.decode(e.ids, skip_special_tokens=True)})
want = ["]~b]", "[e~[", "]~!b[", "]<]minimax[>[", "<mm:think>", "</mm:think>", "]<]image[>[", "]<]video[>[", "]<]frame[>["]
json.dump({"vocab_size": tok.get_vocab_size(), "specials": {k: tok.token_to_id(k) for k in want}, "cases": cases},
          open(sys.argv[3], "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(len(cases), "cases", {k: tok.token_to_id(k) for k in want})
