# Reusable arithmetic graph invocation plans

The development-only `runner_graph_plans.py` lowers syntax to JSON-compatible invocation
plans. It never evaluates operand results or emits FFN weights. The native entrypoint
should accept a bound Program packet, not Python source. Load reusable graphs for widths
1, 2, 4, 8, 16, 32 and 62 at boot; requests select them and bind raw tensor Values.

## Native packet and named plans

The codec follows `tools/runner-graph-native.md` exactly. An inline packet is
`{"version":1,"values":["12","-3"],"ops":[{"op":"add","a":0,"b":1,"width":4}],"result":2}`.
References index values first, then every operation output in append order. Multi-output
calls use `args` and `outputs:2`. The live path uses only
`{"version":1,"plan":"add-4","values":["12","-3"]}`. There are no request-side ops,
weights, dimensions, syntax parsing, or compilation on this path.

`binary_plan(op,width,exponent=None)` returns `{name,inputs,literals,ops,result}` for
the manifest's `plans` list. References index requested inputs first, then built-in
`literals`, then outputs. `plan_id` and `binary_request(source)` select the exact ID:
`add-N`, `sub-N`, `mul-N`, `div-N`, `mod-N`, or `pow-E-62` for exponent E in 0..999.
`binary_request` chooses buckets from literal lengths, reserves one extra digit for
add/sub and the sum of input digit lengths for mul, and caps at 62. Powers share
width-62 mul graphs across all exponent schedules. Their single input is the base;
the exponent is plan selection metadata. The built-in pools contain only zero/one.
Exponent zero checks base error through mul-by-zero and add-one FFNs.

`loop_plan(shape,bound,width=62)` names plans `loop-SHAPE-BOUND-WIDTH`. Shapes are
`sum` (range 0..bound), `squares`, `factorial` (1..bound), `fib` and `list_sum`.
Bounds are 0..128. All except `list_sum` take zero request inputs and bind offline
seed/index literals; `list_sum` takes bound request operands. Fibonacci implements
simultaneous tuple assignment. `plan_catalog()` emits the complete default catalog.

## Native tensor transport

The native helper's `bind` Op supports both required tensor transfers. No new opcode
or arbitrary dimensions are needed. To extract A's digit i into digit zero:
`{"op":"bind","copies":[{"ref":A,"from":i,"to":0,"digits":1}],"flags":A}`.
Its source sign is irrelevant to divstep, which reads only digit/error lanes.

To insert divstep output D's quotient digit into Q at position i:
`{"op":"bind","copies":[{"ref":Q,"from":0,"to":0,"digits":N},{"ref":D,"from":0,"to":i,"digits":1}],"flags":D}`.
D's sign is zero and its sticky error includes every preceding division step.
Copies execute in order, preserve template reference columns at invocation, and do
no numeric decoding. A width-N division plan has 2N binds and N divstep calls followed
by one divfinal; its only built-in literal is zero. Generated plans use precisely
this native schema. Bind copies and flag refs are also remapped during SSA expansion.

## Offline SSA representation

`compile_expression` and `compile_program` retain a development IR before encoding:

```json
{
  "format": "tensorfold.runner.graph-plan",
  "version": 1,
  "literals": [{"digits": [7], "sign": 1, "error": 0}],
  "nodes": [{"op": "add", "a": {"literal": 0}, "b": {"literal": 0}, "width": 2}],
  "rootref": {"node": 0}
}
```

Digits are little-endian decimal lanes, sign is 1 for negative, and error is sticky.
Each reference has exactly one key: `literal` indexes the literal binding table and
`node` indexes a previous call's result. All nodes have exactly `op`, `a`, `b`, `width`.
Operations are `add`, `sub`, `mul`, `div` (Python floor division), and `mod`.
Powers are unrolled into multiply calls using exponentiation by squaring. Loops, sums,
list literals and comprehensions become fixed call sequences; no arithmetic answers
are folded. Widths follow syntactic upper bounds, capped at the 62-digit register limit.
Before a call, zero-extend shorter inputs to its width. Never truncate nonzero high
lanes silently: truncation makes the input error sticky. Zero signs are normalized.

## Native sequencing

`div` and `mod` select a fixed width-specific division plan. Initialize unsigned Q and
R to zero. For every dividend digit from high to low, bind `(R, abs(B), nextdigit)` to
the precompiled `divstep` graph. Move output 0 directly to R; insert output 1's digit 0
into that iteration's fixed quotient position. After exactly width steps, bind
`(Q, R, original A, original B)` to `divfinal`. Its outputs are signed floor quotient
and remainder; `div` selects output 0 and `mod` selects output 1. Metadata selects
lanes; arithmetic and zero/sign decisions execute in FFNs. No text roundtrip is used.

`divstep` shifts the remainder into width+1 lanes, performs nine guarded unsigned
subtractions, and emits an unsigned remainder and quotient digit. `divfinal` corrects
opposite signs with nonzero remainder using `|Q|+1` and `|B|-R`, normalizes zero signs,
and propagates input errors and divisor-zero errors to both results.

`native_packet(plan)` lowers the development IR to the native inline codec, expanding
div/mod into tensor transfers and divstep/divfinal calls. Generic SSA requests carry
the data-only invocation sequence, including offline unrolled T5 bodies, and use the
same preloaded FFN kernels. Validate all sizes, digit ranges, flags,
widths and backward references before binding. Python parsing is offline fixture tooling.

## Corpus and verification handoff

The CLI reads the original `goldens-10000/cases.jsonl.gz` and manifest, routes every
row before examining outputs, and retains all routable requests plus the first 20
samples in original order. It routes only
integer add/sub/mul/floor-div/mod/pow syntax, including mixed expressions. Fractions,
decimal syntax and gcd abstain; the original loop-squares fixture has a fixed plan.
Baseline `outcome` including `machine_limit`, exact machine output bytes, mathematical
expected output, and exit code are copied unchanged. Reference execution remains the
parent's job using Qwenputer's `calc.machine` and register limits. This tool never uses
Python `eval`, `calc.expected`, or host bignums to compute answers. The corpus retains
the original manifest's upstream MIT credit and full license text.

Corpus entries have two execution modes, determined from syntax before reading any
reference outcome: `named` for simple signed binary operations and the known loop,
and `generic_ssa` for compound expressions. Their `request` contains respectively a
named operand-only packet or a generic inline SSA packet. Generic entries have no
`named_plan`; no corpus-ID plan or benchmark-specific plan enters the production bank.
The catalog contains only general binary, exponent and bounded loop plans.

Legacy compatibility is separate: `legacy_backend` is `calc` for every compound
expression and every unsupported source, by structural policy independent of answers.
Thus generic SSA feature checks cannot claim legacy token/output parity when the old
machine reports a complexity limit. Use the original calc fallback for those legacy
checks and count it. Named entries select `graph`; their exact original outputs still
require comparison. `mode_counts`, `legacy_backend_counts`, `baseline_machine_limit_cases`
and `machine_limit_by_mode` report these populations, including unsupported rows.
`machine_limit_cases` counts the limits among graph-routable entries. A generic result
matching the mathematical expected output does not establish a match to `final_output`.

Example parent commands: `python3 tools/runner_graph_plans.py --catalog --output plans.json`
and `python3 tools/runner_graph_plans.py --corpus PATH/goldens-10000 --samples 20 --output corpus.json`.

Verification and integration limits are recorded in `runner-graph-results.md`.
