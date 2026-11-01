"""TF_DSV41_RUNNER_TRACE log -> per-phase round table: baseline, trigger, the rounds after each resume, holds."""
import argparse
import json
import statistics
import sys

MARK = '{"rt":'


def records(paths):
    """Every trace object in the files (server log lines included: each object starts at its marker)."""
    for path in paths:
        with (sys.stdin if path == "-" else open(path, encoding="utf-8", errors="replace")) as f:
            for line in f:
                at = line.find(MARK)
                if at < 0:
                    continue
                try:
                    yield json.loads(line[at:])
                except json.JSONDecodeError:
                    continue


def phase(r, around, split):
    """A round's phase: trigger, after+k (k = rounds since the resume, the feeding round is 1), or baseline."""
    if r["phase"] == "trigger":
        return "trigger"
    since = r.get("since", -1)
    if r["phase"] == "post":
        return f"after+{since}" if since <= split else f"after+{split + 1}..{around}"
    # a round before a hold that is still inside an earlier resume's window is not a baseline
    return "baseline" if since < 0 or since > around else "near-fire"


SPEC_WHY = ("commit", "slots", "unused", "pending", "other")


def drops(r):
    """A round's dropped speculative DSpark passes, every reason."""
    return sum(r.get(f"spec_{w}", 0) for w in SPEC_WHY)


def why(g):
    """The phase's drops by reason (c=commit, s=slots, u=unused, p=pending, o=other)."""
    return " ".join(f"{w[0]}{n}" for w in SPEC_WHY if (n := sum(r.get(f"spec_{w}", 0) for r in g)))


def mean(xs):
    return statistics.fmean(xs) if xs else float("nan")


def median(xs):
    return statistics.median(xs) if xs else float("nan")


def table(rounds, around, split, streams):
    """The phases' rows: means per round, sums of rare events, wall and GPU change against the baseline."""
    groups = {}
    for r in rounds:
        if streams is not None and r.get("streams") != streams:
            continue
        groups.setdefault(phase(r, around, split), []).append(r)
    order = ["baseline", "near-fire", "trigger"] + [f"after+{k}" for k in range(1, split + 1)] + [f"after+{split + 1}..{around}"]
    base = groups.get("baseline", [])
    base_wall = mean([r["wall_us"] for r in base])
    base_gpu = mean([r.get("gpu_us", 0) for r in base])
    head = ["phase", "n", "wall ms", "med", "gpu ms", "host ms", "gap ms", "wall %", "gpu %", "rows", "drafted",
            "kept", "got", "tok/s", "held next", "forced", "spec hit", "spec drop", "drop why", "engram miss",
            "captures", "builds", "evicted", "eager", "lost"]
    out = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for name in order:
        g = groups.get(name)
        if not g:
            continue
        wall = [r["wall_us"] / 1e3 for r in g]
        gpu = [r.get("gpu_us", 0) / 1e3 for r in g]
        gap = [r["gap_us"] / 1e3 for r in g]
        got = sum(r["got"] for r in g)
        span = sum(wall) + sum(gap)
        windows = sum(r.get("windows", 0) for r in g)
        rows = sum(r.get("rows", 0) for r in g) / windows if windows else float("nan")
        cells = [name, len(g), f"{mean(wall):.2f}", f"{median(wall):.2f}", f"{mean(gpu):.2f}",
                 f"{mean(wall) - mean(gpu):.2f}", f"{mean(gap):.2f}",
                 f"{100 * (mean(wall) * 1e3 / base_wall - 1):+.1f}" if base else "",
                 f"{100 * (mean(gpu) * 1e3 / base_gpu - 1):+.1f}" if base and base_gpu > 0 else "",
                 f"{rows:.2f}", f"{mean([r['drafted'] for r in g]):.2f}", f"{mean([r['kept'] for r in g]):.2f}",
                 f"{got / len(g):.2f}", f"{1e3 * got / span:.1f}" if span else "",
                 f"{mean([r['held_next'] for r in g]):.2f}", sum(r["kind"] == "forced" for r in g),
                 f"{mean([r.get('spec_hit', 0) for r in g]):.2f}", f"{mean([drops(r) for r in g]):.2f}", why(g),
                 f"{mean([r.get('engram_misses', 0) for r in g]):.1f}",
                 sum(r.get("captures", 0) for r in g), sum(r.get("builds", 0) for r in g),
                 sum(r.get("evicted", 0) + r.get("dropped", 0) for r in g),
                 sum(r.get("eager", 0) for r in g), sum(r.get("lost", 0) for r in g)]
        out.append("| " + " | ".join(str(c) for c in cells) + " |")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("logs", nargs="*", default=["-"], help="trace files or server logs (- for stdin)")
    ap.add_argument("--split", type=int, default=4, help="rounds after a resume shown one by one")
    ap.add_argument("--streams", type=int, help="only rounds of steps with this many live streams (1: c=1)")
    a = ap.parse_args()
    objs = list(records(a.logs))
    opens = [o for o in objs if o["rt"] == "open"]
    around = opens[-1]["rounds"] if opens else 12
    rounds = [o for o in objs if o["rt"] == "round"]
    resumes = [o for o in objs if o["rt"] == "resume"]
    holds = [o for o in objs if o["rt"] == "hold"]
    if len(opens) > 1:
        print(f"note: {len(opens)} trace opens (boots) in the input; rounds are pooled")
    print(f"{len(holds)} holds, {len(resumes)} resumes, {len(rounds)} rounds traced, {around} rounds each side")
    if resumes:
        held = [o["held_us"] / 1e3 for o in resumes]
        forced = [o["forced"] for o in resumes]
        print(f"held ms: mean {mean(held):.1f}, median {median(held):.1f}, max {max(held):.1f}; "
              f"forced tokens per resume: mean {mean(forced):.1f}")
    print("\n".join(table(rounds, around, min(a.split, around), a.streams)))
    print("wall: step time; gpu: the target windows' GPU time; host = wall - gpu (drafter, sampling, bookkeeping); "
          "gap: time before the step; spec hit/drop: speculative DSpark passes used / dropped per round (drop why: "
          "sums by reason); captures/builds/evicted/eager/lost: sums over the phase")


if __name__ == "__main__":
    main()
