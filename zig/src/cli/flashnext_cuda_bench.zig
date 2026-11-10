//! Flash Next CUDA CLI benches: the memory pool, prompt-file runs and shared-round tok/s with their prompts.
const std = @import("std");
const core = @import("core");
const lanes = @import("lanes");
const flashnext = @import("flashnext_mia");
const cli = @import("flashnext_cuda.zig");

const Engine = flashnext.Engine;
const Allocator = std.mem.Allocator;
const Options = cli.Options;
const parseIds = cli.parseIds;
const now = cli.now;
const since = cli.since;
const generate = cli.generate;
const sha12 = cli.sha12;

/// The tokens the sequences' budget holds at N streams: each grown a step in turn until refused (NoRoom) or full.
pub fn pool(gpa: Allocator, e: *Engine, o: Options) !u8 {
    const N = o.streams;
    const seqs = try gpa.alloc(*flashnext.forward.Seq, N);
    defer gpa.free(seqs);
    var made: usize = 0;
    defer for (seqs[0..made]) |q| e.freeSeq(q);
    for (seqs) |*q| {
        q.* = e.newSeq(e.max_len) catch |err| {
            std.debug.print("RESULT pool x{d}: only {d} sequences fit ({s})\n", .{ N, made, @errorName(err) });
            return 1;
        };
        made += 1;
    }
    var growing = N;
    const done = try gpa.alloc(bool, N);
    defer gpa.free(done);
    @memset(done, false);
    while (growing > 0) {
        for (seqs, done) |q, *d| {
            if (d.*) continue;
            const next = q.st.capacity + flashnext.state.grow_step;
            if (next > q.st.limit) {
                d.* = true;
                growing -= 1;
                continue;
            }
            e.f.grow(q, next) catch |err| switch (err) {
                error.NoRoom => {
                    d.* = true;
                    growing -= 1;
                },
                else => return err,
            };
        }
    }
    var rows: usize = 0;
    for (seqs) |q| rows += q.st.capacity;
    std.debug.print("RESULT pool x{d}: {d} cache rows ({d} a stream), {d:.2} GiB of {d:.2} GiB charged, {d:.0} bytes a row overall\n", .{ N, rows, rows / N, @as(f64, @floatFromInt(e.budget.used)) / (1 << 30), @as(f64, @floatFromInt(e.budget.limit)) / (1 << 30), @as(f64, @floatFromInt(e.budget.used)) / @as(f64, @floatFromInt(@max(rows, 1))) });
    return 0;
}

/// `bench`: every prompt file (ids) `repeat` times in one load; prefill seconds, tok/s and the reply's first tokens.
pub fn bench(gpa: Allocator, io: std.Io, e: *Engine, o: Options, files: []const []const u8) !u8 {
    var prompts: std.ArrayList([]u32) = .empty;
    defer {
        for (prompts.items) |p| gpa.free(p);
        prompts.deinit(gpa);
    }
    for (files) |list| {
        var it = std.mem.tokenizeScalar(u8, list, ',');
        while (it.next()) |path| {
            const text = try std.Io.Dir.cwd().readFileAlloc(io, path, gpa, .limited(1 << 28));
            defer gpa.free(text);
            try prompts.append(gpa, try parseIds(gpa, text));
        }
    }
    if (prompts.items.len == 0) {
        std.debug.print("bench: --tokens-file PATH[,PATH...]\n", .{});
        return 2;
    }
    const s: ?lanes.Sampling = if (o.sampling.temperature > 0) o.sampling else null;
    for (0..o.reps) |rep| {
        for (prompts.items) |p| {
            const g = try generate(gpa, io, e, p, o.max_tokens, s, o.drafts, false);
            defer gpa.free(g.tokens);
            const sha = try sha12(gpa, g.tokens);
            std.debug.print("bench rep {d} prompt {d} tokens: prefill {d:.4}s {d:.1} tok/s; reply {d} tokens sha {s} first {d}", .{ rep, p.len, g.prefill_s, @as(f64, @floatFromInt(p.len)) / g.prefill_s, g.tokens.len, sha, g.tokens[0] });
            if (g.tokens.len > 1) std.debug.print(" decode {d:.2} ms/token", .{g.decode_s * 1e3 / @as(f64, @floatFromInt(g.tokens.len - 1))});
            std.debug.print("\n", .{});
        }
    }
    return 0;
}

/// bench-many's prompts: 32 a kind, stream i takes prompt i % 32 (user turns, the chat template around them).
const bench_code = [_][]const u8{
    "Write a Python function that parses an ISO 8601 date string without using datetime, with tests.",
    "Write a C function that reverses a singly linked list in place, with comments.",
    "Implement an LRU cache class in Python with get and put in O(1).",
    "Write a Rust function that merges two sorted vectors into one sorted vector.",
    "Write a Go HTTP handler that returns the current server time as JSON.",
    "Implement binary search over a sorted array of integers in Java, iteratively and recursively.",
    "Write a Python script that counts word frequencies in a text file and prints the top 20.",
    "Write a TypeScript function that debounces another function, with a usage example.",
    "Implement a min-heap in C++ with push, pop and top, without using the standard library heap.",
    "Write a SQL query that finds the second highest salary in each department, and explain it.",
    "Write a Bash script that backs up a directory into a dated tar.gz and keeps the last seven.",
    "Implement Dijkstra's shortest path algorithm in Python using heapq.",
    "Write a Python class for a bank account with deposit, withdraw and a transaction history.",
    "Write a JavaScript function that deep-clones an object, handling arrays, dates and cycles.",
    "Implement a trie in Python with insert, search and starts_with.",
    "Write a C program that reads integers from stdin and prints their mean and standard deviation.",
    "Write a Python generator that yields the prime numbers forever using an incremental sieve.",
    "Implement a thread-safe bounded queue in Java with put and take.",
    "Write a Rust program that reads a CSV file and sums the third column.",
    "Write a Python function that validates a Sudoku board, with unit tests.",
    "Implement quicksort in Haskell and explain its complexity.",
    "Write a Python decorator that retries a function up to three times with exponential backoff.",
    "Write a Go program that runs ten workers over a channel of jobs and collects their results.",
    "Implement matrix multiplication in NumPy and in pure Python, and compare them.",
    "Write a Python function that converts Roman numerals to integers and back.",
    "Write a C++ class for a 2D vector with operator overloading for +, - and dot product.",
    "Implement a simple tokenizer for arithmetic expressions in Python and evaluate them.",
    "Write a Kotlin data class for a user and a function that sorts users by age then name.",
    "Write a Python async function that fetches several URLs concurrently with aiohttp.",
    "Implement a union-find structure with path compression and union by rank in C.",
    "Write a Python function that flattens an arbitrarily nested list, with tests.",
    "Write a shell one-liner and an explanation that finds the ten largest files under a directory.",
};
const bench_prose = [_][]const u8{
    "Write a short story of about 300 words about a lighthouse keeper who finds a message in a bottle.",
    "Describe a walk through an autumn forest at dusk in about 300 words.",
    "Write a 300-word letter from a sailor to his sister about his first voyage.",
    "Write a short story about a robot who learns to bake bread.",
    "Describe a busy night market in a coastal city, in vivid detail.",
    "Write a fable about a fox and a crow who must share a winter shelter.",
    "Write a diary entry from a girl moving to a new town on the first day of school.",
    "Describe a thunderstorm rolling over a wheat field, from the farmer's porch.",
    "Write a short story about two strangers stuck in a train station overnight.",
    "Write a 300-word scene where an old musician plays one last concert.",
    "Describe the inside of a library that has been closed for fifty years.",
    "Write a short myth that explains why the moon changes shape.",
    "Write a story about a child who finds a door at the back of a wardrobe that leads to a garden.",
    "Describe a morning in a mountain village as the first snow falls.",
    "Write a letter from an astronaut to her younger self.",
    "Write a short story about a cat who runs a small bookshop.",
    "Describe a desert crossing by camel caravan at night under the stars.",
    "Write a scene where a detective realises the culprit is her oldest friend.",
    "Write a story about a gardener who grows a plant that sings.",
    "Describe a harbour at dawn as the fishing boats return.",
    "Write a short story about a clockmaker who can pause time for one minute a day.",
    "Write a monologue of a tree that has stood in a city square for three hundred years.",
    "Describe a family dinner during a power cut, lit only by candles.",
    "Write a short story about a lost dog finding its way home across the city.",
    "Write a poem in prose about the sea in winter.",
    "Describe the last day of summer camp from a counsellor's point of view.",
    "Write a story about a baker who receives a mysterious order every Tuesday.",
    "Write a short story set on a spaceship where the gardener is the captain.",
    "Describe a rainy afternoon in a small cafe in Paris.",
    "Write a story about an old map that changes every time it is unfolded.",
    "Write a scene where two rivals must climb a mountain together.",
    "Describe the sounds of a city waking up, street by street.",
};
const bench_chat = [_][]const u8{
    "What are three practical tips for sleeping better? Answer in a few sentences each.",
    "Explain why the sky is blue to a ten-year-old.",
    "What should I pack for a weekend hiking trip in the mountains?",
    "How do I make a good first impression at a job interview?",
    "What is the difference between a virus and a bacterium?",
    "Can you suggest a simple weekly exercise plan for a beginner?",
    "How does compound interest work? Give a small example.",
    "What are some good ways to learn a new language as an adult?",
    "Why do leaves change colour in autumn?",
    "How can I reduce food waste at home?",
    "What is a black hole, in simple terms?",
    "Give me a recipe for a quick vegetarian dinner.",
    "How do vaccines train the immune system?",
    "What are the pros and cons of working from home?",
    "How do I start a small vegetable garden on a balcony?",
    "Why is the ocean salty?",
    "What are some tips for staying focused while studying?",
    "How do airplanes stay in the air?",
    "What should I consider when adopting a dog?",
    "Explain what inflation is and why it happens.",
    "How can I improve my public speaking?",
    "What causes the seasons on Earth?",
    "How do I write a good cover letter?",
    "What is the difference between weather and climate?",
    "How can I save money on groceries?",
    "Why do we dream?",
    "What are good habits for keeping my computer secure?",
    "How do rainbows form?",
    "What is the best way to prepare for a marathon?",
    "How does the stock market work, briefly?",
    "What are some easy houseplants for beginners?",
    "How do bees make honey?",
};

const bench_structured = [_][]const u8{
    "Return a JSON array of 10 European capital cities with fields name, country, population and founded_year.",
    "Convert this list into a JSON object keyed by id: 1 apple red, 2 banana yellow, 3 grape purple, 4 lime green.",
    "Write a JSON schema for a blog post with title, author, tags, published date and a list of comments.",
    "Produce a YAML configuration for a web service with a database, a cache, three workers and logging settings.",
    "Return a markdown table of the planets of the solar system with columns name, diameter in km, moons and orbit days.",
    "Give a JSON list of 8 recipes, each with name, cuisine, minutes, difficulty and a list of ingredients.",
    "Write a CSV with a header and 15 rows of fictional employees: id, name, department, salary, start_date.",
    "Return a JSON object describing a library: name, address, opening hours per weekday and 5 sections with shelves.",
    "Extract the entities from this sentence as JSON: Maria flew from Lisbon to Tokyo on 3 May 2024 with Air Canada.",
    "Produce an OpenAPI 3 YAML snippet for a REST endpoint that creates, reads, updates and deletes a todo item.",
    "Return a JSON array of 12 months with fields name, days, season in the northern hemisphere and a holiday.",
    "Write a markdown table comparing five programming languages by typing, compilation, memory management and year.",
    "Give a JSON object for a product catalog with 6 products, each with sku, name, price, stock and categories.",
    "Return the periodic table's first 20 elements as JSON with symbol, name, atomic number and atomic mass.",
    "Write a GitHub Actions workflow YAML that tests a Python package on three Python versions and two operating systems.",
    "Produce a JSON timetable for a school week: five days, six periods a day, subject and teacher for each.",
    "Return a JSON array of 10 books with title, author, year, genre and a one-sentence summary.",
    "Write a Kubernetes deployment and service YAML for a container serving on port 8080 with three replicas.",
    "Give a markdown table of 10 countries with capital, currency, official language and continent.",
    "Return a JSON object that models a chess game in progress: players, clock times, moves so far and board state.",
    "Produce a JSON list of 15 HTTP status codes with code, name, category and a short description.",
    "Write an XML document describing a music album with artist, year, label and twelve tracks with durations.",
    "Return a JSON object for a travel itinerary: 5 days, each with city, hotel, activities and meals.",
    "Give a CSV of 20 weather observations with date, city, high, low, precipitation and conditions.",
    "Return a JSON array describing 8 dog breeds with name, size, life_expectancy, temperament and origin.",
    "Produce a TOML configuration for a command line tool with profiles, output formats, retries and logging.",
    "Write a JSON object for an invoice with seller, buyer, ten line items, taxes and totals.",
    "Return a markdown table of the first 15 US presidents with term start, term end and party.",
    "Give a JSON representation of a directory tree with folders src, tests and docs and the files in each.",
    "Return a JSON array of 10 airports with IATA code, city, country, latitude and longitude.",
    "Produce a JSON object for a football league table of 10 teams with played, won, drawn, lost, goals and points.",
    "Write a JSON list of 12 workouts with name, muscle group, sets, reps and rest seconds.",
};

/// The owner's sparkDash decode bench (src/shared/llmPrompts.js; greedy, thinking off): prose, structured, code.
const dash_prose = "Write a detailed step-by-step explanation of how a hash map works, including collision handling, resizing, and time complexity. Be thorough.";
const dash_structured = "Count from 1 to 200. Output only the numbers, separated by spaces. No other text.";
const dash_code_tail = "Output only Python source. No comments, no docstrings, no markdown fences. Then add tests and the helpers this needs. Keep writing code.";
const dash_code = [_][]const u8{
    "binary_search\ndef binary_search(nums, target) -> int: index of target in a sorted list, or -1.\n" ++ dash_code_tail,
    "merge_sort\ndef merge_sort(nums) -> list: stable sort of a list of ints, returning a new list.\n" ++ dash_code_tail,
    "lru_cache\nclass LRUCache: get(key) and put(key, value) with a fixed capacity, evicting the least recently used.\n" ++ dash_code_tail,
    "token_bucket\nclass TokenBucket: allow(n) consumes n tokens refilled at a fixed rate, else returns False.\n" ++ dash_code_tail,
    "ring_buffer\nclass RingBuffer: push and pop over a fixed-capacity array, raising on overflow and underflow.\n" ++ dash_code_tail,
    "dijkstra\ndef dijkstra(graph, src) -> dict: shortest path weights from src on a non-negative weighted graph.\n" ++ dash_code_tail,
    "edit_distance\ndef edit_distance(a, b) -> int: Levenshtein distance between two strings.\n" ++ dash_code_tail,
    "semver_cmp\ndef semver_cmp(a, b) -> int: compare dotted numeric versions, negative if a < b.\n" ++ dash_code_tail,
    "url_parse\ndef url_parse(url) -> dict: scheme, host, port, path, and query pairs. No extra libraries.\n" ++ dash_code_tail,
    "json_pointer\ndef json_pointer(doc, pointer) -> object: follow an RFC 6901 pointer, or None if missing.\n" ++ dash_code_tail,
    "glob_match\ndef glob_match(pattern, text) -> bool: * and ? wildcards, no character classes.\n" ++ dash_code_tail,
    "csv_parse\ndef csv_parse(text) -> list: rows of fields, honoring double-quoted commas and escaped quotes.\n" ++ dash_code_tail,
    "rle\ndef rle_encode(s) -> str and rle_decode(s) -> str: run-length encoding of single-byte runs.\n" ++ dash_code_tail,
    "top_k\ndef top_k(nums, k) -> list: the k largest ints, unordered, using a bounded heap.\n" ++ dash_code_tail,
    "interval_merge\ndef merge_intervals(spans) -> list: merge overlapping [start, end] pairs.\n" ++ dash_code_tail,
    "topo_sort\ndef topo_sort(nodes, edges) -> list: a valid order, or None if the graph has a cycle.\n" ++ dash_code_tail,
    "bloom_filter\nclass BloomFilter: add(item) and might_contain(item) with two hash functions over a bit array.\n" ++ dash_code_tail,
    "moving_average\nclass MovingAverage: next(x) returns the mean of the last window values.\n" ++ dash_code_tail,
    "retry_backoff\ndef backoff_delays(attempts, base_ms, cap_ms) -> list: exponential delays clipped at the cap.\n" ++ dash_code_tail,
    "base64_encode\ndef b64_encode(data) -> str and b64_decode(text) -> bytes: standard base64, no libraries.\n" ++ dash_code_tail,
    "expr_eval\ndef eval_expr(text) -> int: evaluate non-negative ints with + - * / and parentheses.\n" ++ dash_code_tail,
    "histogram_percentile\ndef percentile(samples, p) -> float: nearest-rank percentile of a list of numbers.\n" ++ dash_code_tail,
    "redact_secrets\ndef redact(text) -> str: replace AWS-looking keys and password= values with ***. Keep the rest.\n" ++ dash_code_tail,
    "chunk_text\ndef chunk_text(text, size) -> list: split into chunks of at most size chars, breaking on spaces when possible.\n" ++ dash_code_tail,
    "route_match\ndef route_match(pattern, path) -> dict or None: /users/:id style params.\n" ++ dash_code_tail,
    "crc32\ndef crc32(data) -> int: IEEE CRC-32 of a bytes object.\n" ++ dash_code_tail,
    "fixed_window\nclass FixedWindow: allow() is True up to limit events per window_s, else False.\n" ++ dash_code_tail,
    "diff_lines\ndef diff_lines(a, b) -> list: line diff as equal/delete/insert ops using a simple LCS.\n" ++ dash_code_tail,
    "infix_postfix\ndef infix_to_postfix(tokens) -> list: shunting-yard for + - * / and parentheses.\n" ++ dash_code_tail,
    "consistent_hash\nclass ConsistentHash: add_node, remove_node, and get_node(key) on a ring of virtual nodes.\n" ++ dash_code_tail,
    "utf8_decode\ndef utf8_decode(data) -> str: decode UTF-8 bytes, replacing invalid sequences with U+FFFD.\n" ++ dash_code_tail,
    "dependency_closure\ndef closure(root, deps) -> list: packages reachable from root, each name once, in visit order.\n" ++ dash_code_tail,
};

/// bench-many: each --kind and --rows stream count decoded in shared rounds --reps times: tok/s and each round's parts.
pub fn benchMany(gpa: Allocator, io: std.Io, e: *Engine, o: Options) !u8 {
    const path = try std.fs.path.join(gpa, &.{ o.model, "tokenizer.json" });
    defer gpa.free(path);
    var tok = try core.tokenizer.loadTokenizer(io, gpa, path);
    defer tok.deinit();
    // <|im_start|>user\n TEXT <|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n (the captured prompts' frame)
    const head = [_]u32{ 248045, 846, 198 };
    const tail = [_]u32{ 248046, 198, 248045, 74455, 198, 248068, 271, 248069, 271 };
    var counts: std.ArrayList(usize) = .empty;
    defer counts.deinit(gpa);
    if (o.rows.len > 0) {
        var ct = std.mem.tokenizeScalar(u8, o.rows, ',');
        while (ct.next()) |c| try counts.append(gpa, try std.fmt.parseInt(usize, c, 10));
    } else try counts.append(gpa, o.streams);
    var kinds = std.mem.tokenizeScalar(u8, o.kind, ',');
    while (kinds.next()) |kind| {
        // dash-*: the owner's sparkDash prompts (built a stream count below), greedy
        const dash = std.mem.startsWith(u8, kind, "dash-");
        const texts: []const []const u8 = if (std.mem.eql(u8, kind, "code")) &bench_code else if (std.mem.eql(u8, kind, "prose")) &bench_prose else if (std.mem.eql(u8, kind, "chat")) &bench_chat else if (std.mem.eql(u8, kind, "structured")) &bench_structured else if (std.mem.eql(u8, kind, "dash-code")) &dash_code else if (std.mem.eql(u8, kind, "dash-prose")) &.{dash_prose} else if (std.mem.eql(u8, kind, "dash-structured")) &.{dash_structured} else return error.UnknownKind;
        // code and structured greedy; prose and chat sampled (dbench's rules); --temperature T samples every kind
        const greedy = o.sampling.temperature <= 0 and (dash or std.mem.eql(u8, kind, "code") or std.mem.eql(u8, kind, "structured"));
        const prompts = try gpa.alloc([]u32, texts.len);
        defer {
            for (prompts) |x| gpa.free(x);
            gpa.free(prompts);
        }
        for (texts, prompts) |t, *x| {
            const body = try tok.encode(gpa, t);
            defer gpa.free(body);
            x.* = try std.mem.concat(gpa, u32, &.{ &head, body, &tail });
        }
        for (counts.items) |N| for (0..o.reps) |rep| {
            const reqs = try gpa.alloc(Engine.Request, N);
            defer {
                for (reqs) |*r| r.tokens.deinit(gpa);
                gpa.free(reqs);
            }
            // sparkDash's one-prompt kinds: the prompt itself alone, else " (stream i/N)" on each stream's copy
            const own = try gpa.alloc([]u32, N);
            defer {
                for (own) |x| if (x.len > 0) gpa.free(x);
                gpa.free(own);
            }
            for (own, 0..) |*x, i| {
                x.* = &.{};
                if (!dash or texts.len > 1 or N == 1) continue;
                var tb: [512]u8 = undefined;
                const t = try std.fmt.bufPrint(&tb, "{s} (stream {d}/{d})", .{ texts[0], i + 1, N });
                const bodyt = try tok.encode(gpa, t);
                defer gpa.free(bodyt);
                x.* = try std.mem.concat(gpa, u32, &.{ &head, bodyt, &tail });
            }
            for (reqs, 0..) |*r, i| {
                var smp: lanes.Sampling = .{ .seed = 1000 + i, .temperature = 1.0, .top_k = 20, .top_p = 0.95, .min_p = 0.0 };
                if (o.sampling.temperature > 0) {
                    smp = o.sampling;
                    smp.seed = o.sampling.seed + i;
                }
                r.* = .{ .prompt = if (own[i].len > 0) own[i] else prompts[i % prompts.len], .max_tokens = o.max_tokens, .sampling = if (greedy) null else smp, .drafts = o.drafts, .confidence = if (o.served) flashnext.engine.confidenceSetting() else flashnext.engine.default_confidence, .stop_eos = o.stop_eos };
            }
            const t0 = now(io);
            try e.generateMany(gpa, reqs);
            const secs = since(io, t0);
            const tm = e.times;
            var produced: usize = 0;
            var rounds: usize = 0;
            for (reqs) |r| {
                produced += r.tokens.items.len;
                rounds += r.rounds;
            }
            const nr: f64 = @floatFromInt(@max(1, tm.rounds));
            const dec = tm.verify + tm.sample + tm.draft;
            std.debug.print("RESULT bench-many {s} {s} x{d} rep {d}: {d} tokens, prefill {d:.2}s, {d:.1} tokens/s decoding ({d:.1} with the prefills) over {d} rounds, {d:.2} tokens a stream-round; a round: verify {d:.1} ms (enqueued in {d:.1}), draws+commits {d:.1} ms, drafting {d:.1} ms\n", .{ kind, if (greedy) "greedy" else "sampled", N, rep, produced, tm.prefill, @as(f64, @floatFromInt(produced - N)) / dec, @as(f64, @floatFromInt(produced)) / secs, tm.rounds, @as(f64, @floatFromInt(produced - N)) / @as(f64, @floatFromInt(@max(1, rounds))), tm.verify * 1e3 / nr, tm.verify_host * 1e3 / nr, tm.sample * 1e3 / nr, tm.draft * 1e3 / nr });
        };
    }
    return 0;
}
