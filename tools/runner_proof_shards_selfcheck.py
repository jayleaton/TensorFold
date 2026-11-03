"""Disposable protocol checks for a compiled fixture; parent runs this, never the source-only author."""
import argparse
import json
import os
from pathlib import Path
import struct
import subprocess

from runner_proof_shards_check import catalog_names, records, require, verify

QUERY = struct.pack('<II', 4, 0)
EXIT = struct.pack('<II', 0, 0)


def run_message(index=0):
    return struct.pack('<III', 5, 4, index)


def frames(raw):
    result = []
    offset = 0
    while offset < len(raw):
        require(len(raw) - offset >= 8, 'truncated server header')
        tag, size = struct.unpack_from('<II', raw, offset)
        offset += 8
        require(len(raw) - offset >= size, 'truncated server body')
        result.append((tag, raw[offset:offset + size]))
        offset += size
    return result


def metadata(body):
    string_size, count = struct.unpack_from('<II', body)
    require(len(body) == 8 + count * 8 + string_size, 'metadata size mismatch')
    offsets = struct.unpack_from('<' + 'I' * count, body, 8)
    expected_panic = struct.unpack_from('<' + 'I' * count, body, 8 + count * 4)
    require(all(value == 0 for value in expected_panic), 'unexpected panic expectation')
    strings = body[8 + count * 8:]
    require(strings and strings[0] == 0, 'missing null string')
    names = []
    for offset in offsets:
        require(0 < offset < len(strings), 'metadata name offset')
        end = strings.index(0, offset)
        names.append(strings[offset:end].decode())
    return names


def invoke(binary, args, payload=b'', mode=None):
    environ = dict(os.environ)
    environ.pop('PROOF_SHARD_FIXTURE_MODE', None)
    if mode:
        environ['PROOF_SHARD_FIXTURE_MODE'] = mode
    return subprocess.run([str(binary), *args], input=payload, capture_output=True,
                          timeout=30, env=environ)


def singleton(binary, index, expected_name=None, mode=None):
    args = ['--listen=-', '--seed=0x4d576fd2', '--cache-dir=.zig-cache',
            f'--proof-shard-index={index}']
    payload = QUERY + (run_message() if expected_name is not None else b'') + EXIT
    process = invoke(binary, args, payload, mode)
    require(process.returncode == 0, process.stderr.decode(errors='replace'))
    reply = frames(process.stdout)
    require(reply[0] == (0, b'0.17.0'), 'missing version handshake')
    expected_tags = [0, 3, 5, 4] if expected_name is not None else [0, 3]
    require([tag for tag, _ in reply] == expected_tags, 'unexpected server messages')
    require(metadata(reply[1][1]) == ([] if expected_name is None else [expected_name]),
            'singleton metadata does not match original name')
    if expected_name is None:
        return records(process.stderr.decode()), None
    require(reply[2][1] == b'', 'test_started has a body')
    require(len(reply[3][1]) == 12, 'test result ABI mismatch')
    local_index, flags = struct.unpack('<IQ', reply[3][1])
    require(local_index == 0, 'result index must stay local')
    return records(process.stderr.decode()), {
        'status': flags & 3, 'fuzz': (flags >> 2) & 1,
        'log_err_count': (flags >> 3) & ((1 << 30) - 1), 'leak_count': flags >> 33,
    }


def rejected(call, description):
    try:
        call()
    except ValueError:
        return
    raise ValueError('negative coverage control was accepted: ' + description)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('binary', type=Path)
    args = parser.parse_args()
    binary = args.binary.resolve()
    catalog_run = invoke(binary, ['--proof-catalog'])
    require(catalog_run.returncode == 0 and catalog_run.stdout == b'', 'catalog failed')
    catalog = records(catalog_run.stderr.decode())
    names = catalog_names(catalog)
    require(any('compute runner proof:' in name for name in names), 'substring population lost')
    require(any('fixture JSON "quoted" \\ slash\t tab\n newline' in name for name in names),
            'JSON escaping fixture lost')
    require(any('runner proof:' not in name for name in names), 'anonymous fixture lost')
    require(len(names) < 256, 'fixture must exercise empty metadata too')
    runs = []
    for index in range(256):
        rows, flags = singleton(binary, index, names[index] if index < len(names) else None)
        if flags is not None:
            require(flags == {'status': 0, 'fuzz': 0, 'log_err_count': 0, 'leak_count': 0},
                    'baseline protocol flags')
        runs.extend(rows)
    verified = verify(catalog, runs)
    rejected(lambda: verify(catalog, runs[1:]), 'missing start')
    rejected(lambda: verify(catalog, runs[:-1]), 'missing empty shard')
    rejected(lambda: verify(catalog, runs + [runs[1]]), 'duplicate result')
    rejected(lambda: verify(catalog, catalog), 'catalog cannot establish execution')
    wrong = [dict(row) for row in runs]
    wrong[1]['index'] = (wrong[1]['index'] + 1) % len(names)
    rejected(lambda: verify(catalog, wrong), 'wrong original index')
    for mode, suffix, field, value in [
        ('skip', 'fixture skip', 'status', 2),
        ('fail', 'fixture failure', 'status', 1),
        ('log', 'fixture logged error', 'log_err_count', 1),
        ('leak', 'fixture allocation leak', 'leak_count', 1),
    ]:
        index = next(i for i, name in enumerate(names) if name.endswith(suffix))
        rows, flags = singleton(binary, index, names[index], mode)
        require(flags[field] == value and flags['fuzz'] == 0, 'lost ' + mode + ' flag')
        result = next(row for row in rows if row['event'] == 'result')
        replacement = [result if row['event'] == 'result' and row['index'] == index else row
                       for row in runs]
        rejected(lambda: verify(catalog, replacement), 'failed/skipped qualification: ' + mode)
    index = next(i for i, name in enumerate(names) if name.endswith('fixture panic'))
    crashed = invoke(binary, ['--listen=-', f'--proof-shard-index={index}'],
                     QUERY + run_message() + EXIT, 'panic')
    require(crashed.returncode != 0, 'panic became success')
    require(not any(row['event'] == 'result' for row in records(crashed.stderr.decode())),
            'panic fabricated a result')
    cleanup = invoke(binary, ['--listen=-', f'--proof-shard-index={index}'], EXIT)
    require(cleanup.returncode == 0 and frames(cleanup.stdout) == [(0, b'0.17.0')],
            'SDK crash cleanup handshake failed')
    for argv, payload in [
        (['--listen=-'], QUERY + EXIT),
        (['--listen=-', '--proof-shard-index=256'], QUERY + EXIT),
        (['--listen=-', '--proof-shard-index=-1'], QUERY + EXIT),
        (['--listen=-', '--proof-shard-index=0', '--proof-shard-index=0'], QUERY + EXIT),
        (['--listen=-', '--proof-shard-index=0'], QUERY + run_message(1) + EXIT),
        (['--listen=-', '--proof-shard-index=0'], QUERY + run_message() + run_message() + EXIT),
        (['--listen=-', '--proof-shard-index=0'], QUERY + EXIT),
        (['--listen=-', '--proof-shard-index=0'], run_message() + EXIT),
        (['--listen=-', '--proof-shard-index=255'], QUERY + run_message() + EXIT),
    ]:
        bad = invoke(binary, argv, payload)
        require(bad.returncode != 0, 'bad invocation succeeded: ' + repr((argv, payload)))
    print(json.dumps({'fixture_protocol_checks': 'pass', 'tests': verified['tests'],
                      'singleton_processes': 256, 'coverage_controls': 5,
                      'result_controls': 4, 'panic_controls': 1, 'argument_controls': 9}))


if __name__ == '__main__':
    main()
