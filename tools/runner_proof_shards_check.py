"""Reconcile the binary's catalog with every singleton execution; never infer coverage from counts."""
import argparse
import json
from pathlib import Path

PREFIX = 'PROOF_SHARD '


def require(condition, message):
    if not condition:
        raise ValueError(message)


def records(text):
    rows = []
    for line in text.splitlines():
        if line.startswith(PREFIX):
            row = json.loads(line[len(PREFIX):])
            require(isinstance(row, dict), 'invalid record')
            rows.append(row)
    return rows


def catalog_names(rows):
    require(len(rows) >= 2, 'missing catalog')
    begin, end = rows[0], rows[-1]
    require(begin.get('event') == 'catalog_begin' and end.get('event') == 'catalog_end',
            'incomplete catalog')
    total = begin.get('total')
    require(type(total) is int and 0 < total <= 256 and begin.get('limit') == 256,
            'invalid catalog population')
    require(end.get('total') == total and len(rows) == total + 2, 'catalog size mismatch')
    names = []
    for index, row in enumerate(rows[1:-1]):
        require(row.get('event') == 'catalog' and row.get('index') == index,
                'catalog index mismatch')
        name = row.get('name')
        require(isinstance(name, str) and bool(name), 'invalid catalog name')
        names.append(name)
    require(len(set(names)) == total, 'duplicate full test name')
    return names


def verify(catalog, runs):
    names = catalog_names(catalog)
    if any(row.get('event') in ('catalog_begin', 'catalog', 'catalog_end') for row in runs):
        require(runs[:len(catalog)] == catalog, 'different or misplaced execution catalog')
        runs = runs[len(catalog):]
    started, completed, empty = set(), set(), set()
    events = []
    for row in runs:
        event, index = row.get('event'), row.get('index')
        require(type(index) is int and 0 <= index < 256, 'invalid original index')
        require(row.get('total') == len(names), 'mixed binary populations')
        if event == 'empty':
            require(index >= len(names) and index not in empty, 'duplicate or occupied empty shard')
            empty.add(index)
            events.append((index, event))
            continue
        require(event in ('start', 'result') and index < len(names), 'invalid execution event')
        require(row.get('name') == names[index], 'original name/index mismatch')
        if event == 'start':
            require(index not in started and index not in completed, 'duplicate start')
            started.add(index)
        else:
            require(index in started and index not in completed, 'missing start or duplicate result')
            require(row.get('status') == 'pass', 'test did not pass')
            require(type(row.get('log_err_count')) is int and row['log_err_count'] == 0,
                    'logged error or missing error count')
            require(type(row.get('leak_count')) is int and row['leak_count'] == 0,
                    'allocation leak or missing leak count')
            completed.add(index)
        events.append((index, event))
    require(started == completed == set(range(len(names))), 'missing test execution')
    require(empty == set(range(len(names), 256)), 'missing empty shard receipt')
    expected = [(index, event) for index in range(len(names)) for event in ('start', 'result')]
    expected += [(index, 'empty') for index in range(len(names), 256)]
    require(events == expected, 'execution was not the serial singleton schedule')
    return {'tests': len(names), 'passed': len(completed), 'empty_shards': len(empty),
            'full_name_union': names}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('catalog', type=Path)
    parser.add_argument('runs', type=Path)
    parser.add_argument('--original-catalog', type=Path, required=True)
    args = parser.parse_args()
    catalog = records(args.catalog.read_text())
    original_text = args.original_catalog.read_text()
    original_rows = records(original_text)
    if original_rows:
        original = catalog_names(original_rows)
    else:
        lines = original_text.splitlines()
        markers = [i for i, line in enumerate(lines) if line.startswith('GATE_INVENTORY_COMPLETE ')]
        require(len(markers) == 1, 'missing independent original inventory')
        end = markers[0]
        total = int(lines[end].split()[-1])
        require(0 < total <= end, 'invalid original inventory count')
        original = lines[end - total:end]
        require(len(set(original)) == total, 'duplicate original name')
    names = catalog_names(catalog)
    require(len(original) == len(names) and set(original) == set(names),
            'original compiler population changed')
    result = verify(catalog, records(args.runs.read_text()))
    print(json.dumps(result, ensure_ascii=True, indent=2))


if __name__ == '__main__':
    main()
