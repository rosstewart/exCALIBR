#!/usr/bin/env python3

import csv
import glob
import json
import sys

# Only these columns are emitted / mapped from JSON
CSV_COLUMNS = [
    "dataset",
    "range_-8",
    "range_-7",
    "range_-6",
    "range_-5",
    "range_-4",
    "range_-3",
    "range_-2",
    "range_-1",
    "range_1",
    "range_2",
    "range_3",
    "range_4",
    "range_5",
    "range_6",
    "range_7",
    "range_8",
    "prior",
    "relax",
    "n_c",
    "benign_method",
    "clinvar_2018",
    "scoreset_flipped",
]


def fmt_num(x):
    if x == float("inf"):
        return "Infinity"
    if x == float("-inf"):
        return "-Infinity"
    return str(x)


def fmt_ranges(ranges):
    """
    [[a,b], [c,d]] -> "[a,b];[c,d]"
    """
    if not ranges:
        return ""

    return ";".join(
        f"{fmt_num(lo)} {fmt_num(hi)}"
        for lo, hi in ranges
    )


def build_row(obj):
    row = {}

    row["dataset"] = obj.get("dataset", "")
    row["prior"] = obj.get("prior", "")
    row["relax"] = obj.get("relax", "")
    row["n_c"] = obj.get("n_c", "")
    row["benign_method"] = obj.get("benign_method", "")
    row["clinvar_2018"] = obj.get("clinvar_2018", "")
    row["scoreset_flipped"] = obj.get("scoreset_flipped", "")

    point_ranges = obj.get("point_ranges", {})

    for i in range(-8, 9):
        if i == 0:
            continue

        row[f"range_{i}"] = fmt_ranges(
            point_ranges.get(str(i), [])
        )

    return row


def load_objects(path):
    with open(path) as f:
        data = json.load(f)

    if isinstance(data, list):
        return data

    return [data]


def main():
    if len(sys.argv) < 2:
        print(
            f"usage: {sys.argv[0]} 'glob_pattern' > out.csv",
            file=sys.stderr,
        )
        sys.exit(1)

    files = sorted(sys.argv[1:])

    writer = csv.DictWriter(sys.stdout, fieldnames=CSV_COLUMNS)
    writer.writeheader()

    for path in files:
        for obj in load_objects(path):
            writer.writerow(build_row(obj))


if __name__ == "__main__":
    main()
