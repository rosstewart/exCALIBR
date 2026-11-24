#!/usr/bin/env python3
"""Release version helper for the GitHub workflows.

A release PR into `main` is titled with its version, e.g. "v1.2.0" or
"v1.2.0: consequence-aware fitting". The git tag is the only place the
version lives (pyproject.toml reads it through setuptools-scm).

    release_version.py check "<PR title>"   # title starts with a NEW vX.Y.Z,
                                            # higher than every existing tag
    release_version.py parse "<PR title>"   # just extract vX.Y.Z

Prints the version tag; with --github-output it is also written to
$GITHUB_OUTPUT as `version=vX.Y.Z`. Exits non-zero with a message on failure.
Run from inside the repository (needs the tags: `git fetch --tags`).
"""
import argparse
import os
import re
import subprocess
import sys

TITLE_RE = re.compile(r"^\s*v(\d+)\.(\d+)\.(\d+)(?=$|[\s:,;])")
TAG_RE = re.compile(r"^v(\d+)\.(\d+)\.(\d+)$")


def parse_title(title):
    m = TITLE_RE.match(title)
    if not m:
        sys.exit(
            f"PR title {title!r} must start with the release version, "
            "e.g. 'v1.2.0' or 'v1.2.0: short summary'."
        )
    return tuple(int(x) for x in m.groups())


def existing_versions():
    out = subprocess.run(
        ["git", "tag", "-l", "v*"], check=True, capture_output=True, text=True
    ).stdout
    return sorted(
        tuple(int(x) for x in m.groups())
        for m in (TAG_RE.match(t.strip()) for t in out.splitlines())
        if m
    )


def fmt(v):
    return "v{}.{}.{}".format(*v)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("mode", choices=["check", "parse"])
    ap.add_argument("title")
    ap.add_argument("--github-output", action="store_true",
                    help="also write version=vX.Y.Z to $GITHUB_OUTPUT")
    args = ap.parse_args()

    version = parse_title(args.title)
    if args.mode == "check":
        tags = existing_versions()
        if version in tags:
            sys.exit(f"Tag {fmt(version)} already exists; pick a new version.")
        if tags and version < tags[-1]:
            sys.exit(
                f"{fmt(version)} is lower than the latest release {fmt(tags[-1])}. "
                "Hotfixes for older versions are tagged by hand, not merged into main."
            )

    print(fmt(version))
    if args.github_output:
        with open(os.environ["GITHUB_OUTPUT"], "a") as fh:
            fh.write(f"version={fmt(version)}\n")


if __name__ == "__main__":
    main()
