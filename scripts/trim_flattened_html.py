#!/usr/bin/env python3
import argparse
import fnmatch
import os
import posixpath
import re
import sys
from typing import Iterable, List, Optional, Set, Tuple

DOC_BLOCK_RE = re.compile(r"<document\b[^>]*>.*?</document>", re.DOTALL)
SOURCE_RE = re.compile(r"<source>(.*?)</source>", re.DOTALL | re.IGNORECASE)
WRAP_RE = re.compile(r"<documents\b[^>]*>(.*)</documents>", re.DOTALL | re.IGNORECASE)

def read_all(path: str) -> str:
    if path == "-":
        return sys.stdin.read()
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def write_all(path: str, text: str) -> None:
    if path == "-":
        sys.stdout.write(text)
        return
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)

def to_glob(user_token: str) -> str:
    # If the user supplied an actual glob, use it as-is.
    if any(ch in user_token for ch in "*?[]"):
        return normalize_glob(user_token)
    # Otherwise, treat it as a folder/token and match anywhere in the path.
    # e.g., "examples" -> "*/examples/*"
    return f"*/{user_token}/*"

def normalize_path(p: str) -> str:
    """Normalize paths to forward-slash, strip whitespace, collapse ./ and //."""
    s = p.strip().replace("\\", "/")
    # Remove BOM if present (e.g., from copied file lists)
    s = s.lstrip("\ufeff")
    # Normalize with posixpath to keep forward slashes
    s = posixpath.normpath(s)
    # Remove leading "./"
    if s.startswith("./"):
        s = s[2:]
    return s

def normalize_glob(g: str) -> str:
    """Normalize glob strings to forward slashes."""
    return g.replace("\\", "/")

def matches_any_glob(path: str, patterns: List[str]) -> bool:
    return any(fnmatch.fnmatchcase(path, pat) for pat in patterns)

def extract_documents_area(html: str) -> Tuple[str, str, str]:
    """Return (prefix, inner, suffix). If <documents> wrapper exists, preserve it."""
    m = WRAP_RE.search(html)
    if not m:
        return ("", html, "")
    start, end = m.span(1)
    return (html[:start], html[start:end], html[end:])

def load_list_file(list_path: str) -> Set[str]:
    """Load a newline-delimited list of paths; ignore blanks and lines starting with '#'."""
    items: Set[str] = set()
    with open(list_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip("\n\r")
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            items.add(normalize_path(line))
    return items

def compute_default_output_path(input_path: str, explicit_output: Optional[str]) -> str:
    if explicit_output:
        return explicit_output
    if input_path != "-" and input_path:
        base = os.path.basename(input_path)
        out = f"trimmed_{base}"
        return os.path.join(os.path.dirname(input_path), out)
    # If reading from stdin and no output specified, write to stdout
    return "-"

def main():
    ap = argparse.ArgumentParser(
        description=(
            "Filter flattened <document> blocks by <source> path. "
            "You can provide globs/tokens and/or a file containing an explicit list of paths to include."
        )
    )
    ap.add_argument(
        "patterns",
        nargs="*",
        help=(
            "Glob(s) or bare token(s). Example: examples -> '*/examples/*', "
            "or provide a full glob like 'clients/**/examples/*.py'."
        ),
    )
    ap.add_argument("-l", "--list", dest="list_file", help="Path to a file containing newline-delimited source paths to include.")
    ap.add_argument("-i", "--input", default="-", help="Input HTML file (default: stdin).")
    ap.add_argument("-o", "--output", default=None, help="Output HTML file. "
                     "If not provided and input is a file, defaults to 'trimmed_{original_filename}'.")
    ap.add_argument("--invert", action="store_true",
                    help="Exclude matches instead of including them (applies to both globs and list file).")
    args = ap.parse_args()

    if not args.patterns and not args.list_file:
        ap.error("Provide at least one glob/token pattern or --list FILE with paths to include.")

    # Normalize patterns (convert bare tokens to */token/*)
    globs: List[str] = [to_glob(p) for p in args.patterns] if args.patterns else []
    globs = [normalize_glob(g) for g in globs]

    # Load explicit paths list (exact matches)
    list_paths: Set[str] = load_list_file(args.list_file) if args.list_file else set()

    html = read_all(args.input)
    prefix, inner, suffix = extract_documents_area(html)

    matched_blocks: List[str] = []
    kept, total = 0, 0

    for m in DOC_BLOCK_RE.finditer(inner):
        block = m.group(0)
        total += 1
        sm = SOURCE_RE.search(block)
        source_path_raw = sm.group(1).strip() if sm else ""
        source_path = normalize_path(source_path_raw)

        is_glob_match = matches_any_glob(source_path, globs) if globs else False
        is_list_match = (source_path in list_paths) if list_paths else False

        is_match = is_glob_match or is_list_match
        keep = (not args.invert and is_match) or (args.invert and not is_match)

        if keep:
            matched_blocks.append(block)
            kept += 1

    # If the original had a <documents> wrapper, keep it. Otherwise add one.
    if WRAP_RE.search(html):
        new_inner = "\n".join(matched_blocks)
        out_html = f"{prefix}{new_inner}{suffix}"
    else:
        out_html = "<documents>\n" + "\n".join(matched_blocks) + "\n</documents>\n"

    out_path = compute_default_output_path(args.input, args.output)
    write_all(out_path, out_html)

    # Progress info to stderr so it doesn't pollute stdout when piping
    print(f"Selected {kept} of {total} document blocks. Wrote: {out_path}", file=sys.stderr)

if __name__ == "__main__":
    main()
