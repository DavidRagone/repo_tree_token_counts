#!/usr/bin/env python3
"""
repo_tree_metrics.py — Walk a local directory, print a tree with per-file
line counts and token counts, plus directory rollups. No cloning; purely local.

Usage examples:
  python repo_tree_metrics.py                # analyze current working dir
  python repo_tree_metrics.py /path/to/repo  # analyze specific dir
  python repo_tree_metrics.py . --model cl100k_base --max-bytes 2000000
  python repo_tree_metrics.py . --no-tokens  # skip tokenization (faster)
  python repo_tree_metrics.py . --ignore node_modules --ignore .git --hidden
  python repo_tree_metrics.py . --no-gitignore  # do NOT honor .gitignore

Dependencies:
  - Python 3.9+
  - Optional: `pip install tiktoken` for accurate token counts.
    If not available, a simple fallback estimator is used, or you can pass --no-tokens.
  - Optional: `pip install pathspec` if you want .gitignore support outside of a git repo.

Notes:
  - By default, skips obvious binaries and files larger than --max-bytes (1 MB).
  - Honors `.gitignore` automatically when present. It prefers `git check-ignore`
    (exact git semantics). If the directory is not a git repo, it falls back to
    top-level `.gitignore` via `pathspec` when available.
  - You can redirect stdout to a file for the "save to file" use-case:
      python repo_tree_metrics.py > tree.txt
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, List

# ---------------- Tokenization helpers ----------------

def _load_tiktoken(model: str):
    try:
        import tiktoken  # type: ignore
    except Exception:
        return None
    try:
        enc = tiktoken.encoding_for_model(model)
    except Exception:
        # allow base encoding names like 'cl100k_base' or 'o200k_base'
        try:
            enc = tiktoken.get_encoding(model)
        except Exception:
            return None
    return enc


def count_tokens(text: str, model: str | None) -> int:
    """Count tokens using tiktoken when available; otherwise, a cheap estimator.
    If model is None, returns 0.
    """
    if model is None:
        return 0
    enc = _load_tiktoken(model)
    if enc is not None:
        try:
            return len(enc.encode(text))
        except Exception:
            pass
    # Fallback: rough heuristic. Many BPEs average ~3–4 chars/token in code.
    # We'll use max of whitespace-split and char/4 to avoid severe undercount.
    approx = max(len(text.split()), max(1, len(text) // 4))
    return approx


# ---------------- Data structures ----------------

@dataclass
class Node:
    name: str
    is_dir: bool
    loc: int = 0
    tokens: int = 0
    children: List["Node"] = field(default_factory=list)


# ---------------- .gitignore support ----------------

def build_gitignore_matcher(root: Path, enabled: bool) -> Callable[[Path], bool]:
    """Return a function is_ignored(path) -> bool using git semantics if possible.

    Preference order:
      1) If `enabled` and this is a git repo and `git` is available, use
         `git check-ignore` for exact behavior (respects nested .gitignore,
         core.excludesfile, etc.).
      2) Else if `enabled` and `pathspec` is installed, load top-level
         `.gitignore` (and `.git/info/exclude` if present) as a best-effort
         fallback.
      3) Else, no-op (never ignore).
    """
    if not enabled:
        return lambda p: False

    # 1) Exact semantics via git
    if (root / ".git").exists() and shutil.which("git"):
        cache: dict[str, bool] = {}
        def is_ignored_git(p: Path) -> bool:
            rel = os.path.relpath(p, start=root)
            if rel in cache:
                return cache[rel]
            try:
                # 0 => ignored; 1 => not ignored; 128 => error
                rc = subprocess.run(
                    ["git", "-C", str(root), "check-ignore", "-q", "--", rel],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                ).returncode
                ignored = (rc == 0)
            except Exception:
                ignored = False
            cache[rel] = ignored
            return ignored
        return is_ignored_git

    # 2) Best-effort using pathspec on top-level patterns
    try:
        from pathspec.patterns.gitwildmatch import GitWildMatchPattern
        from pathspec import PathSpec
        patterns: List[str] = []
        gi = root / ".gitignore"
        if gi.exists():
            patterns.extend(gi.read_text().splitlines())
        info_exclude = root / ".git" / "info" / "exclude"
        if info_exclude.exists():
            patterns.extend(info_exclude.read_text().splitlines())
        if patterns:
            spec = PathSpec.from_lines(GitWildMatchPattern, patterns)
            def is_ignored_ps(p: Path) -> bool:
                rel = p.relative_to(root).as_posix()
                return spec.match_file(rel)
            return is_ignored_ps
    except Exception:
        pass

    # 3) No-op
    return lambda p: False


# ---------------- Filesystem scanning ----------------

_DEFAULT_IGNORES = {
    ".git",
    "node_modules",
    "dist",
    "build",
    "target",
    "__pycache__",
    ".venv",
    "venv",
    ".idea",
    ".vscode",
}

_BINARY_EXTS = {
    "png","jpg","jpeg","gif","webp","ico","bmp","tiff","heic",
    "pdf","zip","gz","tar","rar","7z","xz","zst",
    "mp3","wav","ogg","flac","aac","m4a",
    "mp4","mov","avi","mkv","webm",
    "woff","woff2","ttf","otf",
    "class","o","so","dll","dylib","a","jar",
}


def is_hidden(path: Path) -> bool:
    return path.name.startswith(".") and path.name not in {".gitignore", ".gitattributes"}


def count_file_lines(path: Path) -> int:
    """Count lines by scanning for newlines with streaming; +1 if final byte isn't 
.
    Accurate for any bytes; returns 0 for empty files.
    """
    nl = 0
    last_byte: int | None = None
    with path.open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            nl += chunk.count(b"\n")
            last_byte = chunk[-1]
    if last_byte is None:
        return 0
    return nl if last_byte == 0x0A else nl + 1


def read_counts_for_file(path: Path, max_bytes: int, include_binary: bool, model: str | None) -> tuple[int, int]:
    """Return (loc, tokens) for a file, respecting size/binary filters."""
    # Quick extension-based binary filter
    ext = path.suffix.lower().lstrip(".")
    if not include_binary and ext in _BINARY_EXTS:
        return (0, 0)

    try:
        size = path.stat().st_size
    except FileNotFoundError:
        return (0, 0)
    if not include_binary and size > max_bytes:
        return (0, 0)

    # Stream the file once to detect binary-ish content and count lines
    found_null = False
    nl = 0
    last_byte: int | None = None
    try:
        with path.open("rb") as f:
            while True:
                chunk = f.read(1024 * 1024)
                if not chunk:
                    break
                if not include_binary and b"\x00" in chunk:
                    found_null = True
                nl += chunk.count(b"\n")
                last_byte = chunk[-1]
    except Exception:
        return (0, 0)

    if not include_binary and found_null:
        return (0, 0)

    loc = 0 if last_byte is None else (nl if last_byte == 0x0A else nl + 1)

    # Tokenization pass (text decode with ignore errors)
    tok = 0
    if model is not None:
        try:
            with path.open("r", encoding="utf-8", errors="ignore") as tf:
                text = tf.read()
            tok = count_tokens(text, model)
        except Exception:
            tok = 0

    return (loc, tok)


def scan_dir(root: Path, ignores: set[str], include_hidden: bool, follow_symlinks: bool,
             max_bytes: int, include_binary: bool, model: str | None,
             is_ignored: Callable[[Path], bool]) -> Node:
    def sort_key(p: Path):
        # dirs first, then files; names case-insensitive
        return (0 if p.is_dir() else 1, p.name.lower())

    node = Node(root.name, True)

    try:
        entries = sorted(root.iterdir(), key=sort_key)
    except PermissionError:
        return node

    for entry in entries:
        name = entry.name
        if name in ignores:
            continue
        if not include_hidden and is_hidden(entry):
            continue
        # Honor .gitignore (files AND directories)
        try:
            if is_ignored(entry):
                continue
        except Exception:
            # if matcher misbehaves, fail open
            pass

        try:
            if entry.is_symlink() and not follow_symlinks:
                continue
            if entry.is_dir():
                child = scan_dir(entry.resolve() if follow_symlinks else entry,
                                 ignores, include_hidden, follow_symlinks,
                                 max_bytes, include_binary, model, is_ignored)
                node.children.append(child)
            elif entry.is_file():
                loc, tok = read_counts_for_file(entry, max_bytes, include_binary, model)
                child = Node(name, False, loc=loc, tokens=tok)
                node.children.append(child)
        except Exception:
            # Skip problematic entries silently
            continue

    # Roll up directory totals
    node.loc = sum(c.loc for c in node.children)
    node.tokens = sum(c.tokens for c in node.children)
    return node


# ---------------- Rendering ----------------

def print_tree(node: Node, prefix: str = "", is_root: bool = True) -> None:
    if is_root:
        print(f"{node.name}/ ({node.loc} lines, {node.tokens} tokens)")
    # Sort: directories first, then files
    children = sorted(node.children, key=lambda n: (not n.is_dir, n.name.lower()))
    for i, child in enumerate(children):
        last = i == len(children) - 1
        connector = "└──" if last else "├──"
        if child.is_dir:
            print(f"{prefix}{connector} {child.name}/ ({child.loc}, {child.tokens})")
            extension = "    " if last else "│   "
            print_tree(child, prefix + extension, False)
        else:
            print(f"{prefix}{connector} {child.name} ({child.loc}, {child.tokens})")


# ---------------- CLI ----------------

def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Print a repo tree with line & token counts.")
    ap.add_argument("path", nargs="?", default=str(Path.cwd()), help="Directory to scan (default: CWD)")
    ap.add_argument("--model", default="cl100k_base", help="tiktoken model/encoding (e.g. cl100k_base, o200k_base). Use --no-tokens to disable.")
    ap.add_argument("--no-tokens", action="store_true", help="Disable tokenization for speed.")
    ap.add_argument("--max-bytes", type=int, default=1_000_000, help="Skip files larger than this (default: 1MB)")
    ap.add_argument("--include-binary", action="store_true", help="Attempt to read all files including binaries (not recommended)")
    ap.add_argument("--ignore", action="append", default=[], help="Directory or file name to ignore (repeatable)")
    ap.add_argument("--hidden", action="store_true", help="Include hidden files/dirs (names starting with .)")
    ap.add_argument("--follow-symlinks", action="store_true", help="Follow symlinks (may cause cycles)")
    ap.add_argument("--no-gitignore", action="store_true", help="Do NOT honor .gitignore (default is to honor it if present)")

    args = ap.parse_args(argv)

    root = Path(args.path).resolve()
    if not root.exists() or not root.is_dir():
        print(f"Error: path is not a directory: {root}", file=sys.stderr)
        return 2

    ignores = set(_DEFAULT_IGNORES)
    ignores.update(args.ignore)

    model = None if args.no_tokens else args.model

    is_ignored = build_gitignore_matcher(root, enabled=not args.no_gitignore)

    tree = scan_dir(
        root,
        ignores=ignores,
        include_hidden=args.hidden,
        follow_symlinks=args.follow_symlinks,
        max_bytes=args.max_bytes,
        include_binary=args.include_binary,
        model=model,
        is_ignored=is_ignored,
    )

    print_tree(tree)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
