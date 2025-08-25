# repo\_tree\_token\_counts

## What

Prints a tree of a local directory with **lines of code (LOC)** and **token counts** per file, plus directory roll-ups.

* No cloning, no network—works on your local filesystem.
* Respects **.gitignore** by default (uses `git check-ignore` when available; falls back to `pathspec`).

## Why

Quickly see where the weight of a codebase lives—by files and folders—without IDEs or language-specific tooling. Good for sizing work, scoping refactors, and budgeting token costs.

## How

### Requirements

* Python **3.9+**
* Optional:

  * `tiktoken` for accurate token counts (`pip install tiktoken`)
  * `pathspec` to honor `.gitignore` outside a git repo (`pip install pathspec`)

### Usage

```bash
# analyze current directory
python repo_tree_metrics.py

# analyze a specific path
python repo_tree_metrics.py /path/to/repo

# faster: skip tokenization
python repo_tree_metrics.py --no-tokens

# include hidden files and raise size cap to 2MB
python repo_tree_metrics.py --hidden --max-bytes 2000000

# ignore extra dirs/files in addition to defaults
python repo_tree_metrics.py --ignore build --ignore .cache

# do NOT honor .gitignore
python repo_tree_metrics.py --no-gitignore

# save output to a file (for now)
python repo_tree_metrics.py > tree.txt
```

### Options (common)

* `path` — directory to scan (default: current working directory)
* `--model cl100k_base` — tokenizer/encoding for `tiktoken` (default shown)
* `--no-tokens` — disable token counting
* `--max-bytes 1000000` — skip files larger than this (default: 1MB)
* `--include-binary` — attempt to read binaries (not recommended)
* `--ignore NAME` — repeatable; add ignores (on top of `.gitignore` + built-ins)
* `--hidden` — include dotfiles/dirs
* `--follow-symlinks` — follow symlinks (may loop without care)
* `--no-gitignore` — don’t apply `.gitignore`

### Tokenization notes

* Uses `tiktoken` if installed. Otherwise, falls back to a simple estimator or use `--no-tokens`.
* Prefer `cl100k_base` (GPT-3.5/4 era) or `o200k_base` (newer long-context models).

### Troubleshooting

* If Python complains about *unterminated string literal* or *null bytes*, ensure the script is saved as **UTF-8** and that byte literals are plain ASCII quotes: `b"\n"` and `b"\x00"`.

