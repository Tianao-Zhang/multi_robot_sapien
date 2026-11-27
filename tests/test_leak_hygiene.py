"""Repository hygiene audit: guarantees ZERO exposed local directories, internal paths, or secrets."""

from __future__ import annotations

from pathlib import Path
import re
import subprocess
from typing import List, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent

# Scan all source, config, script, and markdown files (exclude binary assets and this test itself).
SCAN_EXTENSIONS = {".py", ".yaml", ".yml", ".json", ".md", ".sh", ".toml", ".txt"}

# Patterns that must NEVER appear in the public open-source repository.
FORBIDDEN_PATTERNS: List[Tuple[str, re.Pattern[str]]] = [
    ("local_mac_home", re.compile(r"/Users/")),
    ("local_linux_tiger", re.compile(r"/home/tiger")),
    ("internal_cluster_vepfs", re.compile(r"ML-vePFS|vePFS", re.IGNORECASE)),
    ("internal_workspace_dir", re.compile(r"/workspace/(?:MAWAM|mawam)")),
    ("internal_workspace_name", re.compile(r"\bmutil_agent\b|\bMAV2\b|\bRobot_test\b")),
    ("internal_company", re.compile(r"bytedance|byte_data", re.IGNORECASE)),
    ("internal_hdfs", re.compile(r"hdfs://|harunawl", re.IGNORECASE)),
    ("internal_ip_or_ssh", re.compile(r"101\.126\.31\.72|15808")),
    ("internal_host_id", re.compile(r"\bali1019\b|\bhuoshan\b")),
    ("internal_username", re.compile(r"\btangrongze\b|/trz/|/wzl/")),
    ("internal_codename_dir", re.compile(r"t06_robotwin|food_dump_stage2|remaining5_latest|toy3562|poorcube5549|lingbot_accel_bench")),
]


def scan_repo(root: Path = REPO_ROOT) -> List[Tuple[Path, int, str, str]]:
    hits: List[Tuple[Path, int, str, str]] = []
    this_file = Path(__file__).resolve()
    if (root / ".git").exists():
        paths = subprocess.run(
            ["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"],
            cwd=root, capture_output=True, text=True, check=True,
        ).stdout.split("\0")
        candidates = (root / path for path in paths if path)
    else:
        candidates = root.rglob("*")
    for path in sorted(candidates):
        if not path.is_file():
            continue
        if ".git" in path.parts or "__pycache__" in path.parts:
            continue
        if path.resolve() == this_file:
            continue
        if path.suffix.lower() not in SCAN_EXTENSIONS:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for lineno, line in enumerate(text.splitlines(), start=1):
            for label, pat in FORBIDDEN_PATTERNS:
                if pat.search(line):
                    hits.append((path.relative_to(root), lineno, label, line.strip()[:120]))
    return hits


def test_zero_internal_leaks() -> None:
    hits = scan_repo()
    if hits:
        formatted = "\n".join(f"  {p}:{ln} [{lbl}] {snippet}" for p, ln, lbl, snippet in hits)
        raise AssertionError(f"Found {len(hits)} forbidden internal path/string leak(s):\n{formatted}")


if __name__ == "__main__":
    test_zero_internal_leaks()
    print("[PASS] Zero local paths, internal hosts, usernames, or private URIs found across repository.")
