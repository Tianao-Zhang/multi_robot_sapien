"""Pluggable coding-agent backends: Claude Code CLI, Codex CLI, OpenAI-compatible API, Callable.

Every backend operates inside a single isolated workspace directory so Generator G^(n),
Evaluator E^(n), and Specifier never share conversation context or hidden state.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import textwrap
import time
from typing import Any, Callable, Dict, List, Optional, Protocol


@dataclass
class AgentResult:
    ok: bool
    backend: str
    stdout: str = ""
    stderr: str = ""
    elapsed_s: float = 0.0
    artifacts: Dict[str, Any] = field(default_factory=dict)
    error: str = ""


class CodingBackend(Protocol):
    name: str

    def run(
        self,
        system_prompt: str,
        user_prompt: str,
        workspace: Path,
        timeout_s: int = 420,
    ) -> AgentResult: ...


_FILE_BLOCK_RE = re.compile(
    r"(?:^|\n)###\s*FILE:\s*([^\n]+?)\s*\n```[a-zA-Z0-9_-]*\n(.*?)\n```",
    re.DOTALL,
)


def apply_file_blocks(text: str, workspace: Path) -> List[str]:
    """Parse `### FILE: rel/path` fenced code blocks and write them safely inside `workspace`."""
    written: List[str] = []
    ws_root = workspace.resolve()
    for match in _FILE_BLOCK_RE.finditer(text):
        rel = match.group(1).strip().lstrip("/")
        content = match.group(2)
        target = (ws_root / rel).resolve()
        if not target.is_relative_to(ws_root) or target == ws_root:
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content + ("\n" if not content.endswith("\n") else ""), encoding="utf-8")
        written.append(rel)
    return written


class ClaudeCodeBackend:
    """Invoke `claude -p` non-interactively inside an isolated workspace directory."""

    name = "claude"

    def __init__(
        self,
        binary: str = "claude",
        model: Optional[str] = None,
        allowed_tools: str = "Read,Edit,Write",
        effort: str = "low",
    ) -> None:
        self.binary = shutil.which(binary) or binary
        self.model = model
        self.allowed_tools = allowed_tools
        self.effort = effort

    def run(
        self,
        system_prompt: str,
        user_prompt: str,
        workspace: Path,
        timeout_s: int = 420,
    ) -> AgentResult:
        workspace.mkdir(parents=True, exist_ok=True)
        sys_file = workspace / "_system_prompt.md"
        sys_file.write_text(system_prompt, encoding="utf-8")
        cmd = [
            self.binary,
            "-p",
            user_prompt,
            "--system-prompt-file",
            str(sys_file),
            "--output-format",
            "text",
            "--permission-mode",
            "bypassPermissions",
            "--tools",
            self.allowed_tools,
            "--no-session-persistence",
            "--add-dir",
            str(workspace.resolve()),
        ]
        if self.effort:
            cmd.extend(["--effort", self.effort])
        if self.model:
            cmd.extend(["--model", self.model])
        t0 = time.perf_counter()
        try:
            proc = subprocess.run(
                cmd,
                cwd=str(workspace.resolve()),
                stdin=subprocess.DEVNULL,
                capture_output=True,
                text=True,
                timeout=timeout_s,
                env=os.environ.copy(),
            )
            dt = round(time.perf_counter() - t0, 3)
            ok = proc.returncode == 0
            return AgentResult(
                ok=ok,
                backend=self.name,
                stdout=proc.stdout,
                stderr=proc.stderr,
                elapsed_s=dt,
                error="" if ok else f"claude exit {proc.returncode}: {proc.stderr[-300:]}",
            )
        except FileNotFoundError:
            return AgentResult(ok=False, backend=self.name, error=f"claude binary not found: {self.binary}")
        except subprocess.TimeoutExpired:
            return AgentResult(
                ok=False,
                backend=self.name,
                elapsed_s=round(time.perf_counter() - t0, 3),
                error=f"claude timed out after {timeout_s}s",
            )


class CodexBackend:
    """Invoke `codex exec` non-interactively with workspace-write sandbox."""

    name = "codex"

    def __init__(self, binary: str = "codex", model: Optional[str] = None) -> None:
        self.binary = shutil.which(binary) or binary
        self.model = model

    def run(
        self,
        system_prompt: str,
        user_prompt: str,
        workspace: Path,
        timeout_s: int = 420,
    ) -> AgentResult:
        workspace.mkdir(parents=True, exist_ok=True)
        (workspace / "outbox").mkdir(exist_ok=True)
        prompt = textwrap.dedent(
            f"""
            SYSTEM INSTRUCTIONS:
            {system_prompt}

            USER TASK:
            {user_prompt}
            """
        ).strip()
        last_msg = workspace / "outbox" / "last_message.txt"
        cmd = [
            self.binary,
            "exec",
            "--skip-git-repo-check",
            "-C",
            str(workspace.resolve()),
            "-s",
            "workspace-write",
            "--add-dir",
            str(workspace.resolve()),
            "-o",
            str(last_msg),
        ]
        if self.model:
            cmd.extend(["-m", self.model])
        cmd.append(prompt)
        t0 = time.perf_counter()
        try:
            proc = subprocess.run(
                cmd,
                cwd=str(workspace.resolve()),
                stdin=subprocess.DEVNULL,
                capture_output=True,
                text=True,
                timeout=timeout_s,
                env=os.environ.copy(),
            )
            dt = round(time.perf_counter() - t0, 3)
            ok = proc.returncode == 0
            out_txt = last_msg.read_text(encoding="utf-8") if last_msg.exists() else proc.stdout
            return AgentResult(
                ok=ok,
                backend=self.name,
                stdout=out_txt,
                stderr=proc.stderr,
                elapsed_s=dt,
                error="" if ok else f"codex exit {proc.returncode}: {proc.stderr[-300:]}",
            )
        except FileNotFoundError:
            return AgentResult(ok=False, backend=self.name, error=f"codex binary not found: {self.binary}")
        except subprocess.TimeoutExpired:
            return AgentResult(
                ok=False,
                backend=self.name,
                elapsed_s=round(time.perf_counter() - t0, 3),
                error=f"codex timed out after {timeout_s}s",
            )


class OpenAICompatBackend:
    """Chat Completions or opt-in OpenAI Responses backend.

    Instructs the model to return updated files as `### FILE: relative/path` fenced
    blocks and applies them safely inside `workspace`.
    """

    name = "openai"

    def __init__(
        self,
        model: str = "gpt-4o",
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
    ) -> None:
        self.model = model
        self.base_url = (base_url or os.environ.get("OPENAI_BASE_URL") or "https://api.openai.com/v1").rstrip("/")
        self.api_key = api_key or os.environ.get("MAWAM_API_KEY") or os.environ.get("OPENAI_API_KEY")
        self.api = os.environ.get("MAWAM_OPENAI_API", "chat/completions")
        if self.api not in ("chat/completions", "responses"):
            raise ValueError("MAWAM_OPENAI_API must be 'chat/completions' or 'responses'")

    def run(
        self,
        system_prompt: str,
        user_prompt: str,
        workspace: Path,
        timeout_s: int = 420,
    ) -> AgentResult:
        if not self.api_key:
            return AgentResult(ok=False, backend=self.name, error="OPENAI_API_KEY / MAWAM_API_KEY is not set")
        workspace.mkdir(parents=True, exist_ok=True)
        # Bundle workspace context files into the prompt for pure-API models
        ctx_parts: List[str] = []
        for p in sorted(workspace.rglob("*")):
            if p.is_file() and not p.name.startswith("_") and p.suffix in (".py", ".yaml", ".json", ".md"):
                rel = p.relative_to(workspace)
                content = p.read_text(encoding="utf-8")
                if len(content) <= 18000:
                    ctx_parts.append(f"### EXISTING FILE: {rel}\n```\n{content}\n```")

        full_user = (
            user_prompt
            + "\n\nWhen creating or editing files, output each file using:\n"
            + "### FILE: relative/path.py\n```python\n<full file content>\n```\n\n"
            + "\n\n".join(ctx_parts)
        )
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": full_user},
            ],
            "temperature": 0.2,
        }
        if self.api == "responses":
            payload = {
                "model": self.model,
                "instructions": system_prompt,
                "input": full_user,
                "reasoning": {"effort": "medium"},
                "max_output_tokens": 16000,
                "store": False,
            }
        t0 = time.perf_counter()
        try:
            import urllib.request

            req = urllib.request.Request(
                f"{self.base_url}/{self.api}",
                data=json.dumps(payload).encode("utf-8"),
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.api_key}",
                },
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=timeout_s) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            if self.api == "responses":
                if data.get("status") != "completed":
                    raise RuntimeError(f"OpenAI response status: {data.get('status')}")
                text = "\n".join(
                    content["text"] for item in data["output"] if item.get("type") == "message"
                    for content in item["content"] if content.get("type") == "output_text"
                )
                if not text.strip():
                    raise RuntimeError("OpenAI response contained no text")
            else:
                text = data["choices"][0]["message"]["content"]
            written = apply_file_blocks(text, workspace)
            return AgentResult(
                ok=True,
                backend=self.name,
                stdout=text,
                elapsed_s=round(time.perf_counter() - t0, 3),
                artifacts={"written_files": written, "model": data.get("model"), "usage": data.get("usage")},
            )
        except Exception as exc:  # noqa: BLE001
            return AgentResult(
                ok=False,
                backend=self.name,
                elapsed_s=round(time.perf_counter() - t0, 3),
                error=str(exc),
            )


class CallableBackend:
    """Programmatic backend for deterministic unit tests and custom integrations."""

    name = "callable"

    def __init__(self, handler: Callable[[str, str, Path], AgentResult]) -> None:
        self._handler = handler

    def run(
        self,
        system_prompt: str,
        user_prompt: str,
        workspace: Path,
        timeout_s: int = 420,
    ) -> AgentResult:
        workspace.mkdir(parents=True, exist_ok=True)
        t0 = time.perf_counter()
        res = self._handler(system_prompt, user_prompt, workspace)
        if res.elapsed_s == 0.0:
            res.elapsed_s = round(time.perf_counter() - t0, 3)
        return res


def make_backend(name: str, **kwargs: Any) -> CodingBackend:
    key = name.strip().lower()
    if key in ("claude", "claude-code"):
        return ClaudeCodeBackend(
            **{k: v for k, v in kwargs.items() if k in ("binary", "model", "allowed_tools") and v is not None}
        )
    if key == "codex":
        return CodexBackend(**{k: v for k, v in kwargs.items() if k in ("binary", "model") and v is not None})
    if key in ("openai", "api", "deepseek", "litellm"):
        return OpenAICompatBackend(
            **{k: v for k, v in kwargs.items() if k in ("model", "base_url", "api_key") and v is not None}
        )
    raise ValueError(f"Unknown backend {name!r}; expected claude | codex | openai")
