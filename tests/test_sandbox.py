"""Tests for the sandboxed executor (harness/sandbox.py) and its wiring into
run_shell / run_python (harness/builtins/shell.py).

Layering under test, per sandbox.py's own docstring:
  - the hard, unconditional destructive-command floor (always on, regardless
    of policy.enabled / exec_policy)
  - opt-in network-pattern blocking (network="deny")
  - run_python's real in-process filesystem write-guard (builtins.open AND
    the pathlib.Path equivalents)
  - the symlink-escape closing realpath-based containment check
  - the memory watchdog (psutil-based; this is what actually enforces a
    ceiling on Windows, where POSIX rlimits aren't available)
  - config.py -> sandbox.load_policy() wiring (sandbox_enabled/sandbox_mem_mb/
    sandbox_network actually reaching SandboxPolicy)

None of this touches or re-tests the approval gate (Agent._default_approve /
exec_policy) — that's covered elsewhere and is explicitly out of scope here;
the sandbox is a layer underneath it, never a substitute.
"""

from __future__ import annotations

import subprocess
import sys

import pytest

from pleiades import config
from pleiades.harness import sandbox as sb
from pleiades.harness.builtins import shell


# --------------------------------------------------------------------------- #
# Hard, unconditional destructive-command floor
# --------------------------------------------------------------------------- #
_DESTRUCTIVE = [
    "rm -rf /",
    "rm -rf /*",
    ":(){ :|:& };:",
    "mkfs.ext4 /dev/sda1",
    "dd if=/dev/zero of=/dev/sda",
    "diskpart",
    "format C:",
    "rd /s /q C:\\",
    "Remove-Item -Recurse -Force C:\\",
    "reg delete HKLM\\Software\\Foo",
]


@pytest.mark.parametrize("cmd", _DESTRUCTIVE)
def test_hard_deny_blocks_destructive_patterns_even_when_disabled(cmd):
    # enabled=False is the *configurable* sandbox; the floor must still fire.
    policy = sb.SandboxPolicy(enabled=False)
    with pytest.raises(sb.SandboxViolation):
        sb.check_command_safety(cmd, policy)


def test_benign_commands_pass_safety_check():
    policy = sb.SandboxPolicy()
    sb.check_command_safety("echo hello world", policy)
    sb.check_command_safety("ls -la /tmp", policy)
    sb.check_command_safety("git status", policy)
    sb.check_command_safety("python script.py --flag value", policy)


# --------------------------------------------------------------------------- #
# Regex-anchor bypass regression (security review finding #1): the original
# patterns ended on a trailing whitespace-or-end-of-string anchor, which is
# defeated by completely ordinary shell syntax -- quoting, chaining,
# redirection -- around the dangerous target. These must all still be caught
# by the negative-lookahead-based patterns.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("cmd", [
    'rm -rf "/"',
    "rm -rf '/'",
    "rm -rf /;",
    "rm -rf / && echo done",
    "rm -rf / # cleanup",
    "rd /s /q C:\\;",
    "rd /s /q C:\\ && echo done",
    "Remove-Item -Recurse -Force C:\\;",
    "Remove-Item -Recurse -Force C:\\ ; echo done",
])
def test_hard_deny_blocks_quoted_and_chained_destructive_variants(cmd):
    policy = sb.SandboxPolicy(enabled=False)
    with pytest.raises(sb.SandboxViolation):
        sb.check_command_safety(cmd, policy)


@pytest.mark.parametrize("cmd", [
    "rm -rf /home/user/tmp",
    "rm -rf /home/user/tmp/build",
    "rd /s /q C:\\Users\\me\\AppData\\Temp",
    "Remove-Item -Recurse -Force C:\\Users\\me\\scratch",
])
def test_hard_deny_allows_legitimate_subpaths(cmd):
    # Regression guard for the negative-lookahead rewrite: it must keep
    # rejecting "this is actually a longer path", not just re-broaden back
    # to matching any rm -rf/rd/Remove-Item invocation at all.
    policy = sb.SandboxPolicy(enabled=False)
    sb.check_command_safety(cmd, policy)  # must NOT raise


# --------------------------------------------------------------------------- #
# Opt-in network egress control
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("cmd", [
    "curl http://example.com",
    "wget http://example.com/file",
    "git clone https://github.com/x/y",
    "pip install requests",
    "ssh user@host",
])
def test_network_deny_blocks_known_network_patterns(cmd):
    policy = sb.SandboxPolicy(network="deny")
    with pytest.raises(sb.SandboxViolation):
        sb.check_command_safety(cmd, policy)


def test_network_allow_is_the_default_and_does_not_block():
    policy = sb.SandboxPolicy()  # default network="allow"
    assert policy.network == "allow"
    sb.check_command_safety("curl http://example.com", policy)


# --------------------------------------------------------------------------- #
# run_python in-process write guard (the one truly enforceable fs scope)
# --------------------------------------------------------------------------- #
def test_wrap_python_source_is_noop_when_sandbox_disabled():
    policy = sb.SandboxPolicy(enabled=False)
    code = "print('hi')"
    assert sb.wrap_python_source(code, policy) == code


def test_wrap_python_source_injects_guard_when_enabled():
    policy = sb.SandboxPolicy(enabled=True)
    code = "print('hi')"
    wrapped = sb.wrap_python_source(code, policy)
    assert "sandbox guard" in wrapped
    assert code in wrapped


def test_run_python_blocks_write_outside_workspace_root(tmp_path, monkeypatch):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    outside = tmp_path / "outside.txt"

    policy = sb.SandboxPolicy(enabled=True, fs_root=str(workspace), network="allow")
    monkeypatch.setattr(sb, "load_policy", lambda cfg=None: policy)

    code = f"with open(r'{outside}', 'w') as f:\n    f.write('pwned')\n"
    out = shell.run_python(code, timeout=20)

    assert not outside.exists()
    assert "permission" in out.lower() or "denied" in out.lower()


def test_run_python_allows_write_inside_workspace_root(tmp_path, monkeypatch):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    target = workspace / "ok.txt"

    policy = sb.SandboxPolicy(enabled=True, fs_root=str(workspace), network="allow")
    monkeypatch.setattr(sb, "load_policy", lambda cfg=None: policy)

    code = f"with open(r'{target}', 'w') as f:\n    f.write('fine')\n"
    out = shell.run_python(code, timeout=20)

    assert out.startswith("[exit 0]")
    assert target.exists()
    assert target.read_text() == "fine"


# --------------------------------------------------------------------------- #
# pathlib write-guard bypass regression (security review finding #2): the
# original guard only patched builtins.open/os.remove/os.unlink/os.rmdir/
# shutil.rmtree. Path.write_text/write_bytes/open/unlink/rmdir route through
# pathlib's own accessor layer and were NOT covered -- confirmed to bypass
# the old guard entirely. These must now be blocked the same as plain open().
# --------------------------------------------------------------------------- #
def test_run_python_blocks_pathlib_write_text_outside_workspace_root(tmp_path, monkeypatch):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    outside = tmp_path / "outside.txt"

    policy = sb.SandboxPolicy(enabled=True, fs_root=str(workspace), network="allow")
    monkeypatch.setattr(sb, "load_policy", lambda cfg=None: policy)

    code = f"from pathlib import Path\nPath(r'{outside}').write_text('pwned')\n"
    out = shell.run_python(code, timeout=20)

    assert not outside.exists()
    assert "permission" in out.lower() or "denied" in out.lower()


def test_run_python_blocks_pathlib_open_outside_workspace_root(tmp_path, monkeypatch):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    outside = tmp_path / "outside.txt"

    policy = sb.SandboxPolicy(enabled=True, fs_root=str(workspace), network="allow")
    monkeypatch.setattr(sb, "load_policy", lambda cfg=None: policy)

    code = (
        f"from pathlib import Path\n"
        f"with Path(r'{outside}').open('w') as f:\n    f.write('pwned')\n"
    )
    out = shell.run_python(code, timeout=20)

    assert not outside.exists()
    assert "permission" in out.lower() or "denied" in out.lower()


def test_run_python_blocks_pathlib_unlink_outside_workspace_root(tmp_path, monkeypatch):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_text("do not delete me")

    policy = sb.SandboxPolicy(enabled=True, fs_root=str(workspace), network="allow")
    monkeypatch.setattr(sb, "load_policy", lambda cfg=None: policy)

    code = f"from pathlib import Path\nPath(r'{outside}').unlink()\n"
    out = shell.run_python(code, timeout=20)

    assert outside.exists()
    assert outside.read_text() == "do not delete me"
    assert "permission" in out.lower() or "denied" in out.lower()


def test_run_python_allows_pathlib_write_text_inside_workspace_root(tmp_path, monkeypatch):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    target = workspace / "ok.txt"

    policy = sb.SandboxPolicy(enabled=True, fs_root=str(workspace), network="allow")
    monkeypatch.setattr(sb, "load_policy", lambda cfg=None: policy)

    code = f"from pathlib import Path\nPath(r'{target}').write_text('fine')\n"
    out = shell.run_python(code, timeout=20)

    assert out.startswith("[exit 0]")
    assert target.exists()
    assert target.read_text() == "fine"


# --------------------------------------------------------------------------- #
# Symlink-escape regression (security review finding #3): os.path.abspath()
# does not resolve symlinks, so a symlink planted inside the workspace root
# but pointing outside it used to pass the containment check while the
# actual write/delete landed outside the sandbox. The guard now resolves
# symlinks (realpath) before the containment check.
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(sys.platform == "win32",
                     reason="symlink creation needs elevation/dev-mode on Windows CI runners")
def test_run_python_blocks_write_through_symlink_escaping_workspace_root(tmp_path, monkeypatch):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    secret_dir = tmp_path / "secret_outside"
    secret_dir.mkdir()
    secret = secret_dir / "secret.txt"
    secret.write_text("should not be touched")

    link = workspace / "innocent_link.txt"
    link.symlink_to(secret)

    policy = sb.SandboxPolicy(enabled=True, fs_root=str(workspace), network="allow")
    monkeypatch.setattr(sb, "load_policy", lambda cfg=None: policy)

    code = f"with open(r'{link}', 'w') as f:\n    f.write('pwned')\n"
    out = shell.run_python(code, timeout=20)

    assert secret.read_text() == "should not be touched"
    assert "permission" in out.lower() or "denied" in out.lower()


# --------------------------------------------------------------------------- #
# run_shell / run_python wiring: hard floor + network deny actually reachable
# through the public tool functions (not just the unit-level checker)
# --------------------------------------------------------------------------- #
def test_run_shell_blocks_destructive_command_without_spawning_it():
    out = shell.run_shell("rm -rf /", timeout=5)
    assert out.startswith("Error:")
    assert "blocked" in out.lower()


def test_run_shell_network_deny_blocks_curl(monkeypatch):
    policy = sb.SandboxPolicy(enabled=True, network="deny")
    monkeypatch.setattr(sb, "load_policy", lambda cfg=None: policy)

    out = shell.run_shell("curl http://example.com", timeout=5)
    assert out.startswith("Error:")
    assert "network" in out.lower()


def test_run_shell_benign_command_still_works():
    out = shell.run_shell("echo hello-sandbox-test", timeout=10)
    assert "[exit 0]" in out
    assert "hello-sandbox-test" in out


# --------------------------------------------------------------------------- #
# Memory watchdog (the real enforcement mechanism on Windows, where POSIX
# rlimits via `resource` aren't available)
# --------------------------------------------------------------------------- #
def test_psutil_is_available_so_memory_enforcement_is_real_not_degraded():
    assert sb.psutil is not None


def test_memwatchdog_kills_process_over_its_memory_budget():
    proc = subprocess.Popen(
        [sys.executable, "-c",
         "import time; b=bytearray(200*1024*1024); time.sleep(10)"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    watchdog = sb._MemWatchdog(proc.pid, mem_mb=50, interval=0.1)
    watchdog.start()
    try:
        proc.wait(timeout=8)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()
    finally:
        watchdog.stop()
    assert watchdog.triggered is True


def test_run_sandboxed_reports_memory_kill():
    policy = sb.SandboxPolicy(enabled=True, mem_mb=50, watchdog_interval=0.1)
    code = "import time; b=bytearray(200*1024*1024); time.sleep(10)"
    proc = sb.run_sandboxed([sys.executable, "-c", code], shell=False,
                            timeout=15, policy=policy)
    assert proc.returncode == -9
    assert "killed" in proc.stdout.lower()
    assert "memory" in proc.stdout.lower()


def test_run_sandboxed_does_not_kill_well_behaved_process():
    policy = sb.SandboxPolicy(enabled=True, mem_mb=4096, watchdog_interval=0.1)
    proc = sb.run_sandboxed([sys.executable, "-c", "print('fine')"],
                            shell=False, timeout=10, policy=policy)
    assert proc.returncode == 0
    assert "fine" in proc.stdout
    assert "killed" not in proc.stdout.lower()


# --------------------------------------------------------------------------- #
# config.py -> sandbox.load_policy() wiring
# --------------------------------------------------------------------------- #
def test_load_policy_reads_settings_defaults():
    cfg = config.Settings()
    policy = sb.load_policy(cfg)
    assert policy.enabled is True
    assert policy.mem_mb == 4096
    assert policy.network == "allow"


def test_load_policy_respects_settings_overrides():
    cfg = config.Settings(sandbox_enabled=False, sandbox_mem_mb=256, sandbox_network="deny")
    policy = sb.load_policy(cfg)
    assert policy.enabled is False
    assert policy.mem_mb == 256
    assert policy.network == "deny"
