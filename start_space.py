from __future__ import annotations

import os
import signal
import time
from pathlib import Path


def _listener_inodes(port: int) -> set[str]:
    target_port = f"{port:04X}"
    inodes: set[str] = set()
    for table_path in ("/proc/net/tcp", "/proc/net/tcp6"):
        table = Path(table_path)
        if not table.exists():
            continue
        lines = table.read_text(encoding="utf-8").splitlines()[1:]
        for line in lines:
            columns = line.split()
            if len(columns) < 10:
                continue
            local_address = columns[1]
            state = columns[3]
            inode = columns[9]
            _, port_hex = local_address.rsplit(":", 1)
            if state == "0A" and port_hex == target_port:
                inodes.add(inode)
    return inodes


def _pid_command(pid: int) -> str:
    try:
        raw = Path(f"/proc/{pid}/cmdline").read_bytes()
    except OSError:
        return ""
    return raw.replace(b"\x00", b" ").decode("utf-8", errors="replace").strip()


def _listener_pids(port: int) -> list[int]:
    inodes = _listener_inodes(port)
    if not inodes:
        return []

    pids: list[int] = []
    for proc_dir in Path("/proc").iterdir():
        if not proc_dir.name.isdigit():
            continue
        fd_dir = proc_dir / "fd"
        try:
            fds = list(fd_dir.iterdir())
        except OSError:
            continue
        for fd in fds:
            try:
                target = os.readlink(fd)
            except OSError:
                continue
            if target.startswith("socket:[") and target[8:-1] in inodes:
                pids.append(int(proc_dir.name))
                break
    return sorted(set(pids))


def _terminate_stale_listeners(port: int) -> None:
    stale_pids = [pid for pid in _listener_pids(port) if pid not in {os.getpid(), os.getppid()}]
    if not stale_pids:
        return

    print(f"Port {port} is already in use; terminating stale listener(s): {stale_pids}", flush=True)
    for pid in stale_pids:
        try:
            print(f" - pid {pid}: {_pid_command(pid)}", flush=True)
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            continue

    deadline = time.time() + 10.0
    while time.time() < deadline:
        remaining = [pid for pid in stale_pids if Path(f"/proc/{pid}").exists()]
        if not remaining and not _listener_pids(port):
            return
        time.sleep(0.25)

    remaining = [pid for pid in _listener_pids(port) if pid not in {os.getpid(), os.getppid()}]
    for pid in remaining:
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            continue

    if _listener_pids(port):
        raise RuntimeError(f"Port {port} is still busy after terminating stale listeners")


def main() -> None:
    port = int(os.environ.get("PORT") or os.environ.get("SPACE_APP_PORT") or "7860")
    os.environ.setdefault("STREAMLIT_BROWSER_GATHER_USAGE_STATS", "false")
    os.environ.setdefault("STREAMLIT_SERVER_FILE_WATCHER_TYPE", "none")
    os.environ.setdefault("STREAMLIT_SERVER_RUN_ON_SAVE", "false")

    _terminate_stale_listeners(port)

    os.execvp(
        "python3",
        [
            "python3",
            "-m",
            "streamlit",
            "run",
            "app.py",
            f"--server.port={port}",
            "--server.address=0.0.0.0",
            "--server.headless=true",
            "--server.fileWatcherType=none",
            "--server.runOnSave=false",
            "--browser.gatherUsageStats=false",
        ],
    )


if __name__ == "__main__":
    main()
