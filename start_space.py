from __future__ import annotations

import fcntl
import os
import socket
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


def _acquire_startup_lock(lock_path: str = "/tmp/catcon_space_repo_start.lock") -> object:
    lock_file = Path(lock_path).open("w")
    fcntl.flock(lock_file, fcntl.LOCK_EX)
    os.set_inheritable(lock_file.fileno(), True)
    return lock_file


def _can_bind_port(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            probe.bind(("0.0.0.0", port))
        except OSError:
            return False
    return True


def _wait_for_port_release(port: int, timeout: float = 120.0) -> None:
    deadline = time.time() + timeout
    last_reported: tuple[int, ...] | None = None

    while True:
        listeners = [pid for pid in _listener_pids(port) if pid not in {os.getpid(), os.getppid()}]
        if not listeners and _can_bind_port(port):
            return

        listener_signature = tuple(listeners)
        if listener_signature != last_reported:
            if listeners:
                print(f"Port {port} is already in use; waiting for listener(s) to exit: {listeners}", flush=True)
                for pid in listeners:
                    print(f" - pid {pid}: {_pid_command(pid)}", flush=True)
            else:
                print(f"Port {port} is still not bindable; waiting before starting Streamlit", flush=True)
            last_reported = listener_signature

        if time.time() >= deadline:
            raise RuntimeError(f"Port {port} is still busy after waiting {timeout:.0f}s for stale listeners to exit")

        time.sleep(1.0)


def main() -> None:
    port = int(os.environ.get("PORT") or os.environ.get("SPACE_APP_PORT") or "7860")
    os.environ.setdefault("STREAMLIT_BROWSER_GATHER_USAGE_STATS", "false")
    os.environ.setdefault("STREAMLIT_SERVER_FILE_WATCHER_TYPE", "none")
    os.environ.setdefault("STREAMLIT_SERVER_RUN_ON_SAVE", "false")

    _startup_lock = _acquire_startup_lock()
    _wait_for_port_release(port)

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
