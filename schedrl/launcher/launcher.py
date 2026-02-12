from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True, slots=True)
class RayLauncherConfig:
    head: bool
    port: int
    address: Optional[str] = None
    node_name: Optional[str] = None
    dashboard_port: Optional[int] = None


def _base_env() -> dict[str, str]:
    env = dict(os.environ)
    env.setdefault("RAY_kill_child_processes_on_worker_exit", "1")
    env.setdefault("RAY_kill_child_processes_on_worker_exit_with_raylet_subreaper", "1")
    return env


def _ray_cli_path() -> str:
    python_bin_dir = Path(sys.executable).parent
    return str(python_bin_dir / "ray")


def ray_stop_force() -> None:
    subprocess.run([_ray_cli_path(), "stop", "--force"], check=False, env=_base_env())


def ray_start(cfg: RayLauncherConfig) -> None:
    if cfg.head:
        cmd = [_ray_cli_path(), "start", "--head", f"--port={cfg.port}"]
    else:
        if not cfg.address:
            raise ValueError("cfg.address is required when head=False")
        cmd = [_ray_cli_path(), "start", f"--address={cfg.address}"]

    if cfg.node_name:
        cmd.append(f"--node-name={cfg.node_name}")
    if cfg.dashboard_port is not None:
        cmd.append(f"--dashboard-port={cfg.dashboard_port}")

    subprocess.run(cmd, check=True, env=_base_env())
