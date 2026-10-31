"""Prepare, start, resume, or inspect a detached scale demo without automatic retries.

The runner owns scientific-plan validation and per-cell checkpoints. This launcher
only preserves source, logs, exit status, and a separate tmux server per campaign.
Credentials are read by the runner from its configured environment file, never
copied into the source archive, launch script, or tmux environment.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import tarfile
import time


DEFAULT_CONFIG = "configs/demo.structured_nq.json"
DEFAULT_API_CONFIG = "configs/api.glm52.json"
SOURCE_DIRECTORIES = ("specnet", "tools", "tests")
SOURCE_METADATA = ("README.md", ".gitignore", "pyproject.toml", "requirements-experiment.txt")


def isolated_environment():
    # Do not inherit API keys, proxy tunnels, Python import overrides, shell init
    # files, or a parent tmux socket. Provider transport already uses trust_env=False.
    names = ("PATH", "HOME", "USER", "LOGNAME", "LANG", "LC_ALL", "LC_CTYPE",
             "TERM", "TMPDIR", "TZ", "XDG_RUNTIME_DIR")
    environment = {name: os.environ[name] for name in names if name in os.environ}
    environment.setdefault("PATH", "/usr/local/bin:/usr/bin:/bin")
    environment["PYTHONUNBUFFERED"] = "1"
    return environment


def tmux_identity(output):
    suffix = hashlib.sha256(str(output).encode()).hexdigest()[:12]
    return "specnet-scale-" + suffix, "scale-" + suffix


def read_json(path, default=None):
    if not path.is_file():
        return default
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError) as exc:
        return {"read_error": type(exc).__name__, "path": str(path)}


def write_private_json(path, value):
    # These files contain paths and arguments only, but need no shared access.
    with path.open("x", encoding="utf-8") as stream:
        os.chmod(path, 0o600)
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def source_snapshot(project, output):
    """Archive only named source roots; deliberately exclude configs/data/envs."""
    paths = []
    for directory in SOURCE_DIRECTORIES:
        for path in sorted((project / directory).rglob("*.py")):
            relative = path.relative_to(project)
            if (path.is_file() and "__pycache__" not in relative.parts
                    and not any(part.startswith(".") for part in relative.parts)
                    and not any(parent.is_symlink() for parent in (path, *path.parents)
                                if parent != project and project in parent.parents)
                    and path.resolve().is_relative_to(project)):
                paths.append(path)
    paths.extend(project / name for name in SOURCE_METADATA
                 if (project / name).is_file() and not (project / name).is_symlink())
    destination = output / "CODE_SNAPSHOT.tar.gz"
    if destination.exists():
        raise FileExistsError("source snapshot already exists")
    manifest = {str(path.relative_to(project)): hashlib.sha256(path.read_bytes()).hexdigest()
                for path in paths}
    with tarfile.open(destination, "x:gz") as archive:
        for path in paths:
            archive.add(path, arcname=str(path.relative_to(project)), recursive=False)
    destination.chmod(0o600)
    result = dict(archive=str(destination),
                  sha256=hashlib.sha256(destination.read_bytes()).hexdigest(),
                  files=manifest,
                  scope="Python source and project metadata only; no datasets, configs, credentials, or outputs")
    write_private_json(output / "CODE_SNAPSHOT.json", result)
    return result


def launch_script(project, command, log, exit_file):
    # No eval, credential substitution, login shell, or shell initialization.
    return ("#!/usr/bin/env bash\nset -u\numask 077\n"
            "scale_demo_write_exit() {\n"
            "  local experiment_exit_status=$?\n"
            "  trap - EXIT\n"
            "  printf '{\"exit_code\":%s}\\n' \"$experiment_exit_status\" > "
            + shlex.quote(str(exit_file)) + "\n"
            "  exit \"$experiment_exit_status\"\n"
            "}\n"
            "trap scale_demo_write_exit EXIT\n"
            "trap 'exit 130' INT\n"
            "trap 'exit 143' TERM\n"
            "exec > " + shlex.quote(str(log)) + " 2>&1\n"
            "cd -- " + shlex.quote(str(project)) + " || exit 1\n"
            + shlex.join(command) + "\n"
            "exit \"$?\"\n")


def status_report(output, socket, session, live, tmux_available):
    launcher_root = output / "launcher"
    launches = sorted(path for path in launcher_root.iterdir() if path.is_dir()) if launcher_root.is_dir() else []
    latest = launches[-1] if launches else None
    return dict(output=str(output), tmux_socket=socket, tmux_session=session,
                tmux_available=tmux_available, session_alive=live,
                status=read_json(output / "STATUS.json", {"state": "not_started"}),
                usage=read_json(output / "USAGE.json"),
                usage_note="Persisted runner accounting; a running unit may not yet appear in USAGE.json.",
                launch_attempts=len(launches),
                latest_launch=(dict(directory=str(latest), log=str(latest / "run.log"),
                                    exit=read_json(latest / "EXIT.json")) if latest else None))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("start", "resume", "status"))
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--config", type=Path,
                        help=f"start only; default {DEFAULT_CONFIG}; resume uses frozen PLAN.config")
    parser.add_argument("--api-config", type=Path,
                        help=f"default {DEFAULT_API_CONFIG}; resume reuses its launch metadata path")
    parser.add_argument("--through", choices=("development", "calibration", "main"), default="main")
    args = parser.parse_args(argv)
    if args.config and args.action != "start":
        parser.error("--config is only accepted for start; resume uses the frozen PLAN.json")
    project = Path(__file__).resolve().parents[1]
    output = args.output.resolve()
    socket, session = tmux_identity(output)
    environment = isolated_environment()
    tmux = shutil.which("tmux", path=environment["PATH"])
    live = bool(tmux and subprocess.run([tmux, "-L", socket, "-f", "/dev/null", "has-session", "-t", session],
                                       capture_output=True, env=environment).returncode == 0)
    if args.action == "status":
        print(json.dumps(status_report(output, socket, session, live, bool(tmux)), ensure_ascii=False, indent=2))
        return
    if not tmux:
        raise SystemExit("tmux is unavailable; no campaign was started")
    if live:
        raise SystemExit("campaign session already running: " + session)
    python = project / ".venv/bin/python"
    if not python.is_file():
        raise SystemExit("project interpreter is missing: " + str(python))
    command = [str(python), "-m", "specnet.scale_demo", "--output", str(output)]
    metadata = read_json(output / "LAUNCHER.json", {})
    if not isinstance(metadata, dict) or "read_error" in metadata:
        raise SystemExit("invalid LAUNCHER.json; original metadata retained")
    api_config = (args.api_config.resolve() if args.api_config else
                  Path(metadata.get("api_config", str(project / DEFAULT_API_CONFIG))).resolve())
    if args.action == "start":
        if output.exists():
            raise SystemExit("output already exists; use resume or a new output path")
        config = args.config.resolve() if args.config else project / DEFAULT_CONFIG
        subprocess.run(command + ["--config", str(config), "--api-config", str(api_config),
                                  "--prepare-only", "--through", args.through],
                       cwd=project, env=environment, check=True)
        if not (output / "PLAN.json").is_file():
            raise SystemExit("prepare completed without PLAN.json; no campaign was started")
        snapshot = source_snapshot(project, output)
        write_private_json(output / "LAUNCHER.json", dict(
            project=str(project), api_config=str(api_config), tmux_socket=socket,
            tmux_session=session, source_archive_sha256=snapshot["sha256"],
            credential_handling="Runner reads its configured credential file; no inherited API keys.",
        ))
    elif not (output / "PLAN.json").is_file():
        raise SystemExit("no frozen PLAN.json to resume")
    launch = output / "launcher" / str(time.time_ns())
    launch.mkdir(parents=True, mode=0o700)
    log, exit_file = launch / "run.log", launch / "EXIT.json"
    script = launch / "run.sh"
    command += ["--api-config", str(api_config), "--resume", "--through", args.through]
    script.write_text(launch_script(project, command, log, exit_file))
    script.chmod(0o700)
    write_private_json(launch / "LAUNCH.json", dict(action=args.action, command=command,
                                                   tmux_socket=socket, tmux_session=session))
    # Each campaign owns its server and sanitized environment, so an unrelated
    # long-lived tmux server cannot supply stale proxy or credential variables.
    try:
        started = subprocess.run([tmux, "-L", socket, "-f", "/dev/null", "new-session", "-d", "-s", session,
                                  "-c", str(project), shlex.join(["bash", str(script)])],
                                 env=environment, capture_output=True, text=True)
    except OSError as exc:
        write_private_json(exit_file, dict(exit_code=127, stage="tmux_launch", error=type(exc).__name__))
        log.write_text("tmux launch failed: " + type(exc).__name__ + "\n")
        log.chmod(0o600)
        raise SystemExit("tmux launch failed; failure retained at " + str(launch)) from exc
    if started.returncode:
        write_private_json(exit_file, dict(exit_code=started.returncode, stage="tmux_launch"))
        log.write_text(started.stdout + started.stderr)
        log.chmod(0o600)
        raise SystemExit("tmux launch failed; launch metadata and exit status retained at " + str(launch))
    print(json.dumps(dict(tmux_socket=socket, session=session, output=str(output),
                          through=args.through, log=str(log), exit_file=str(exit_file)), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
