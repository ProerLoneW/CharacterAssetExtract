"""Keep the viewer running and reload it when its run configuration changes."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path(__file__).with_name('viewer_config.json'))
    parser.add_argument('--host', default=os.environ.get('RESULT_VIEWER_HOST'),
                        help='默认取配置 listen_host')
    parser.add_argument('--port', type=int, default=int(os.environ.get('RESULT_VIEWER_PORT') or 0),
                        help='默认取配置 service_port')
    parser.add_argument('--root-path', default=os.environ.get('RESULT_VIEWER_ROOT_PATH') or '',
                        help='代理保留路径前缀时传入')
    parser.add_argument('--interval', type=float, default=3.0, help='配置轮询间隔秒数')
    args = parser.parse_args()
    config = args.config.resolve()
    repo = Path(__file__).resolve().parents[1]
    stopping = False

    def stop(*_):
        nonlocal stopping
        stopping = True

    def config_snapshot():
        try:
            data = config.read_bytes()
            parsed = json.loads(data)  # Never restart on an incomplete config write.
            return hashlib.sha256(data).hexdigest(), parsed
        except (OSError, ValueError):
            return None

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    child = None
    previous = None
    try:
        while not stopping:
            snapshot = config_snapshot()
            current = snapshot[0] if snapshot else None
            changed = current is not None and current != previous
            if child is None or child.poll() is not None or changed:
                if child is not None and child.poll() is None:
                    print('[supervise] viewer config changed; restarting backend…', flush=True)
                    child.terminate()
                    try:
                        child.wait(timeout=20)
                    except subprocess.TimeoutExpired:
                        child.kill()
                        child.wait()
                if snapshot is not None:
                    raw = snapshot[1]
                    host = args.host or str(raw.get('listen_host') or '0.0.0.0')
                    port = args.port or int(raw.get('service_port') or 8930)
                    cmd = [
                        sys.executable, '-u', '-m', 'result_viewer.backend.app',
                        '--config', str(config), '--host', host, '--port', str(port)
                    ]
                    prefix = args.root_path.rstrip('/')
                    if prefix:
                        cmd += ['--root-path', prefix]
                    child = subprocess.Popen(cmd, cwd=repo)
                    previous = current
                    print(f'[supervise] viewer started: PID={child.pid} '
                          f'http://127.0.0.1:{port}{prefix or "/"}', flush=True)
            time.sleep(max(0.5, args.interval))
    finally:
        if child is not None and child.poll() is None:
            child.terminate()
            try:
                child.wait(timeout=20)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()


if __name__ == '__main__':
    main()
