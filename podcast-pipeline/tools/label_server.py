#!/usr/bin/env python3
"""Serve the arm-comparison results and a page to label them by hand, saving the labels next to each sample.

    python tools/label_server.py --root OUT [--arms-dir OUT/diar_arms] [--audio-dir AUDIO] [--port 8002]
    ssh -L 8002:localhost:8002 <dgx>          # from the laptop, then http://localhost:8002/label

It is tools/serve_results.py (every file under --root, HTTP Range for long audio) plus:

    GET  /label                 the labelling page (tools/label_page.html)
    GET  /api/label/samples     the samples with a clip.wav, their arms and the state of their labels
    GET  /api/label/<sample>    the labels of one sample (an empty document when there are none)
    PUT  /api/label/<sample>    save them: {"labels": {...}, "base_rev": 3, "force": false}; 409 when base_rev is stale

Labels land in <arms-dir>/<sample>/labels.json with the last 30 versions in labels_history/ (tools/label_store.py).
It binds to 127.0.0.1 by default and, like serve_results.py, has no login: --host 0.0.0.0 lets anyone who can reach the
port read the results and overwrite the labels.
"""
import argparse
import json
import os
import sys
import urllib.parse
from http import HTTPStatus
from http.server import ThreadingHTTPServer
from pathlib import Path

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import label_store as store  # noqa: E402
import serve_results  # noqa: E402

PAGE = os.path.join(HERE, "label_page.html")


class Handler(serve_results.Handler):
    server_version = "LabelServer/1.0"

    def _json(self, code, obj):
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _route(self):
        path = urllib.parse.unquote(self.path.split("?", 1)[0])
        return path

    def do_GET(self):
        path = self._route()
        if path in ("/label", "/label/"):
            with open(PAGE, "rb") as fh:
                body = fh.read()
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)
            return
        if path == "/api/label/samples":
            return self._json(HTTPStatus.OK, {"samples": store.list_samples(self.server.arms_dir)})
        if path.startswith("/api/label/"):
            sample = path[len("/api/label/"):]
            try:
                return self._json(HTTPStatus.OK, store.load(self.server.arms_dir, sample))
            except (ValueError, FileNotFoundError):
                return self._json(HTTPStatus.NOT_FOUND, {"error": "unknown sample"})
        return super().do_GET()

    def do_PUT(self):
        path = self._route()
        if not path.startswith("/api/label/"):
            return self._json(HTTPStatus.NOT_FOUND, {"error": "not found"})
        sample = path[len("/api/label/"):]
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            length = -1
        if not (0 < length <= store.MAX_BYTES):
            return self._json(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, {"error": "body missing or too large"})
        try:
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            saved = store.save(self.server.arms_dir, sample, payload.get("labels"),
                               base_rev=payload.get("base_rev"), force=bool(payload.get("force")))
        except store.Conflict as exc:
            return self._json(HTTPStatus.CONFLICT, {"error": "conflict", "current": exc.current})
        except FileNotFoundError:
            return self._json(HTTPStatus.NOT_FOUND, {"error": "unknown sample"})
        except (ValueError, UnicodeDecodeError, AttributeError) as exc:
            return self._json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
        return self._json(HTTPStatus.OK, saved)


class LabelServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, address, root, arms_dir=None, audio_dir=None, verbose=False):
        super().__init__(address, Handler)
        self.root = Path(root)
        self.arms_dir = str(arms_dir or (Path(root) / "diar_arms"))
        self.audio_dir = Path(audio_dir) if audio_dir else None
        self.verbose = verbose


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, help="results folder to serve (the OUT of the runs)")
    ap.add_argument("--arms-dir", help="folder of the sample folders (default: <root>/diar_arms)")
    ap.add_argument("--audio-dir", help="folder with the original recordings, served at /_audio/")
    ap.add_argument("--port", type=int, default=8002)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--verbose", action="store_true")
    a = ap.parse_args(argv)
    root = Path(a.root).resolve()
    arms = Path(a.arms_dir).resolve() if a.arms_dir else root / "diar_arms"
    if not arms.is_dir():
        ap.error(f"no sample folder: {arms}")
    server = LabelServer((a.host, a.port), root, arms, a.audio_dir, a.verbose)
    port = server.server_address[1]
    n = len(store.list_samples(str(arms)))
    print(f"serving {root}; {n} sample(s) to label in {arms}")
    print(f"  http://localhost:{port}/label")
    print(f"from the laptop:  ssh -L {port}:localhost:{port} <dgx>")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print()
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
