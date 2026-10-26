#!/usr/bin/env python3
"""Serve a results folder to a browser, with HTTP Range so a long WAV plays and seeks without being downloaded whole.

    python tools/serve_results.py --root OUT --audio-dir AUDIO            # then http://localhost:8001/lanes/
    ssh -L 8001:localhost:8001 <dgx>                                       # from the laptop

`python -m http.server` sends every file in full and ignores Range, so a 47-minute WAV (~135 MB) has to arrive
before the browser can seek in it. This server answers `Range: bytes=a-b` with 206 and reads the file in chunks,
so the browser fetches only what it is playing. Standard library only.

  /                 the files under --root (directories are listed)
  /_audio/<file>    the files under --audio-dir: the lanes page links the original recording there when a variant has
                    no wav of its own (a variant without any filter and with the music steps off)

It binds to 127.0.0.1 by default. --host 0.0.0.0 makes the folder readable by anyone who can reach the port.
"""
import argparse
import html
import mimetypes
import os
import posixpath
import re
import sys
import urllib.parse
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

CHUNK = 1 << 20
AUDIO_PREFIX = "/_audio/"
for ext, mime in ((".wav", "audio/wav"), (".json", "application/json"), (".html", "text/html"),
                  (".npy", "application/octet-stream")):
    mimetypes.add_type(mime, ext)


class Handler(BaseHTTPRequestHandler):
    server_version = "ResultsServer/1.0"
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):
        if getattr(self.server, "verbose", False):
            super().log_message(fmt, *args)

    def do_HEAD(self):
        self._serve(head=True)

    def do_GET(self):
        self._serve(head=False)

    # -- path handling
    def _resolve(self):
        """Filesystem path for the request, or None. Lexical check only: `..` segments are refused, symlinks that
        live inside the tree are followed (the line's input links point at the original recordings)."""
        raw = self.path.split("?", 1)[0].split("#", 1)[0]
        try:
            path = urllib.parse.unquote(raw, errors="strict")
        except UnicodeDecodeError:
            return None
        if not path.startswith("/") or "\x00" in path or ".." in path.split("/"):
            return None
        if path.startswith(AUDIO_PREFIX):
            base, rel = self.server.audio_dir, path[len(AUDIO_PREFIX):]
        else:
            base, rel = self.server.root, path
        if base is None:
            return None
        parts = [p for p in posixpath.normpath("/" + rel).split("/") if p]
        return Path(os.path.join(str(base), *parts)), path

    def _send_error(self, code, extra=None):
        body = f"{code.value} {code.phrase}\n".encode()
        self.send_response(code)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _serve(self, head):
        resolved = self._resolve()
        if resolved is None:
            return self._send_error(HTTPStatus.NOT_FOUND)
        target, path = resolved
        if target.is_dir():
            if not path.endswith("/"):
                self.send_response(HTTPStatus.MOVED_PERMANENTLY)
                self.send_header("Location", urllib.parse.quote(path) + "/")
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            return self._listing(target, path, head)
        if not target.is_file():
            return self._send_error(HTTPStatus.NOT_FOUND)
        try:
            return self._file(target, head)
        except (BrokenPipeError, ConnectionResetError):
            self.close_connection = True

    # -- directory listing
    def _listing(self, target, path, head):
        try:
            names = sorted(os.listdir(target), key=lambda n: (not (target / n).is_dir(), n.lower()))
        except OSError:
            return self._send_error(HTTPStatus.NOT_FOUND)
        rows = "\n".join(
            f'<li><a href="{urllib.parse.quote(n)}{"/" if (target / n).is_dir() else ""}">'
            f'{html.escape(n)}{"/" if (target / n).is_dir() else ""}</a></li>' for n in names)
        body = (f'<!doctype html><meta charset="utf-8"><title>{html.escape(path)}</title>'
                f"<h3>{html.escape(path)}</h3><ul>{rows}</ul>").encode()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        if not head:
            self.wfile.write(body)

    # -- one file, whole or a byte range
    def _range(self, size):
        """(start, end, partial) | "bad" for an unsatisfiable range. A header that is not a single simple range is
        ignored, as RFC 9110 allows, and the whole file is sent."""
        header = self.headers.get("Range")
        m = re.fullmatch(r"bytes=(\d*)-(\d*)", (header or "").strip())
        if not m or not (m.group(1) or m.group(2)):
            return 0, size - 1, False
        if m.group(1) == "":
            n = int(m.group(2))
            start, end = max(0, size - n), size - 1
            if n == 0:
                return "bad"
        else:
            start = int(m.group(1))
            end = min(int(m.group(2)), size - 1) if m.group(2) else size - 1
        if size == 0 or start >= size or start > end:
            return "bad"
        return start, end, True

    def _file(self, target, head):
        size = target.stat().st_size
        rng = self._range(size)
        if rng == "bad":
            return self._send_error(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE,
                                    {"Content-Range": f"bytes */{size}"})
        start, end, partial = rng
        ctype = mimetypes.guess_type(str(target))[0] or "application/octet-stream"
        if ctype in ("text/html", "text/plain", "application/json"):
            ctype += "; charset=utf-8"
        self.send_response(HTTPStatus.PARTIAL_CONTENT if partial else HTTPStatus.OK)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(max(0, end - start + 1)))
        self.send_header("Accept-Ranges", "bytes")
        if partial:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        if head or size == 0:
            return
        remaining = end - start + 1
        with open(target, "rb") as f:
            f.seek(start)
            while remaining > 0:
                chunk = f.read(min(CHUNK, remaining))
                if not chunk:
                    break
                self.wfile.write(chunk)
                remaining -= len(chunk)


class ResultsServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, address, root, audio_dir=None, verbose=False):
        super().__init__(address, Handler)
        self.root = Path(root)
        self.audio_dir = Path(audio_dir) if audio_dir else None
        self.verbose = verbose


def make_server(root, audio_dir=None, host="127.0.0.1", port=0, verbose=False):
    return ResultsServer((host, port), root, audio_dir, verbose)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=".", help="results folder to serve (default: the current folder)")
    ap.add_argument("--audio-dir", help="folder with the original recordings, served at /_audio/")
    ap.add_argument("--port", type=int, default=8001)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--verbose", action="store_true", help="log every request")
    a = ap.parse_args(argv)
    root = Path(a.root).resolve()
    if not root.is_dir():
        ap.error(f"--root is not a folder: {root}")
    if a.audio_dir and not Path(a.audio_dir).is_dir():
        ap.error(f"--audio-dir is not a folder: {a.audio_dir}")
    server = make_server(root, a.audio_dir, a.host, a.port, a.verbose)
    base = f"http://{'localhost' if a.host in ('127.0.0.1', '0.0.0.0') else a.host}:{server.server_address[1]}"
    print(f"serving {root}" + (f" and {Path(a.audio_dir).resolve()} at /_audio/" if a.audio_dir else ""))
    pages = sorted((root / "lanes").glob("*.html")) + ([root / "compare.html"] if (root / "compare.html").is_file() else [])
    for p in pages:
        print(f"  {base}/{p.relative_to(root).as_posix()}")
    print(f"  {base}/        (all files)")
    print(f"from the laptop:  ssh -L {server.server_address[1]}:localhost:{server.server_address[1]} <dgx>")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print()
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
