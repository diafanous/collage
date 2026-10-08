"""py server.py [port]  ->  opens the UI at http://127.0.0.1:8765"""
import json
import sys
import threading
import time
import uuid
import webbrowser
import zipfile
from functools import lru_cache
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote

import collage

ROOT = Path(__file__).parent
STORE = {}   # id -> collage.Source, for this session only


@lru_cache
def font(weight):
    """Paper Mono woff2, read straight out of the zip in ./fonts (the linked folder)."""
    for z in (ROOT / "fonts").glob("*.zip"):
        with zipfile.ZipFile(z) as f:
            for n in f.namelist():
                if n.endswith(f"webfonts/PaperMono-{weight}.woff2"):
                    return f.read(n)


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def reply(self, code, body=b"", ctype="application/json", **headers):
        if isinstance(body, (dict, list)):
            body = json.dumps(body).encode()
        try:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            for k, v in headers.items():
                self.send_header(k.replace("_", "-"), v)
            self.end_headers()
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass   # the browser gave up on a stale preview

    def guarded(self, fn):
        host, origin = self.headers.get("Host", ""), self.headers.get("Origin")
        if host.split(":")[0] not in ("127.0.0.1", "localhost") or origin not in (None, "http://" + host):
            return self.reply(403, {"error": "forbidden"})   # DNS rebinding / cross-site POST
        try:
            fn()
        except (ValueError, KeyError, TypeError, IndexError, json.JSONDecodeError) as e:
            self.reply(415 if "can't read" in str(e) else 400, {"error": str(e) or "bad request"})
        except Exception as e:   # keep the server alive; the UI shows the message
            self.reply(500, {"error": f"{type(e).__name__}: {e}"})

    def body(self):
        return self.rfile.read(int(self.headers.get("Content-Length") or 0))

    def do_GET(self):
        def go():
            if self.path == "/":
                self.reply(200, (ROOT / "index.html").read_bytes(), "text/html; charset=utf-8")
            elif self.path.startswith("/font/") and self.path.endswith(".woff2") and self.path[6:-6] in ("Light", "Regular", "Medium"):
                f = font(self.path[6:-6])
                self.reply(200 if f else 404, f or b"", "font/woff2", Cache_Control="max-age=86400")
            else:
                self.reply(404, {"error": "not found"})
        self.guarded(go)

    def do_POST(self):
        def go():
            if self.path == "/source":
                name = unquote(self.headers.get("X-Name", "untitled"))
                s = collage.open_source(name, self.body())
                sid = uuid.uuid4().hex[:8]
                STORE[sid] = s
                self.reply(200, dict(id=sid, name=name, pdf=s.pdf, pages=s.pages, w=s.w_in, aspect=s.aspect, assumed=s.assumed))
            elif self.path == "/preview":
                req, t0 = json.loads(self.body()), time.perf_counter()
                jpg = collage.preview(req["p"], req["sources"], STORE)
                self.reply(200, jpg, "image/jpeg", X_Ms=str(round((time.perf_counter() - t0) * 1000)))
            elif self.path.startswith("/export/"):
                fmt, req = self.path[8:], json.loads(self.body())
                data = collage.export(req["p"], req["sources"], STORE, fmt)
                self.reply(200, data, "application/octet-stream")
            else:
                self.reply(404, {"error": "not found"})
        self.guarded(go)

    def do_DELETE(self):
        def go():
            STORE.pop(self.path.removeprefix("/source/"), None)
            self.reply(200, {})
        self.guarded(go)


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8765
    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    print(f"collage  http://127.0.0.1:{port}  (ctrl+c to quit)")
    threading.Timer(0.4, webbrowser.open, [f"http://127.0.0.1:{port}"]).start()
    server.serve_forever()
