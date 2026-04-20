from __future__ import annotations

import socketserver
import threading
from http import server
from typing import Optional

import cv2

from ..config import CameraConfig
from ..logging_setup import get_logger
from .capture import FrameSource

log = get_logger(__name__)

_BOUNDARY = "frame"


def _make_handler(source: FrameSource, quality: int):
    class MJPEGHandler(server.BaseHTTPRequestHandler):
        def log_message(self, format, *args):  # noqa: A002 - stdlib signature
            log.debug("http: " + format, *args)

        def do_GET(self):  # noqa: N802 - stdlib signature
            if self.path not in ("/", "/stream.mjpg"):
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header("Content-Type", f"multipart/x-mixed-replace; boundary={_BOUNDARY}")
            self.end_headers()
            try:
                while True:
                    frame = source.get_frame()
                    if frame is None:
                        continue
                    ok, jpg = cv2.imencode(
                        ".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), quality]
                    )
                    if not ok:
                        continue
                    self.wfile.write(f"--{_BOUNDARY}\r\n".encode())
                    self.send_header("Content-Type", "image/jpeg")
                    self.send_header("Content-Length", str(len(jpg)))
                    self.end_headers()
                    self.wfile.write(jpg.tobytes())
                    self.wfile.write(b"\r\n")
            except (BrokenPipeError, ConnectionResetError):
                pass

    return MJPEGHandler


class _ThreadingServer(socketserver.ThreadingMixIn, server.HTTPServer):
    allow_reuse_address = True
    daemon_threads = True


class MJPEGStreamer:
    """Lightweight MJPEG-over-HTTP preview stream for debugging."""

    def __init__(self, source: FrameSource, cfg: CameraConfig):
        self.cfg = cfg
        self.source = source
        self._server: Optional[_ThreadingServer] = None
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        if not self.cfg.stream_enabled:
            log.info("mjpeg stream disabled by config")
            return
        handler = _make_handler(self.source, self.cfg.jpeg_quality)
        self._server = _ThreadingServer((self.cfg.stream_host, self.cfg.stream_port), handler)
        self._thread = threading.Thread(
            target=self._server.serve_forever, name="MJPEGStreamer", daemon=True
        )
        self._thread.start()
        log.info("mjpeg stream on http://%s:%d/stream.mjpg",
                 self.cfg.stream_host, self.cfg.stream_port)

    def stop(self) -> None:
        if self._server:
            self._server.shutdown()
            self._server.server_close()
        if self._thread:
            self._thread.join(timeout=2.0)
