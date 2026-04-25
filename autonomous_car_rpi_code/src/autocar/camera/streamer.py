from __future__ import annotations

import json
import socketserver
import threading
from http import server
from pathlib import Path
from typing import Callable, Optional

import cv2

from ..config import AppConfig, CameraConfig
from ..logging_setup import get_logger
from .. import tuning
from ..yaml_patch import patch_yaml_file

log = get_logger(__name__)

_BOUNDARY = "frame"


class _JpegBroker:
    """Single-slot pub/sub for the latest encoded MJPEG frame. The streamer's
    encoder thread is the sole publisher; every connected `/stream.mjpg`
    viewer blocks in `wait_next` until a strictly-newer frame is available.
    Decoupling encode from each handler thread (a) caps stream FPS at the
    producer rate, (b) collapses duplicate sends, and (c) amortizes JPEG
    encode cost across viewers (one encode for N watchers)."""

    def __init__(self):
        self._cv = threading.Condition()
        self._jpeg: Optional[bytes] = None
        self._frame_id: int = 0
        self._closed = False

    def publish(self, frame_id: int, jpeg: bytes) -> None:
        with self._cv:
            self._jpeg = jpeg
            self._frame_id = frame_id
            self._cv.notify_all()

    def wait_next(self, after_id: int, timeout: float = 2.0):
        with self._cv:
            self._cv.wait_for(
                lambda: self._closed or self._frame_id > after_id,
                timeout=timeout,
            )
            if self._closed or self._frame_id <= after_id or self._jpeg is None:
                return after_id, None
            return self._frame_id, self._jpeg

    def close(self) -> None:
        with self._cv:
            self._closed = True
            self._cv.notify_all()


def _make_handler(source, quality: int, tuner_hooks: "TunerHooks",
                  broker: "_JpegBroker"):
    class MJPEGHandler(server.BaseHTTPRequestHandler):
        def log_message(self, format, *args):  # noqa: A002 - stdlib signature
            log.debug("http: " + format, *args)

        def do_GET(self):  # noqa: N802 - stdlib signature
            if self.path in ("/stream.mjpg",):
                self._serve_mjpeg()
                return
            if self.path in ("/raw.jpg", "/raw"):
                self._serve_snapshot(raw=True)
                return
            if self.path in ("/annotated.jpg", "/annotated"):
                self._serve_snapshot(raw=False)
                return
            if self.path in ("/", "/tune", "/tune/"):
                self._serve_text(_TUNER_HTML, "text/html; charset=utf-8")
                return
            if self.path.startswith("/config.json"):
                self._serve_json(tuner_hooks.snapshot())
                return
            if self.path.startswith("/knobs.json"):
                self._serve_json([
                    {
                        "label": k.label, "path": k.path,
                        "min": k.min_val, "max": k.max_val, "step": k.step,
                        "kind": k.kind,
                    }
                    for k in tuning.KNOBS
                ])
                return
            if self.path.startswith("/run.json"):
                self._serve_json({"running": tuner_hooks.running()})
                return
            self.send_error(404)

        def do_POST(self):  # noqa: N802
            length = int(self.headers.get("Content-Length") or 0)
            body = self.rfile.read(length) if length > 0 else b""
            if self.path.startswith("/config"):
                try:
                    payload = json.loads(body.decode("utf-8") or "{}")
                except Exception as e:
                    self.send_error(400, f"bad json: {e}")
                    return
                applied = tuner_hooks.apply(payload)
                self._serve_json({"applied": applied})
                return
            if self.path.startswith("/save"):
                try:
                    path = tuner_hooks.save()
                except Exception as e:
                    self.send_error(500, f"save failed: {e}")
                    return
                self._serve_json({"saved": str(path)})
                return
            if self.path.startswith("/run"):
                try:
                    payload = json.loads(body.decode("utf-8") or "{}")
                except Exception as e:
                    self.send_error(400, f"bad json: {e}")
                    return
                running = bool(payload.get("running", True))
                actual = tuner_hooks.set_running(running)
                self._serve_json({"running": actual})
                return
            self.send_error(404)

        # --- helpers ---

        def _serve_mjpeg(self) -> None:
            self.send_response(200)
            self.send_header(
                "Content-Type", f"multipart/x-mixed-replace; boundary={_BOUNDARY}"
            )
            self.end_headers()
            last = -1
            try:
                while True:
                    fid, jpg = broker.wait_next(last)
                    if jpg is None:
                        continue       # timeout — loop, no spin
                    last = fid
                    self.wfile.write(f"--{_BOUNDARY}\r\n".encode())
                    self.send_header("Content-Type", "image/jpeg")
                    self.send_header("Content-Length", str(len(jpg)))
                    self.end_headers()
                    self.wfile.write(jpg)
                    self.wfile.write(b"\r\n")
            except (BrokenPipeError, ConnectionResetError):
                pass

        def _serve_snapshot(self, raw: bool) -> None:
            getter = getattr(source, "get_raw_frame", None) if raw else None
            frame = getter() if getter else source.get_frame()
            if frame is None:
                self.send_error(503, "no frame")
                return
            ok, jpg = cv2.imencode(
                ".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), quality]
            )
            if not ok:
                self.send_error(500, "encode failed")
                return
            self.send_response(200)
            self.send_header("Content-Type", "image/jpeg")
            self.send_header("Content-Length", str(len(jpg)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(jpg.tobytes())

        def _serve_text(self, text: str, ctype: str) -> None:
            data = text.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)

        def _serve_json(self, obj) -> None:
            data = json.dumps(obj).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)

    return MJPEGHandler


class TunerHooks:
    """Closure over the live AppConfig + yaml save path. Thread-safe getters
    and a mutex around apply/save so concurrent HTTP requests don't race
    the control loop. Also exposes a run/pause toggle so the web UI can
    e-stop the car without killing the process."""

    def __init__(self, config: AppConfig, config_path: Optional[Path] = None,
                 on_apply: Optional[Callable[[dict], None]] = None,
                 set_running: Optional[Callable[[bool], None]] = None,
                 get_running: Optional[Callable[[], bool]] = None):
        self._config = config
        self._path = config_path
        self._on_apply = on_apply
        self._set_running = set_running
        self._get_running = get_running
        self._lock = threading.Lock()

    def snapshot(self) -> dict:
        with self._lock:
            return tuning.dump_snapshot(self._config)

    def apply(self, patch: dict) -> dict:
        applied: dict = {}
        with self._lock:
            for key, value in patch.items():
                try:
                    actual = tuning.set_value(self._config, key, value)
                    applied[key] = actual
                except Exception as e:
                    log.warning("tuner apply %s=%s failed: %s", key, value, e)
            if self._on_apply is not None:
                try:
                    self._on_apply(applied)
                except Exception as e:
                    log.warning("tuner on_apply hook failed: %s", e)
        return applied

    def save(self) -> Path:
        with self._lock:
            if self._path is None:
                raise RuntimeError("no config path configured")
            updates = tuning.dump_snapshot(self._config)
            patch_yaml_file(self._path, updates)
            log.info("tuner wrote %d values back to %s", len(updates), self._path)
            return self._path

    def running(self) -> bool:
        return bool(self._get_running()) if self._get_running else True

    def set_running(self, running: bool) -> bool:
        if self._set_running is not None:
            self._set_running(bool(running))
        return self.running()


class _ThreadingServer(socketserver.ThreadingMixIn, server.HTTPServer):
    allow_reuse_address = True
    daemon_threads = True


class MJPEGStreamer:
    """Serves:
        /stream.mjpg   — continuous MJPEG feed (annotated or raw, depending
                         on what the source returns)
        /raw.jpg       — one-shot un-annotated frame
        /annotated.jpg — one-shot HUD-annotated frame
        /              — HTML live-tuner page with sliders + save button
        /config.json   — GET current tunable values (JSON)
        /knobs.json    — GET spec of all sliders (JSON)
        /config        — POST {path: value, ...} to mutate config live
        /save          — POST to write current values back to yaml
        /run.json      — GET {"running": bool}
        /run           — POST {"running": bool} to e-stop / resume"""

    def __init__(self, source, cfg: CameraConfig,
                 tuner_hooks: Optional[TunerHooks] = None):
        self.cfg = cfg
        self.source = source
        self._server: Optional[_ThreadingServer] = None
        self._thread: Optional[threading.Thread] = None
        self._tuner_hooks = tuner_hooks
        # JPEG encode is centralised on a dedicated thread so HTTP handlers
        # don't each pay encode CPU per frame. Handlers wait on the broker
        # for a strictly-newer encoded buffer.
        self._broker = _JpegBroker()
        self._encoder_thread: Optional[threading.Thread] = None
        self._encoder_stop = threading.Event()

    def start(self) -> None:
        if not self.cfg.stream_enabled:
            log.info("mjpeg stream disabled by config")
            return
        hooks = self._tuner_hooks or TunerHooks(_NullConfig())
        handler = _make_handler(self.source, self.cfg.jpeg_quality, hooks,
                                self._broker)
        self._server = _ThreadingServer(
            (self.cfg.stream_host, self.cfg.stream_port), handler
        )
        self._thread = threading.Thread(
            target=self._server.serve_forever, name="MJPEGStreamer", daemon=True
        )
        self._thread.start()
        self._encoder_stop.clear()
        self._encoder_thread = threading.Thread(
            target=self._encoder_loop, name="MJPEGEncoder", daemon=True,
        )
        self._encoder_thread.start()
        log.info(
            "mjpeg on http://%s:%d/  (tuner UI on /  —  stream.mjpg | raw.jpg | annotated.jpg)",
            self.cfg.stream_host, self.cfg.stream_port,
        )

    def stop(self) -> None:
        self._encoder_stop.set()
        self._broker.close()
        if self._server:
            self._server.shutdown()
            self._server.server_close()
        if self._thread:
            self._thread.join(timeout=2.0)
        if self._encoder_thread:
            self._encoder_thread.join(timeout=2.0)

    def _encoder_loop(self) -> None:
        """Wait on the AnnotatedFrameProvider for each fresh frame, encode
        once, publish to the broker. If the source doesn't support `wait_next`
        (legacy), we degrade to polling at 30 Hz."""
        last = -1
        quality = int(self.cfg.jpeg_quality)
        wait_next = getattr(self.source, "wait_next", None)
        while not self._encoder_stop.is_set():
            if wait_next is not None:
                fid, frame = wait_next(last, timeout=0.5)
            else:
                # Legacy poll fallback — never the case for AnnotatedFrameProvider.
                self._encoder_stop.wait(1.0 / 30)
                fid, frame = (last + 1, self.source.get_frame())
            if frame is None:
                continue
            try:
                ok, jpg = cv2.imencode(
                    ".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), quality]
                )
            except Exception as e:
                log.debug("jpeg encode failed: %s", e)
                continue
            if not ok:
                continue
            last = fid
            self._broker.publish(fid, jpg.tobytes())


class _NullConfig:
    """Fallback for callers that haven't wired a tuner — exposes an empty
    snapshot and rejects mutations."""

    def __getattr__(self, item):
        raise AttributeError(item)


# ---------------------------------------------------------------------------
# Tuner HTML (served at `/`). Plain page, no external deps — works on the
# Pi even if the browser machine has no internet.
# ---------------------------------------------------------------------------

_TUNER_HTML = """<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>autocar tuner</title>
  <style>
    :root { color-scheme: dark; }
    body { margin: 0; background:#111; color:#ddd; font-family: ui-sans-serif, system-ui, sans-serif; }
    .wrap { display:flex; flex-direction:row; gap:12px; padding:12px; align-items:flex-start; }
    .stream { flex:0 0 auto; border:1px solid #333; }
    .stream img { display:block; max-width: 640px; max-height: 480px; }
    .controls { flex:1 1 auto; display:grid; grid-template-columns: 1fr; gap:8px; font-size: 13px; }
    .knob { display:grid; grid-template-columns: 170px 1fr 64px; gap:8px; align-items:center; }
    .knob label { color:#aaa; }
    .knob input[type=range] { width: 100%; accent-color:#8be9fd; }
    .knob .value { text-align:right; color:#f1fa8c; font-variant-numeric: tabular-nums; }
    .section { margin-top: 10px; color:#50fa7b; border-bottom: 1px solid #333; padding-bottom: 2px; }
    .actions { display:flex; gap:8px; margin-top: 12px; }
    button { background:#282a36; color:#f8f8f2; border:1px solid #444; border-radius:4px;
             padding:6px 12px; cursor:pointer; font-size: 13px; }
    button:hover { background:#3a3c52; }
    button:disabled { opacity:0.4; cursor: default; }
    #status { color:#6272a4; margin-left: 8px; }
    .runbar { display:flex; gap:8px; align-items:center; margin-bottom: 10px; }
    button.estop { background:#5b1620; border-color:#d9434b; color:#ffd5da;
                   font-weight:600; padding:10px 22px; font-size: 15px; }
    button.estop:hover:not(:disabled) { background:#7a1c2a; }
    button.go    { background:#143b1a; border-color:#3ec45a; color:#cdfbd6;
                   font-weight:600; padding:10px 22px; font-size: 15px; }
    button.go:hover:not(:disabled) { background:#1f5a27; }
    .runstate { font-weight:600; font-variant-numeric: tabular-nums; }
    .runstate.on  { color:#50fa7b; }
    .runstate.off { color:#ff5555; }
  </style>
</head>
<body>
  <div class="wrap">
    <div class="stream">
      <div class="runbar">
        <button id="stop"  class="estop">⏹ E-STOP</button>
        <button id="start" class="go">▶ START</button>
        <span class="runstate" id="runstate">…</span>
      </div>
      <img src="/stream.mjpg" alt="live" />
    </div>
    <div class="controls" id="controls">
      <div class="section">loading…</div>
    </div>
  </div>
  <script>
  (async () => {
    const ctrlEl = document.getElementById('controls');
    const [knobs, values] = await Promise.all([
      fetch('/knobs.json').then(r => r.json()),
      fetch('/config.json').then(r => r.json()),
    ]);

    // Group knobs by the leading cfg path segment (vision / control).
    const groups = {};
    for (const k of knobs) {
      const g = k.path.split('.')[0];
      (groups[g] ||= []).push(k);
    }

    const fmt = (k, v) => k.kind === 'float' ? Number(v).toFixed(3) : String(Math.round(Number(v)));

    let html = '';
    for (const [g, items] of Object.entries(groups)) {
      html += `<div class="section">${g}</div>`;
      for (const k of items) {
        const v = values[k.path];
        html += `
        <div class="knob" data-path="${k.path}" data-kind="${k.kind}">
          <label title="${k.path}">${k.label}</label>
          <input type="range" min="${k.min}" max="${k.max}" step="${k.step}" value="${v}">
          <span class="value">${fmt(k, v)}</span>
        </div>`;
      }
    }
    html += `
      <div class="actions">
        <button id="save">Save → default.yaml</button>
        <button id="revert">Revert (reload from Pi)</button>
        <span id="status"></span>
      </div>`;
    ctrlEl.innerHTML = html;

    const status = document.getElementById('status');
    const setStatus = (t) => { status.textContent = t; };

    // Debounced POST to /config
    const pending = {};
    let timer = null;
    function schedulePush() {
      if (timer) return;
      timer = setTimeout(async () => {
        timer = null;
        const body = JSON.stringify(pending);
        Object.keys(pending).forEach(k => delete pending[k]);
        try {
          const res = await fetch('/config', {method:'POST', body});
          const j = await res.json();
          // Sync UI with the actually-applied (clamped / rounded) values.
          for (const [p, v] of Object.entries(j.applied || {})) {
            const row = document.querySelector(`.knob[data-path="${CSS.escape(p)}"]`);
            if (!row) continue;
            const kind = row.dataset.kind;
            row.querySelector('input').value = v;
            row.querySelector('.value').textContent = kind === 'float'
              ? Number(v).toFixed(3) : String(Math.round(Number(v)));
          }
          setStatus('');
        } catch (e) {
          setStatus('push failed: ' + e);
        }
      }, 60);
    }

    ctrlEl.addEventListener('input', (ev) => {
      const row = ev.target.closest('.knob');
      if (!row) return;
      const path = row.dataset.path;
      const value = ev.target.value;
      row.querySelector('.value').textContent = row.dataset.kind === 'float'
        ? Number(value).toFixed(3) : String(Math.round(Number(value)));
      pending[path] = value;
      schedulePush();
    });

    document.getElementById('save').onclick = async () => {
      setStatus('saving…');
      try {
        const res = await fetch('/save', {method:'POST', body:'{}'});
        const j = await res.json();
        setStatus('saved → ' + j.saved);
      } catch (e) {
        setStatus('save failed: ' + e);
      }
    };

    document.getElementById('revert').onclick = async () => {
      // "Revert" just reloads the page so current yaml values repopulate.
      // (The Pi keeps its live values; this resets the browser view only.)
      setStatus('reloading…');
      location.reload();
    };

    // --- E-STOP / START -----------------------------------------------------
    const stopBtn  = document.getElementById('stop');
    const startBtn = document.getElementById('start');
    const runEl    = document.getElementById('runstate');
    const reflectRun = (running) => {
      stopBtn.disabled  = !running;   // can't stop if already stopped
      startBtn.disabled = running;    // can't start if already running
      runEl.textContent = running ? 'RUNNING' : 'STOPPED';
      runEl.className   = 'runstate ' + (running ? 'on' : 'off');
    };

    async function postRun(running) {
      try {
        const res = await fetch('/run', {
          method:'POST',
          body: JSON.stringify({running}),
        });
        const j = await res.json();
        reflectRun(j.running);
        setStatus(j.running ? 'running' : 'STOPPED');
      } catch (e) { setStatus('run toggle failed: ' + e); }
    }

    stopBtn.onclick  = () => postRun(false);
    startBtn.onclick = () => postRun(true);

    // Spacebar = panic stop. Skip when typing in a control.
    document.addEventListener('keydown', (e) => {
      if (e.code === 'Space' && !['INPUT','TEXTAREA'].includes(e.target.tagName)) {
        e.preventDefault();
        postRun(false);
      }
    });

    // Initial state from server.
    fetch('/run.json').then(r => r.json()).then(j => reflectRun(j.running))
      .catch(() => reflectRun(true));
  })();
  </script>
</body>
</html>
"""
