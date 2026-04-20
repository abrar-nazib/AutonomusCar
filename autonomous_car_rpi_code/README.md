# autocar — Raspberry Pi side

Raspberry Pi controller for a basic line-following autonomous car. The Pi:

1. Captures frames from the Pi Camera (falls back to black frames if the camera is missing).
2. Optionally streams a debug MJPEG preview over HTTP.
3. Detects the track's center point with OpenCV.
4. Runs a PID on the lateral offset.
5. Sends per-side motor commands over UART to the Arduino, which drives the BTS7960 H-bridges.

## Layout

```
autonomous_car_rpi_code/
├── pyproject.toml            # package metadata + deps
├── config/
│   └── default.yaml          # runtime config (camera, vision, PID, UART)
├── src/autocar/
│   ├── __main__.py           # CLI entry point (python -m autocar)
│   ├── app.py                # control loop orchestrator
│   ├── config.py             # dataclass config + YAML loader
│   ├── logging_setup.py
│   ├── camera/               # frame capture + MJPEG debug stream
│   ├── vision/               # track detection
│   ├── control/              # PID + differential mixer
│   └── comms/                # UART link to the Arduino
└── tests/                    # pytest unit tests
```

## Install

### On the Raspberry Pi (recommended)

Do **not** `pip install picamera2` — it pulls `python-prctl` / `libcamera` native builds and typically fails (`libcap` dev headers, libcamera mismatch, etc.). Use the apt-packaged `picamera2` and make the venv see system packages:

```bash
sudo apt update
sudo apt install -y python3-picamera2 python3-libcamera python3-kms++

python3 -m venv --system-site-packages .venv
source .venv/bin/activate
pip install -e .            # note: NOT ".[pi]" — picamera2 comes from apt
```

Verify: `python -c "from picamera2 import Picamera2"` should import with no error.

> **numpy / opencv ABI note:** the apt-packaged `picamera2` is compiled against system numpy 1.x. `pyproject.toml` therefore pins `numpy<2` and `opencv-python<4.11` (opencv-python ≥ 4.13 forces numpy ≥ 2, which breaks picamera2). Do not relax these pins on the Pi. If you ever see `numpy.dtype size changed, may indicate binary incompatibility` when importing `picamera2`, a numpy 2.x got into your venv — fix with `pip install --force-reinstall "numpy<2" "opencv-python<4.11"`.

### On a dev machine (no Pi Camera)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"     # adds pytest + ruff
```

The code falls back to black frames when `picamera2` isn't installed, so the full pipeline still runs.

### The `[pi]` extra

`pip install -e ".[pi]"` exists for completeness, but on Raspberry Pi OS it will fail unless you first `sudo apt install -y libcap-dev libcamera-dev python3-libcamera` and accept a pip-built libcamera stack. The apt path above is simpler and better supported.

## Run

```bash
python -m autocar                       # uses config/default.yaml
python -m autocar -c path/to/other.yaml
```

The debug MJPEG preview (when enabled) is at `http://<pi-ip>:8000/stream.mjpg`.

## UART wire format

ASCII line `M <left> <right>\n`, with `left` and `right` signed PWM in `[-255, 255]`.
Negative values mean reverse on that side. The Arduino firmware is expected to parse
this and drive the motors accordingly (with a watchdog that stops motors if no
command arrives within a timeout).

## Tests

```bash
pytest
```
