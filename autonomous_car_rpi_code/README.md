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
├── deploy.sh                 # scp source tree to the Pi
├── config/
│   └── default.yaml          # runtime config (camera, vision, PID, UART, logging)
├── src/autocar/
│   ├── __main__.py           # CLI entry point (python -m autocar)
│   ├── app.py                # control loop orchestrator
│   ├── config.py             # dataclass config + YAML loader
│   ├── logging_setup.py
│   ├── camera/               # picamera2 capture + MJPEG debug stream
│   ├── vision/               # track detection
│   ├── control/              # PID + differential mixer
│   └── comms/                # UART link to the Arduino
└── tests/                    # pytest unit tests
```

## Hardware assumptions

- Raspberry Pi with a Pi Camera (picamera2 stack).
- UART on `/dev/serial0` wired to the Arduino at `115200 baud`.
- Arduino drives two BTS7960 H-bridges (see `../autonomous_car_arduino_code/`).

## First-time setup on the Raspberry Pi

### 1. Enable the camera and UART

```bash
sudo raspi-config
#  → Interface Options → Serial Port
#      "login shell over serial?"  → No
#      "serial hardware enabled?"  → Yes
#  → Interface Options → Camera    → Enable (on older images)
sudo reboot
```

Confirm `/dev/serial0` exists and your user is in the `dialout` group:

```bash
ls -l /dev/serial0
groups                              # should include 'dialout'
sudo usermod -aG dialout "$USER"    # if not; log out/in afterward
```

### 2. Install system packages

Do **not** `pip install picamera2` — it pulls `python-prctl` / `libcamera` native builds and typically fails on the Pi. Use the apt-packaged `picamera2` instead:

```bash
sudo apt update
sudo apt install -y python3-picamera2 python3-libcamera python3-kms++ python3-venv
```

### 3. Get the source on the Pi

From your dev machine (not the Pi), push the source tree with [deploy.sh](deploy.sh):

```bash
./deploy.sh                                  # defaults: abir@raspberrypi.local:~/autonomous_car_rpi_code
REMOTE_HOST=pi@192.168.1.50 ./deploy.sh      # override target
```

`deploy.sh` cleans `__pycache__/` and sends only `pyproject.toml`, `README.md`, `.gitignore`, `config/`, `src/`, `tests/`. `venv/`, caches, and the deploy script itself are never copied.

### 4. Create the venv and install the package

On the Pi:

```bash
cd ~/autonomous_car_rpi_code
python3 -m venv --system-site-packages .venv    # --system-site-packages so apt's picamera2 is visible
source .venv/bin/activate
pip install -e .                                # pulls numpy<2, opencv-python<4.11, pyserial, PyYAML
```

Verify picamera2 imports cleanly:

```bash
python -c "from picamera2 import Picamera2; print('ok')"
```

> **numpy / opencv ABI note:** the apt-packaged `picamera2` is compiled against system numpy 1.x. `pyproject.toml` therefore pins `numpy<2` and `opencv-python<4.11` (opencv-python ≥ 4.13 forces numpy ≥ 2, which breaks picamera2). Do not relax these pins on the Pi. If you ever see `numpy.dtype size changed, may indicate binary incompatibility` when importing `picamera2`, a numpy 2.x got into your venv — fix with `pip install --force-reinstall "numpy<2" "opencv-python<4.11"`.

## Running the car

From the project root on the Pi, inside the venv:

```bash
python -m autocar                       # uses config/default.yaml
python -m autocar -c path/to/other.yaml # use a different config
```

`Ctrl-C` stops the loop; the app sends a zero-PWM stop command to the Arduino on shutdown.

### Watching the camera from your PC

With `camera.stream_enabled: true` in the config (default), open:

```
http://<pi-host>:8000/stream.mjpg
```

Host/port/JPEG quality are under `camera.*` in `config/default.yaml`. The stream is currently bundled with the control loop — running `python -m autocar` both streams video and sends motor commands over UART.

### Rotating the camera feed 180°

If the Pi Camera is mounted upside-down on the chassis, set:

```yaml
camera:
  rotate_180: true
```

This applies a libcamera `Transform(hflip=1, vflip=1)` at camera-open time, so the rotation happens in hardware with no CPU cost and affects both the MJPEG preview and the frames fed into `TrackDetector`. On start-up you should see:

```
INFO [autocar.camera.capture] picamera2 started at 640x480 (rotate_180=True)
```

libcamera only supports 0° and 180° rotations directly. 90° / 270° would require a software rotate in `FrameSource._capture_one` — not currently implemented.

## Run on boot (systemd)

[systemd/autocar.service](systemd/autocar.service) starts `python -m autocar` on boot under the `abir` user, with `dialout` and `video` supplementary groups, and restarts on failure. It assumes the repo lives at `/home/abir/autonomous_car_rpi_code` and the venv at `./.venv` inside it — edit `User=`, `WorkingDirectory=`, and `ExecStart=` if yours differ.

Install on the Pi:

```bash
sudo cp systemd/autocar.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable autocar.service     # start on boot
sudo systemctl start autocar.service      # start now
```

Check status and live logs:

```bash
systemctl status autocar.service
journalctl -u autocar.service -f
```

Stop / disable:

```bash
sudo systemctl stop autocar.service
sudo systemctl disable autocar.service
```

`systemctl stop` sends `SIGTERM`; [app.py](src/autocar/app.py) catches it, sends `M 0 0\n` over UART, and shuts the camera and MJPEG server down. `TimeoutStopSec=5` caps the cleanup window.

Re-deploying (`./deploy.sh`) pushes a new copy of `systemd/autocar.service` to the Pi, but does **not** copy it into `/etc/systemd/system/` — after editing the unit file, re-run the `cp` + `daemon-reload` + `restart` steps above.

## Configuration

[config/default.yaml](config/default.yaml) is the single source of runtime settings. Load it (or any other YAML with the same shape) via `AppConfig.from_yaml`. Key sections:

- `camera` — resolution, framerate, MJPEG host/port/quality, `stream_enabled`, `rotate_180` (hardware 180° flip when the camera is mounted upside-down — see below).
- `vision` — ROI bounds, `binarize_threshold`, `min_contour_area`. Tune per track.
- `control.pid` — `kp`/`ki`/`kd`. Tune on-track.
- `control` — `base_speed`, `max_speed`, `loop_hz`.
- `comms` — `uart_port`, `uart_baud`, `command_timeout_s`.
- `logging` — `level`, optional `file`.

Use a copy, not the default, for per-car overrides:

```bash
cp config/default.yaml config/car.yaml
# edit config/car.yaml
python -m autocar -c config/car.yaml
```

## Dev machine setup (no Pi Camera)

For editing, running tests, and smoke-testing the pipeline off-Pi:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"     # adds pytest + ruff
pytest
```

`FrameSource` falls back to black frames when `picamera2` isn't installed, so `python -m autocar` still runs the full loop — the detector just won't find a track, and the UART link no-ops if `/dev/serial0` is missing.

## UART wire format

ASCII line, one per command:

```
M <left> <right>\n
```

`left` and `right` are signed PWM in `[-255, 255]`. Negative means reverse on that side. The Arduino firmware is expected to parse this and drive the motors accordingly, with a watchdog that stops motors if no command arrives within `comms.command_timeout_s`.

## Troubleshooting

- **`picamera2 unavailable (numpy.dtype size changed ...)`** — numpy 2.x leaked into the venv. `pip install --force-reinstall "numpy<2" "opencv-python<4.11"`.
- **`picamera2 unavailable (No module named 'picamera2')`** — venv wasn't created with `--system-site-packages`, or `python3-picamera2` isn't installed via apt. Fix both.
- **`uart open failed ([Errno 13] Permission denied)`** — add your user to `dialout` (`sudo usermod -aG dialout "$USER"`) and log out/in.
- **`uart open failed ([Errno 2] ... '/dev/serial0')`** — serial hardware not enabled; re-run `sudo raspi-config`.
- **MJPEG stream shows only black frames** — picamera2 failed to open; check `dmesg` and `libcamera-hello --list-cameras`.

## Tests

```bash
pytest
```
