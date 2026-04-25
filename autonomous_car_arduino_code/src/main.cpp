#include <Arduino.h>

// =====================================================================
// Autocar Arduino Uno firmware.
//
// Two build modes. Change `MODE` to pick one and re-upload.
//
// MODE_ECHO:
//   USB-serial echo loop, no motor output. BTS7960 enables held LOW so
//   the drivers are hardware-disabled. Use this to verify the serial link
//   (baud, port, framing).
//
// MODE_DRIVE (default):
//   On boot, Arduino asks for its PID config via serial and blocks in
//   setup() until the host sends it:
//
//     host → Arduino:   C <kp> <ki> <kd> <pwm_min> <pwm_max>\n
//     Arduino replies:  CFG_OK          (accepted, motors armed)
//                       CFG_ERR <reason>  (stayed in CFG request loop)
//
//   Once running, the host feeds per-loop lane-center errors and can
//   pause/resume drive at any time:
//     host → Arduino:   E <offset>\n    (offset in [-1.0, 1.0])
//     host → Arduino:   S\n             (pause; motors off, drivers disabled)
//     host → Arduino:   G\n             (resume; drivers re-armed, PID reset)
//
//   There is NO internal time limit — the Pi owns the pause/resume cadence.
//   Safety still lives at two layers:
//     * if no E arrives inside OFFSET_STALE_MS the PID falls back to
//       offset 0 (go straight), rather than reusing a stale command;
//     * on S, the drivers are hardware-disabled, the PID state is wiped,
//       and the main loop ignores further E commands until G arrives.
//
//   Async chatter (Arduino → host):
//     READY    on boot (right after Serial is up)
//     CFG?     requesting configuration
//     RUNNING  motors armed, waiting for E commands (boot or after G)
//     PAUSED   S received; ignoring E until G
// =====================================================================

#define MODE_ECHO  0
#define MODE_DRIVE 1
#define MODE       MODE_DRIVE

static const unsigned long BAUD_RATE    = 115200;
static const char          LINE_TERM    = '\n';
static const size_t        BUFFER_LIMIT = 120;

// ===== BTS 1 (LEFT SIDE - REVERSED) =====
int RPWM_L = 5;
int LPWM_L = 6;
int REN_L  = 4;
int LEN_L  = 7;

// ===== BTS 2 (RIGHT SIDE - NORMAL) =====
int RPWM_R = 9;
int LPWM_R = 10;
int REN_R  = 12;
int LEN_R  = 13;

// Function declarations
void setForward(int speedVal);
void setBackward(int speedVal);
void rampForward(int startSpeed, int endSpeed, int stepDelay);
void rampBackward(int startSpeed, int endSpeed, int stepDelay);
void stopMotors();
void enableDrivers(bool on);

#if MODE == MODE_ECHO
// ---------------------------------------------------------------------
// Echo mode
// ---------------------------------------------------------------------

static const unsigned long ECHO_DELAY_MS = 100;
String rxBuffer;

void setup() {
  pinMode(RPWM_L, OUTPUT); pinMode(LPWM_L, OUTPUT);
  pinMode(REN_L,  OUTPUT); pinMode(LEN_L,  OUTPUT);
  pinMode(RPWM_R, OUTPUT); pinMode(LPWM_R, OUTPUT);
  pinMode(REN_R,  OUTPUT); pinMode(LEN_R,  OUTPUT);

  enableDrivers(false);
  stopMotors();

  Serial.begin(BAUD_RATE);
  Serial.println("READY");
}

void loop() {
  while (Serial.available() > 0) {
    const char c = (char)Serial.read();
    if (c == '\r') continue;
    if (c == LINE_TERM) {
      delay(ECHO_DELAY_MS);
      Serial.print("ECHO: ");
      Serial.println(rxBuffer);
      rxBuffer = "";
      continue;
    }
    if (rxBuffer.length() < BUFFER_LIMIT) rxBuffer += c;
  }
}

#elif MODE == MODE_DRIVE
// ---------------------------------------------------------------------
// Drive mode: PID + 30s safety + CFG handshake
// ---------------------------------------------------------------------

static const unsigned long PID_PERIOD_MS   = 20UL;     // 50 Hz PID
static const unsigned long OFFSET_STALE_MS = 500UL;    // fall back to 0 if no update

// PID gains + PWM clamp supplied by the host at boot.
float kp = 0.0f;
float ki = 0.0f;
float kd = 0.0f;
int   pwm_min = 0;
int   pwm_max = 0;

// Runtime state
float latest_offset   = 0.0f;
unsigned long offset_t = 0;
unsigned long last_pid_t  = 0;
float pid_integral   = 0.0f;
float pid_prev_error = 0.0f;
bool  paused          = false;

String rxBuffer;

bool parseConfigLine(const String &line);
void processRunLine(const String &line);
void driveFromSteering(float steering);
void pauseDrive();
void resumeDrive();

void setup() {
  pinMode(RPWM_L, OUTPUT); pinMode(LPWM_L, OUTPUT);
  pinMode(REN_L,  OUTPUT); pinMode(LEN_L,  OUTPUT);
  pinMode(RPWM_R, OUTPUT); pinMode(LPWM_R, OUTPUT);
  pinMode(REN_R,  OUTPUT); pinMode(LEN_R,  OUTPUT);

  // Hardware-disabled until we've got a config.
  enableDrivers(false);
  stopMotors();

  Serial.begin(BAUD_RATE);
  delay(200);                 // settle after USB auto-reset
  Serial.println("READY");
  Serial.println("CFG?");

  // Block in setup() until a valid config arrives.
  rxBuffer = "";
  for (;;) {
    while (Serial.available() > 0) {
      const char c = (char)Serial.read();
      if (c == '\r') continue;
      if (c == LINE_TERM) {
        if (parseConfigLine(rxBuffer)) {
          rxBuffer = "";
          goto config_done;
        }
        rxBuffer = "";
        continue;
      }
      if (rxBuffer.length() < BUFFER_LIMIT) rxBuffer += c;
    }
  }
config_done:

  enableDrivers(true);
  pid_integral = 0.0f;
  pid_prev_error = 0.0f;
  latest_offset = 0.0f;
  offset_t = 0;
  last_pid_t = millis();

  Serial.println("RUNNING");
}

void loop() {
  // Drain any incoming bytes.
  while (Serial.available() > 0) {
    const char c = (char)Serial.read();
    if (c == '\r') continue;
    if (c == LINE_TERM) {
      processRunLine(rxBuffer);
      rxBuffer = "";
      continue;
    }
    if (rxBuffer.length() < BUFFER_LIMIT) rxBuffer += c;
  }

  const unsigned long now = millis();

  // PID tick at 50 Hz. Skipped while paused — motors stay off and the
  // PID state is reset in pauseDrive(), so resume starts from a clean slate.
  if (!paused && (now - last_pid_t >= PID_PERIOD_MS)) {
    const float dt = (now - last_pid_t) * 0.001f;
    last_pid_t = now;

    const bool fresh = (offset_t > 0) && ((now - offset_t) < OFFSET_STALE_MS);
    const float err = fresh ? latest_offset : 0.0f;

    pid_integral += err * dt;
    const float deriv = (dt > 0.0f) ? ((err - pid_prev_error) / dt) : 0.0f;
    float steering = kp * err + ki * pid_integral + kd * deriv;
    pid_prev_error = err;
    if (steering >  1.0f) steering =  1.0f;
    if (steering < -1.0f) steering = -1.0f;

    driveFromSteering(steering);
  }
}

// Parses "C <kp> <ki> <kd> <pwm_min> <pwm_max>". Returns true on success,
// false (and prints CFG_ERR) on failure.
bool parseConfigLine(const String &line) {
  if (line.length() < 2 || line.charAt(0) != 'C') {
    Serial.println("CFG_ERR expected 'C <kp> <ki> <kd> <pwm_min> <pwm_max>'");
    return false;
  }
  float vals[5] = {0, 0, 0, 0, 0};
  int count = 0;
  int i = 1;
  const int n = (int)line.length();
  while (i < n && count < 5) {
    while (i < n && line.charAt(i) == ' ') i++;
    const int start = i;
    while (i < n && line.charAt(i) != ' ') i++;
    if (start < i) {
      vals[count++] = line.substring(start, i).toFloat();
    }
  }
  if (count != 5) {
    Serial.print("CFG_ERR need 5 args got "); Serial.println(count);
    return false;
  }
  const int p_min = (int)vals[3];
  const int p_max = (int)vals[4];
  if (p_min < 0 || p_max <= p_min || p_max > 255) {
    Serial.println("CFG_ERR pwm_min/max out of range");
    return false;
  }
  kp = vals[0]; ki = vals[1]; kd = vals[2];
  pwm_min = p_min; pwm_max = p_max;
  Serial.println("CFG_OK");
  return true;
}

void processRunLine(const String &line) {
  if (line.length() == 0) return;
  const char cmd = line.charAt(0);
  if (cmd == 'E') {
    if (paused) return;   // ignore offsets while paused — Pi may keep streaming
    String rest = line.substring(1);
    rest.trim();
    latest_offset = rest.toFloat();
    offset_t = millis();
  } else if (cmd == 'S') {
    pauseDrive();
  } else if (cmd == 'G') {
    resumeDrive();
  }
  // Unknown command lines are silently ignored.
}

// Write a SIGNED PWM to a BTS7960 side (left or right).
// `isLeft=true` uses the LEFT driver (wiring is reversed on this side).
// Positive pwm = forward, negative = reverse. Magnitude is |pwm|.
static void driveSideSigned(bool isLeft, int signed_pwm) {
  const int m = abs(signed_pwm);
  if (isLeft) {
    // LEFT SIDE (REVERSED wiring) — see setForward().
    if (signed_pwm >= 0) {
      analogWrite(RPWM_L, 0);
      analogWrite(LPWM_L, m);
    } else {
      analogWrite(RPWM_L, m);
      analogWrite(LPWM_L, 0);
    }
  } else {
    // RIGHT SIDE (normal).
    if (signed_pwm >= 0) {
      analogWrite(RPWM_R, m);
      analogWrite(LPWM_R, 0);
    } else {
      analogWrite(RPWM_R, 0);
      analogWrite(LPWM_R, m);
    }
  }
}

// Differential mix with SIGNED per-side PWM so one side can reverse during a
// sharp turn. Goes-straight baseline is (pwm_min+pwm_max)/2, steering
// authority is pwm_max — at steering=±1 one side clamps to +pwm_max while
// the other clamps to roughly -pwm_max/2 (wheel reverses).
//
// +steering == turn right (left side faster / right side slower or reverse).
void driveFromSteering(float steering) {
  const int base      = (pwm_min + pwm_max) / 2;
  const int authority = pwm_max;
  const int delta     = (int)(steering * (float)authority);
  int left  = base + delta;
  int right = base - delta;
  // Clamp each side to [-pwm_max, +pwm_max]. A side is allowed to go past
  // pwm_min into reverse (down to -pwm_max) when steering saturates.
  if (left  >  pwm_max) left  =  pwm_max;
  if (left  < -pwm_max) left  = -pwm_max;
  if (right >  pwm_max) right =  pwm_max;
  if (right < -pwm_max) right = -pwm_max;
  driveSideSigned(true,  left);
  driveSideSigned(false, right);
}

void pauseDrive() {
  stopMotors();
  enableDrivers(false);
  pid_integral   = 0.0f;
  pid_prev_error = 0.0f;
  latest_offset  = 0.0f;
  offset_t       = 0;
  paused         = true;
  Serial.println("PAUSED");
}

void resumeDrive() {
  enableDrivers(true);
  pid_integral   = 0.0f;
  pid_prev_error = 0.0f;
  latest_offset  = 0.0f;
  offset_t       = 0;
  last_pid_t     = millis();
  paused         = false;
  Serial.println("RUNNING");
}

#endif  // MODE selection

// =====================================================================
// Shared helpers
// =====================================================================

void enableDrivers(bool on) {
  digitalWrite(REN_L, on ? HIGH : LOW);
  digitalWrite(LEN_L, on ? HIGH : LOW);
  digitalWrite(REN_R, on ? HIGH : LOW);
  digitalWrite(LEN_R, on ? HIGH : LOW);
}

// ===== FORWARD (LEFT FIXED) =====
void setForward(int speedVal) {

  // LEFT SIDE (REVERSED)
  analogWrite(RPWM_L, 0);
  analogWrite(LPWM_L, speedVal);

  // RIGHT SIDE (NORMAL)
  analogWrite(RPWM_R, speedVal);
  analogWrite(LPWM_R, 0);
}

// ===== BACKWARD =====
void setBackward(int speedVal) {

  // LEFT SIDE (REVERSED)
  analogWrite(RPWM_L, speedVal);
  analogWrite(LPWM_L, 0);

  // RIGHT SIDE (NORMAL)
  analogWrite(RPWM_R, 0);
  analogWrite(LPWM_R, speedVal);
}

// ===== RAMP FORWARD =====
void rampForward(int startSpeed, int endSpeed, int stepDelay) {

  if (startSpeed < endSpeed) {
    for (int s = startSpeed; s <= endSpeed; s++) {
      setForward(s);
      delay(stepDelay);
    }
  } else {
    for (int s = startSpeed; s >= endSpeed; s--) {
      setForward(s);
      delay(stepDelay);
    }
  }
}

// ===== RAMP BACKWARD =====
void rampBackward(int startSpeed, int endSpeed, int stepDelay) {

  if (startSpeed < endSpeed) {
    for (int s = startSpeed; s <= endSpeed; s++) {
      setBackward(s);
      delay(stepDelay);
    }
  } else {
    for (int s = startSpeed; s >= endSpeed; s--) {
      setBackward(s);
      delay(stepDelay);
    }
  }
}

// ===== STOP =====
void stopMotors() {
  analogWrite(RPWM_L, 0);
  analogWrite(LPWM_L, 0);
  analogWrite(RPWM_R, 0);
  analogWrite(LPWM_R, 0);
}
