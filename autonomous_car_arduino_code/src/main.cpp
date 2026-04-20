#include <Arduino.h>

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

void setup() {
  pinMode(RPWM_L, OUTPUT);
  pinMode(LPWM_L, OUTPUT);
  pinMode(REN_L, OUTPUT);
  pinMode(LEN_L, OUTPUT);

  pinMode(RPWM_R, OUTPUT);
  pinMode(LPWM_R, OUTPUT);
  pinMode(REN_R, OUTPUT);
  pinMode(LEN_R, OUTPUT);

  // Enable drivers
  digitalWrite(REN_L, HIGH);
  digitalWrite(LEN_L, HIGH);
  digitalWrite(REN_R, HIGH);
  digitalWrite(LEN_R, HIGH);
}

void loop() {

  // 🔵 Start Forward (low speed)
  setForward(80);
  delay(2000);

  // 🔼 Gradually increase speed
  rampForward(80, 200, 20);

  delay(2000);

  // 🔽 Gradually decrease speed
  rampForward(200, 0, 20);

  delay(2000);

  stopMotors();
  delay(2000);

  // 🔴 Backward
  setBackward(80);
  delay(2000);

  rampBackward(80, 200, 20);

  delay(2000);

  rampBackward(200, 0, 20);

  stopMotors();
  delay(3000);
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