// GRIPPER_DENGFOC_P_FINAL_CAL459_ESP32 firmware version: v3.0-FINAL-INTERFACES
// Adapted from ToanTech/DengFOC_Lib (GPL-2.0).
// Source: https://github.com/ToanTech/DengFOC_Lib
#include <Arduino.h>
#include <driver/twai.h>
#include <esp_system.h>
#include "grip_can.h"
#include "grip_control.h"
#include "grip_motion.h"
#include "final_radio.h"
#include <Wire.h>
#include <Preferences.h>
#include <math.h>
#include <string.h>
HardwareSerial ArmLink(2);
constexpr int ARM_RX_PIN = 32;
constexpr int ARM_TX_PIN = 33;

// Drain telemetry only into available UART capacity; never wait for the wire.
class BufferedTelemetry : public Print {
 public:
  explicit BufferedTelemetry(HardwareSerial &serial) : port(serial) {}
  size_t write(uint8_t value) override {
    if (pending == sizeof(data)) { ++dropped; return 0; }
    data[tail] = value; tail = (tail + 1) % sizeof(data); ++pending;
    return 1;
  }
  void drain() {
    int capacity = port.availableForWrite();
    if (capacity <= 0 || pending == 0) return;
    size_t count = pending;
    if (count > static_cast<size_t>(capacity)) count = capacity;
    if (count > 64) count = 64;
    if (count > sizeof(data) - head) count = sizeof(data) - head;
    const size_t sent = port.write(data + head, count);
    head = (head + sent) % sizeof(data); pending -= sent;
  }
  size_t freeSpace() const { return sizeof(data) - pending; }
  uint32_t dropped = 0;
 private:
  HardwareSerial &port;
  uint8_t data[4096];
  size_t head = 0, tail = 0, pending = 0;
};
BufferedTelemetry UsbOutput(Serial), ArmOutput(ArmLink);
uint32_t controlMaxPeriodUs = 0;

// Mechanical-design monitor edition of the proven DengFOC pure-P branch.
// Boundaries are monitored and reported but never disable or latch the motor.
constexpr int PWM_A_PIN = 25;
constexpr int PWM_B_PIN = 13;
constexpr int PWM_C_PIN = 14;
constexpr int ENABLE_PIN = 27;
constexpr int AS5600_SDA_PIN = 22;
constexpr int AS5600_SCL_PIN = 21;

constexpr uint32_t PWM_FREQUENCY_HZ = 30000;
constexpr uint8_t PWM_RESOLUTION_BITS = 10;
constexpr uint8_t PWM_CHANNEL_A = 0;
constexpr uint8_t PWM_CHANNEL_B = 1;
constexpr uint8_t PWM_CHANNEL_C = 2;
constexpr uint32_t PWM_MAX_DUTY = (1UL << PWM_RESOLUTION_BITS) - 1UL;

constexpr int MOTOR_POLE_PAIRS = 7;
constexpr int SENSOR_DIRECTION = -1;
constexpr float BUS_VOLTAGE = 12.0f;
// Repeated 12 V SimpleFOC calibration after restoring the original phase order:
// zero electric angle repeatedly measured near 4.59 rad, direction remained -1.
constexpr float ZERO_ELECTRIC_ANGLE_RAD = 4.585066f;

// Exact DengFOC V0.1 example gain: Uq = 0.133 * position_error_degrees.
// The original 12 V example limits Uq to half the DC bus.
constexpr float POSITION_KP_V_PER_DEG = 0.133f;
constexpr float UQ_LIMIT_V = BUS_VOLTAGE * 0.5f;
constexpr float DEFAULT_TARGET_SLEW_RAD_S = 2.00f;
constexpr float MIN_TARGET_SLEW_RAD_S = 0.10f;
constexpr float MAX_TARGET_SLEW_RAD_S = 2.00f;

constexpr float PI_F = 3.14159265359f;
constexpr float TWO_PI_F = 6.28318530718f;
constexpr float PI_OVER_2_F = 1.57079632679f;
constexpr float PI_OVER_3_F = 1.04719755120f;
constexpr float SQRT3_F = 1.73205080757f;
constexpr float RAD_TO_DEG_F = 57.2957795131f;

constexpr float OPEN_ANGLE_RAD = -0.388097f;
constexpr float DEFAULT_CLOSED_ANGLE_RAD = -4.534447f;
constexpr float MIN_ALLOWED_CLOSED_ANGLE_RAD = -4.600000f;
constexpr float MAX_ALLOWED_CLOSED_ANGLE_RAD = -4.400000f;
constexpr float ABSOLUTE_CLOSED_HARD_RAD = -4.614447f;
constexpr float GAP_MM_PER_MOTOR_RAD = 18.0f;
constexpr float MIN_COMMAND_GAP_MM = 0.0f;

constexpr char CONFIG_NVS_NAMESPACE[] = "grip-deng-adj";
constexpr uint32_t CONFIG_NVS_MAGIC = 0x44475031UL;
constexpr uint32_t CONFIG_SCHEMA_VERSION = 2UL;

// Pure-P control retains a small static-friction position error. The measured
// Measured residuals were 0.069 rad at open and 0.0813 rad at closed. A 0.090
// rad mechanical-test arrival tolerance avoids false NO_PROGRESS reports. The
// P loop remains active and continues holding the exact target after arrival.
constexpr float TARGET_TOLERANCE_RAD = 0.090f;
constexpr unsigned long TARGET_STABLE_MS = 300;
constexpr float SOFT_LIMIT_TRIGGER_RAD = 0.010f;
constexpr float SOFT_LIMIT_RELEASE_RAD = 0.003f;
constexpr float OPEN_HARD_MARGIN_RAD = 0.300f;
constexpr unsigned long NO_PROGRESS_TIME_MS = 4000;
constexpr float PROGRESS_STEP_RAD = 0.010f;
constexpr uint8_t MAX_CONSECUTIVE_I2C_ERRORS = 3;

float clampValue(float value, float low, float high) {
  if (value < low) return low;
  if (value > high) return high;
  return value;
}

float moveTowards(float current, float target, float maximumStep) {
  return current + clampValue(target - current, -maximumStep, maximumStep);
}

float normalizeAngle(float angleRad) {
  float value = fmodf(angleRad, TWO_PI_F);
  return value >= 0.0f ? value : value + TWO_PI_F;
}

class DengAS5600 {
 public:
  bool begin(TwoWire* bus) {
    wire = bus;
    float firstAngle = 0.0f;
    if (!readMechanicalAngle(firstAngle)) return false;
    delayMicroseconds(10);
    anglePrevious = firstAngle;
    velocityAnglePrevious = firstAngle;
    angleTimestampUs = micros();
    velocityTimestampUs = angleTimestampUs;
    return true;
  }

  bool update() {
    float angle = 0.0f;
    if (!readMechanicalAngle(angle)) {
      if (consecutiveErrors < 255) ++consecutiveErrors;
      return false;
    }

    consecutiveErrors = 0;
    angleTimestampUs = micros();
    const float delta = angle - anglePrevious;
    if (fabsf(delta) > 0.8f * TWO_PI_F) {
      fullRotations += delta > 0.0f ? -1 : 1;
    }
    anglePrevious = angle;
    return true;
  }

  float mechanicalAngle() const {
    return anglePrevious;
  }

  float continuousAngle() const {
    return static_cast<float>(fullRotations) * TWO_PI_F + anglePrevious;
  }

  float velocity() {
    const uint32_t elapsedUs = angleTimestampUs - velocityTimestampUs;
    if (elapsedUs < 100) return lastVelocityRadS;

    const float dt = elapsedUs * 1.0e-6f;
    const float current = static_cast<float>(fullRotations) * TWO_PI_F
                        + anglePrevious;
    const float previous = static_cast<float>(velocityFullRotationsPrevious)
                         * TWO_PI_F + velocityAnglePrevious;
    lastVelocityRadS = (current - previous) / dt;

    // DengFOC updates these values even when angle delta is zero. Therefore a
    // stationary shaft reports 0 instead of retaining an old nonzero speed.
    velocityAnglePrevious = anglePrevious;
    velocityFullRotationsPrevious = fullRotations;
    velocityTimestampUs = angleTimestampUs;
    return lastVelocityRadS;
  }

  uint8_t errorCode() const { return lastWireError; }
  uint8_t errorCount() const { return consecutiveErrors; }

 private:
  bool readMechanicalAngle(float& angleRad) {
    wire->beginTransmission(0x36);
    wire->write(0x0C);
    // AS5600 register-pointer write and read are separate STOP-terminated transfers.
    // Avoid the combined write/read path implicated by the captured IDF panic.
    lastWireError = wire->endTransmission(true);
    if (lastWireError != 0) return false;

    const uint8_t received = wire->requestFrom(0x36, static_cast<uint8_t>(2));
    if (received != 2 || wire->available() < 2) {
      lastWireError = 0xFE;
      return false;
    }

    const uint8_t highByte = wire->read();
    const uint8_t lowByte = wire->read();
    const uint16_t raw =
      (static_cast<uint16_t>(highByte & 0x0F) << 8) | lowByte;
    angleRad = static_cast<float>(raw) * (TWO_PI_F / 4096.0f);
    return true;
  }

  TwoWire* wire = nullptr;
  float anglePrevious = 0.0f;
  uint32_t angleTimestampUs = 0;
  int32_t fullRotations = 0;
  float velocityAnglePrevious = 0.0f;
  uint32_t velocityTimestampUs = 0;
  int32_t velocityFullRotationsPrevious = 0;
  float lastVelocityRadS = 0.0f;
  uint8_t lastWireError = 0;
  uint8_t consecutiveErrors = 0;
};

DengAS5600 sensor;
Preferences preferences;

constexpr uint8_t BOUNDARY_NORMAL = 0;
constexpr uint8_t BOUNDARY_SOFT_OPEN = 1;
constexpr uint8_t BOUNDARY_SOFT_CLOSED = 2;
constexpr uint8_t BOUNDARY_HARD_OPEN = 3;
constexpr uint8_t BOUNDARY_HARD_CLOSED = 4;

bool driveEnabled = false;
bool motionActive = false;
bool targetReported = false;
uint8_t boundaryMonitorState = BOUNDARY_NORMAL;
float branchOffsetRad = 0.0f;
float closedLimitAngleRad = DEFAULT_CLOSED_ANGLE_RAD;
float targetSlewRateRadS = DEFAULT_TARGET_SLEW_RAD_S;
float targetAngleCanonicalRad = OPEN_ANGLE_RAD;
float trajectoryTargetAngleRad = OPEN_ANGLE_RAD;
float trajectoryVelocityRadS = 0.0f;
float currentAngleCanonicalRad = 0.0f;
float measuredVelocityRadS = 0.0f;
float filteredVelocityRadS = 0.0f;
float uqCommandV = 0.0f;
float lastProgressAngleRad = 0.0f;
uint32_t previousVelocityFilterUs = 0;
uint32_t previousTrajectoryUs = 0;
unsigned long lastProgressMs = 0;
unsigned long targetStableSinceMs = 0;

char serialLine[48];
uint8_t serialLength = 0;

float safeClosedAngle() {
  return closedLimitAngleRad;
}

float maximumGapMm() {
  return (OPEN_ANGLE_RAD - closedLimitAngleRad) * GAP_MM_PER_MOTOR_RAD;
}

float canonicalizeAngle(float angleRad) {
  const float midpoint = 0.5f * (OPEN_ANGLE_RAD + closedLimitAngleRad);
  while (angleRad - midpoint > PI_F) angleRad -= TWO_PI_F;
  while (angleRad - midpoint < -PI_F) angleRad += TWO_PI_F;
  return angleRad;
}

float canonicalAngleFromSensor() {
  return SENSOR_DIRECTION * sensor.continuousAngle() - branchOffsetRad;
}

float gapFromCanonicalAngle(float angleRad) {
  return clampValue(
    (angleRad - closedLimitAngleRad) * GAP_MM_PER_MOTOR_RAD,
    0.0f,
    maximumGapMm()
  );
}

float canonicalAngleFromGap(float gapMm) {
  const float safeGap = clampValue(
    gapMm,
    MIN_COMMAND_GAP_MM,
    maximumGapMm()
  );
  return closedLimitAngleRad + safeGap / GAP_MM_PER_MOTOR_RAD;
}

bool angleInsideHardEnvelope(float angleRad) {
  return angleRad >= ABSOLUTE_CLOSED_HARD_RAD
      && angleRad <= OPEN_ANGLE_RAD + OPEN_HARD_MARGIN_RAD;
}

float electricalAngle() {
  return normalizeAngle(
    static_cast<float>(SENSOR_DIRECTION * MOTOR_POLE_PAIRS)
      * sensor.mechanicalAngle()
    - ZERO_ELECTRIC_ANGLE_RAD
  );
}

void writeDuty(uint8_t channel, float duty) {
  const float safeDuty = clampValue(duty, 0.0f, 1.0f);
  ledcWriteChannel(
    channel,
    static_cast<uint32_t>(safeDuty * static_cast<float>(PWM_MAX_DUTY))
  );
}

void setNeutralPwm() {
  writeDuty(PWM_CHANNEL_A, 0.5f);
  writeDuty(PWM_CHANNEL_B, 0.5f);
  writeDuty(PWM_CHANNEL_C, 0.5f);
}

void setPwmOff() {
  writeDuty(PWM_CHANNEL_A, 0.0f);
  writeDuty(PWM_CHANNEL_B, 0.0f);
  writeDuty(PWM_CHANNEL_C, 0.0f);
}

void setTorqueSVPWM(float requestedUqV, float electricalAngleRad) {
  float uqV = clampValue(requestedUqV, -UQ_LIMIT_V, UQ_LIMIT_V);
  if (uqV < 0.0f) electricalAngleRad += PI_F;
  uqV = fabsf(uqV);

  const float angle = normalizeAngle(electricalAngleRad + PI_OVER_2_F);
  int sector = static_cast<int>(floorf(angle / PI_OVER_3_F)) + 1;
  sector = static_cast<int>(clampValue(sector, 1, 6));

  const float t1 = SQRT3_F
                 * sinf(sector * PI_OVER_3_F - angle)
                 * uqV / BUS_VOLTAGE;
  const float t2 = SQRT3_F
                 * sinf(angle - (sector - 1.0f) * PI_OVER_3_F)
                 * uqV / BUS_VOLTAGE;
  const float t0 = 1.0f - t1 - t2;

  float dutyA = 0.5f;
  float dutyB = 0.5f;
  float dutyC = 0.5f;
  switch (sector) {
    case 1:
      dutyA = t1 + t2 + t0 * 0.5f;
      dutyB = t2 + t0 * 0.5f;
      dutyC = t0 * 0.5f;
      break;
    case 2:
      dutyA = t1 + t0 * 0.5f;
      dutyB = t1 + t2 + t0 * 0.5f;
      dutyC = t0 * 0.5f;
      break;
    case 3:
      dutyA = t0 * 0.5f;
      dutyB = t1 + t2 + t0 * 0.5f;
      dutyC = t2 + t0 * 0.5f;
      break;
    case 4:
      dutyA = t0 * 0.5f;
      dutyB = t1 + t0 * 0.5f;
      dutyC = t1 + t2 + t0 * 0.5f;
      break;
    case 5:
      dutyA = t2 + t0 * 0.5f;
      dutyB = t0 * 0.5f;
      dutyC = t1 + t2 + t0 * 0.5f;
      break;
    case 6:
      dutyA = t1 + t2 + t0 * 0.5f;
      dutyB = t0 * 0.5f;
      dutyC = t1 + t0 * 0.5f;
      break;
  }

  writeDuty(PWM_CHANNEL_A, dutyA);
  writeDuty(PWM_CHANNEL_B, dutyB);
  writeDuty(PWM_CHANNEL_C, dutyC);
}

bool initializePwm() {
  pinMode(ENABLE_PIN, OUTPUT);
  digitalWrite(ENABLE_PIN, LOW);
  const bool aOk = ledcAttachChannel(
    PWM_A_PIN, PWM_FREQUENCY_HZ, PWM_RESOLUTION_BITS, PWM_CHANNEL_A);
  const bool bOk = ledcAttachChannel(
    PWM_B_PIN, PWM_FREQUENCY_HZ, PWM_RESOLUTION_BITS, PWM_CHANNEL_B);
  const bool cOk = ledcAttachChannel(
    PWM_C_PIN, PWM_FREQUENCY_HZ, PWM_RESOLUTION_BITS, PWM_CHANNEL_C);
  setPwmOff();
  return aOk && bOk && cOk;
}

bool saveConfiguration() {
  if (!preferences.begin(CONFIG_NVS_NAMESPACE, false)) return false;
  preferences.clear();
  const bool ok = preferences.putUInt("magic", CONFIG_NVS_MAGIC) > 0
               && preferences.putUInt("schema", CONFIG_SCHEMA_VERSION) > 0
               && preferences.putFloat("closed", closedLimitAngleRad) > 0
               && preferences.putFloat("slew", targetSlewRateRadS) > 0;
  preferences.end();
  return ok;
}

void loadConfiguration() {
  if (!preferences.begin(CONFIG_NVS_NAMESPACE, true)) return;
  const uint32_t magic = preferences.getUInt("magic", 0);
  const uint32_t schema = preferences.getUInt("schema", 0);
  const float savedClosed =
    preferences.getFloat("closed", DEFAULT_CLOSED_ANGLE_RAD);
  const float savedSlew =
    preferences.getFloat("slew", DEFAULT_TARGET_SLEW_RAD_S);
  preferences.end();

  const bool validExistingConfig = magic == CONFIG_NVS_MAGIC;
  if (validExistingConfig
      && isfinite(savedClosed)
      && savedClosed >= MIN_ALLOWED_CLOSED_ANGLE_RAD
      && savedClosed <= MAX_ALLOWED_CLOSED_ANGLE_RAD) {
    closedLimitAngleRad = savedClosed;
  }
  if (validExistingConfig
      && schema >= CONFIG_SCHEMA_VERSION
      && isfinite(savedSlew)
      && savedSlew >= MIN_TARGET_SLEW_RAD_S
      && savedSlew <= MAX_TARGET_SLEW_RAD_S) {
    targetSlewRateRadS = savedSlew;
  }

  // v2.1.1 migration: keep the calibrated mechanical closed boundary but
  // replace a legacy saved speed with the new R2 default exactly once.
  if (validExistingConfig && schema < CONFIG_SCHEMA_VERSION) {
    targetSlewRateRadS = DEFAULT_TARGET_SLEW_RAD_S;
    UsbOutput.println(saveConfiguration()
      ? "CONFIG_MIGRATED_DEFAULT_R2"
      : "ERR_CONFIG_MIGRATION_DEFAULT_R2");
  }
}

bool applyClosedBoundary(float requestedAngleRad) {
  if (driveEnabled) {
    UsbOutput.println("ERR_DISABLE_DRIVE_BEFORE_SETTING_CLOSED_BOUNDARY");
    return false;
  }
  if (!isfinite(requestedAngleRad)
      || requestedAngleRad < MIN_ALLOWED_CLOSED_ANGLE_RAD
      || requestedAngleRad > MAX_ALLOWED_CLOSED_ANGLE_RAD) {
    UsbOutput.print("ERR_CLOSED_BOUNDARY_RANGE allowed=");
    UsbOutput.print(MIN_ALLOWED_CLOSED_ANGLE_RAD, 6);
    UsbOutput.print("..");
    UsbOutput.println(MAX_ALLOWED_CLOSED_ANGLE_RAD, 6);
    return false;
  }

  closedLimitAngleRad = requestedAngleRad;
  targetAngleCanonicalRad = clampValue(
    targetAngleCanonicalRad,
    closedLimitAngleRad,
    OPEN_ANGLE_RAD
  );
  trajectoryTargetAngleRad = clampValue(
    trajectoryTargetAngleRad,
    closedLimitAngleRad,
    OPEN_ANGLE_RAD
  );
  UsbOutput.print(saveConfiguration()
    ? "CLOSED_BOUNDARY_SAVED angle_rad="
    : "ERR_CONFIG_SAVE closed_angle_rad=");
  UsbOutput.println(closedLimitAngleRad, 6);
  return true;
}

void setClosedBoundary(float requestedAngleRad) {
  applyClosedBoundary(requestedAngleRad);
}

void captureClosedBoundary() {
  applyClosedBoundary(currentAngleCanonicalRad);
}

void setTargetSlewRate(float requestedRateRadS) {
  if (!isfinite(requestedRateRadS)
      || requestedRateRadS < MIN_TARGET_SLEW_RAD_S
      || requestedRateRadS > MAX_TARGET_SLEW_RAD_S) {
    UsbOutput.print("ERR_SLEW_RANGE allowed=");
    UsbOutput.print(MIN_TARGET_SLEW_RAD_S, 2);
    UsbOutput.print("..");
    UsbOutput.println(MAX_TARGET_SLEW_RAD_S, 2);
    return;
  }

  targetSlewRateRadS = requestedRateRadS;
  UsbOutput.print(saveConfiguration()
    ? "TARGET_SLEW_SAVED rad_s="
    : "ERR_CONFIG_SAVE slew_rad_s=");
  UsbOutput.println(targetSlewRateRadS, 2);
}

void disableDrive(const char* reason) {
  digitalWrite(ENABLE_PIN, LOW);
  setPwmOff();
  driveEnabled = false;
  motionActive = false;
  uqCommandV = 0.0f;
  UsbOutput.print("DRIVE_DISABLED reason=");
  UsbOutput.println(reason);
}

void enableDriveAtCurrentPosition() {
  targetAngleCanonicalRad = clampValue(
    currentAngleCanonicalRad,
    safeClosedAngle(),
    OPEN_ANGLE_RAD
  );
  trajectoryTargetAngleRad = targetAngleCanonicalRad;
  trajectoryVelocityRadS = 0.0f;
  previousTrajectoryUs = micros();
  uqCommandV = 0.0f;
  setNeutralPwm();
  digitalWrite(ENABLE_PIN, HIGH);
  driveEnabled = true;
  motionActive = false;
  targetReported = false;
  targetStableSinceMs = 0;
  UsbOutput.print("DRIVE_ENABLED_TARGET_LOCKED angle_rad=");
  UsbOutput.println(targetAngleCanonicalRad, 6);
}

void startMoveToCanonicalAngle(float requestedTargetRad, const char* label) {
  if (!driveEnabled) {
    UsbOutput.println("ERR_DRIVE_DISABLED send E1 first");
    return;
  }

  targetAngleCanonicalRad = clampValue(
    requestedTargetRad,
    safeClosedAngle(),
    OPEN_ANGLE_RAD
  );
  // Preserve the running trajectory and velocity when the hand changes target.
  motionActive = true;
  targetReported = false;
  targetStableSinceMs = 0;
  lastProgressAngleRad = currentAngleCanonicalRad;
  lastProgressMs = millis();

  UsbOutput.print("MOVE_START target=");
  UsbOutput.print(label);
  UsbOutput.print(" target_angle_rad=");
  UsbOutput.print(targetAngleCanonicalRad, 6);
  UsbOutput.print(" target_gap_mm=");
  UsbOutput.println(gapFromCanonicalAngle(targetAngleCanonicalRad), 2);
}

void commandGapMm(float requestedGapMm) {
  const float safeGap = clampValue(
    requestedGapMm,
    MIN_COMMAND_GAP_MM,
    maximumGapMm()
  );
  if (safeGap != requestedGapMm) {
    UsbOutput.print("GAP_CLAMPED requested_mm=");
    UsbOutput.print(requestedGapMm, 2);
    UsbOutput.print(" applied_mm=");
    UsbOutput.println(safeGap, 2);
  }
  startMoveToCanonicalAngle(canonicalAngleFromGap(safeGap), "GAP");
}

void updateVelocityTelemetry() {
  const uint32_t nowUs = micros();
  const uint32_t elapsedUs = nowUs - previousVelocityFilterUs;
  if (elapsedUs < 5000) return;

  const float dt = elapsedUs * 1.0e-6f;
  measuredVelocityRadS = SENSOR_DIRECTION * sensor.velocity();
  const float filterAlpha = dt / (0.030f + dt);
  filteredVelocityRadS +=
    filterAlpha * (measuredVelocityRadS - filteredVelocityRadS);
  if (fabsf(filteredVelocityRadS) < 0.003f
      && fabsf(measuredVelocityRadS) < 0.003f) {
    filteredVelocityRadS = 0.0f;
  }
  previousVelocityFilterUs = nowUs;
}

const char* boundaryMonitorStateName(uint8_t state) {
  switch (state) {
    case BOUNDARY_NORMAL: return "NORMAL";
    case BOUNDARY_SOFT_OPEN: return "SOFT_OPEN";
    case BOUNDARY_SOFT_CLOSED: return "SOFT_CLOSED";
    case BOUNDARY_HARD_OPEN: return "HARD_OPEN";
    case BOUNDARY_HARD_CLOSED: return "HARD_CLOSED";
  }
  return "UNKNOWN";
}

uint8_t detectBoundaryMonitorState() {
  if (currentAngleCanonicalRad > OPEN_ANGLE_RAD + OPEN_HARD_MARGIN_RAD) {
    return BOUNDARY_HARD_OPEN;
  }
  if (currentAngleCanonicalRad < ABSOLUTE_CLOSED_HARD_RAD) {
    return BOUNDARY_HARD_CLOSED;
  }
  const float openThreshold = boundaryMonitorState == BOUNDARY_SOFT_OPEN
    ? OPEN_ANGLE_RAD - SOFT_LIMIT_RELEASE_RAD
    : OPEN_ANGLE_RAD + SOFT_LIMIT_TRIGGER_RAD;
  const float closedThreshold = boundaryMonitorState == BOUNDARY_SOFT_CLOSED
    ? closedLimitAngleRad + SOFT_LIMIT_RELEASE_RAD
    : closedLimitAngleRad - SOFT_LIMIT_TRIGGER_RAD;
  if (currentAngleCanonicalRad > openThreshold) {
    return BOUNDARY_SOFT_OPEN;
  }
  if (currentAngleCanonicalRad < closedThreshold) {
    return BOUNDARY_SOFT_CLOSED;
  }
  return BOUNDARY_NORMAL;
}

void updateBoundaryMonitor() {
  const uint8_t detected = detectBoundaryMonitorState();
  if (detected == boundaryMonitorState) return;
  boundaryMonitorState = detected;
  UsbOutput.print("BOUNDARY_MONITOR state=");
  UsbOutput.print(boundaryMonitorStateName(boundaryMonitorState));
  UsbOutput.print(" angle_rad=");
  UsbOutput.print(currentAngleCanonicalRad, 6);
  UsbOutput.println(" action=MONITOR_ONLY");
}

void runDengPositionLoop() {
  const uint32_t nowUs = micros();
  uint32_t elapsedUs = nowUs - previousTrajectoryUs;
  if (elapsedUs == 0) elapsedUs = 1;
  float dt = elapsedUs * 1.0e-6f;
  if (dt > 0.010f) dt = 0.002f;
  previousTrajectoryUs = nowUs;

  // Only an explicit position command advances this trajectory. A physical
  // displacement, including crossing either soft boundary, must never rewrite
  // it. After release, the unchanged trajectory therefore produces an immediate
  // pure-P restoring voltage toward the original position.
  gm_step(&trajectoryTargetAngleRad, &trajectoryVelocityRadS,
    targetAngleCanonicalRad, closedLimitAngleRad, OPEN_ANGLE_RAD,
    targetSlewRateRadS, dt);

  const float errorRad =
    trajectoryTargetAngleRad - currentAngleCanonicalRad;
  const float errorDeg = errorRad * RAD_TO_DEG_F;
  uqCommandV = clampValue(
    POSITION_KP_V_PER_DEG * errorDeg,
    -UQ_LIMIT_V,
    UQ_LIMIT_V
  );
  setTorqueSVPWM(uqCommandV, electricalAngle());
}

void updateArrivalAndProgress() {
  if (!motionActive) return;

  const unsigned long now = millis();
  const float errorRad =
    targetAngleCanonicalRad - currentAngleCanonicalRad;
  if (fabsf(currentAngleCanonicalRad - lastProgressAngleRad)
      >= PROGRESS_STEP_RAD) {
    lastProgressAngleRad = currentAngleCanonicalRad;
    lastProgressMs = now;
  }

  if (fabsf(errorRad) <= TARGET_TOLERANCE_RAD) {
    if (targetStableSinceMs == 0) targetStableSinceMs = now;
    if (!targetReported && now - targetStableSinceMs >= TARGET_STABLE_MS) {
      targetReported = true;
      motionActive = false;
      UsbOutput.println("TARGET_REACHED_TARGET_REMAINS_LOCKED");
    }
  } else {
    targetStableSinceMs = 0;
  }

  if (fabsf(errorRad) > TARGET_TOLERANCE_RAD
      && now - lastProgressMs > NO_PROGRESS_TIME_MS) {
    motionActive = false;
    UsbOutput.println("NO_PROGRESS_TARGET_REMAINS_LOCKED");
  }
}

// Bench gripper only: bounded commands, immutable feedback, no arm motion forwarding.
constexpr int GRIP_CAN_TX_PIN = 17;
constexpr int GRIP_CAN_RX_PIN = 35;
bool gripCanReady = false;
uint32_t gripCanRequests = 0, gripCanDuplicates = 0, gripCanInvalid = 0;
uint32_t gripCanTxFull = 0, gripCanBusOff = 0, gripCanLastSeq = 0;
uint16_t gripCanBoot = 0;
uint8_t gripCanSnapshot[GC_BYTES];
unsigned gripCanPart = GC_PARTS;
uint32_t gripCanLastTxMs = 0;
bool gripCanCached = false;
uint32_t gripSession = 0, gripSensorMs = 0, gripLeaseMs = 0;
bool gripOwned = false, gripAckPending = false;
uint8_t gripFault = 0, gripAck[8];
GP_Cache gripCommands = {};
GP_Cache radioCommands[2] = {};
int gripOwner = -1;
GP_Assembly gripAssembly = {};
void handleGripperControl(const uint8_t *raw, int ingress = 0) {
  GP_Command command;
  if (!gp_decode(raw, &command)) { ++gripCanInvalid; return; }
  uint8_t result;
  if (command.session != gripSession) result = GP_SESSION;
  else if (command.origin != ingress) result = GP_OWNER;
  else if (!gp_duplicate(command.origin == 0 ? &gripCommands : &radioCommands[command.origin-1], &command, raw, &result)) {
    const bool fresh = gripSensorMs && millis()-gripSensorMs <= 10U
      && !sensor.errorCode() && isfinite(currentAngleCanonicalRad)
      && currentAngleCanonicalRad >= safeClosedAngle() && currentAngleCanonicalRad <= OPEN_ANGLE_RAD;
    result = gp_gate(&command, gripSession, millis(), gripOwned?gripOwner:-1,
      driveEnabled, fresh, static_cast<uint16_t>(lroundf(maximumGapMm()*100)));
    if (result == GP_OK) {
      switch(command.op) {
        case GP_ENABLE:
          if (!driveEnabled) enableDriveAtCurrentPosition();
          gripOwned = true; gripOwner = command.origin; gripFault = 0; break;
        case GP_DISABLE:
          disableDrive("CAN_STOP"); gripOwned = false; break;
        case GP_GAP: commandGapMm(command.arg/100.0f); break;
        case GP_RATE: targetSlewRateRadS = command.arg/1000.0f; break;
        case GP_OPEN: startMoveToCanonicalAngle(OPEN_ANGLE_RAD,"CAN_OPEN"); break;
        case GP_CLOSE: startMoveToCanonicalAngle(safeClosedAngle(),"CAN_CLOSE"); break;
        case GP_PING: break;
      }
      if (gripOwned) gripLeaseMs = millis();
    }
    gp_cache(command.origin == 0 ? &gripCommands : &radioCommands[command.origin-1], &command, raw, result);
  }
  gp_ack(gripAck, &command, result); gripAckPending = ingress == 0;
}
void initGripperCAN() {
  twai_general_config_t general = TWAI_GENERAL_CONFIG_DEFAULT(
    static_cast<gpio_num_t>(GRIP_CAN_TX_PIN), static_cast<gpio_num_t>(GRIP_CAN_RX_PIN), TWAI_MODE_NORMAL);
  general.tx_queue_len = 8; general.rx_queue_len = 16;
  general.alerts_enabled = TWAI_ALERT_BUS_OFF | TWAI_ALERT_BUS_RECOVERED;
  twai_timing_config_t timing = TWAI_TIMING_CONFIG_500KBITS();
  twai_filter_config_t filter = { GC_REQUEST_ID << 3, (0xffUL << 3) | 7U, true };
  esp_err_t installed = twai_driver_install(&general, &timing, &filter);
  gripCanReady = installed == ESP_OK && twai_start() == ESP_OK;
  gripCanBoot = static_cast<uint16_t>(esp_random());
  if (!gripCanBoot) gripCanBoot = 1;
  gripSession = esp_random(); if (!gripSession) gripSession = 1;
}
void pollGripperCAN() {
  if (gripOwned && driveEnabled && millis()-gripLeaseMs > 750U) {
    disableDrive("CAN_LEASE_TIMEOUT"); gripOwned = false; gripFault = 1;
  }
  if (gripOwned && (!driveEnabled || sensor.errorCode())) {
    if (driveEnabled) disableDrive("CAN_SENSOR_FAULT");
    gripOwned = false; gripFault = 2;
  }
  if (!gripCanReady) return;
  uint32_t alerts = 0;
  if (twai_read_alerts(&alerts, 0) == ESP_OK) {
    if (alerts & TWAI_ALERT_BUS_OFF) {
      ++gripCanBusOff; gripCanPart = GC_PARTS;
      twai_initiate_recovery();
    }
    if (alerts & TWAI_ALERT_BUS_RECOVERED) twai_start();
  }
  twai_message_t frame;
  for (unsigned n=0; n<4 && twai_receive(&frame,0)==ESP_OK; ++n) {
    if (!frame.extd || frame.rtr || frame.data_length_code != 8) {
      ++gripCanInvalid; continue;
    }
    if (frame.identifier >= GC_CONTROL_ID && frame.identifier < GC_CONTROL_ID+3U) {
      if (gp_feed(&gripAssembly, frame.identifier-GC_CONTROL_ID, frame.data, millis()))
        handleGripperControl(gripAssembly.raw);
      continue;
    }
    // Ignore other node-7 replies observed on this shared bus.
    if (frame.identifier != GC_REQUEST_ID) continue;
    if (!gc_request_valid(frame.data)) {
      ++gripCanInvalid; continue;
    }
    const uint32_t sequence = gc_get32(frame.data+2);
    if (gripCanCached && sequence == gripCanLastSeq) ++gripCanDuplicates;
    else {
      ++gripCanRequests; gripCanLastSeq = sequence; gripCanCached = true;
      gc_put32(gripCanSnapshot, static_cast<uint32_t>(static_cast<int32_t>(lroundf(currentAngleCanonicalRad*1000))));
      gc_put32(gripCanSnapshot+4, static_cast<uint32_t>(static_cast<int32_t>(lroundf(targetAngleCanonicalRad*1000))));
      gc_put16(gripCanSnapshot+8, static_cast<uint16_t>(lroundf(gapFromCanonicalAngle(currentAngleCanonicalRad)*100)));
      gripCanSnapshot[10] = (driveEnabled?1:0) | (motionActive?2:0) | (boundaryMonitorState<<2) | (targetReported?32:0);
      gripCanSnapshot[11] = sensor.errorCode();
      gc_put32(gripCanSnapshot+12, controlMaxPeriodUs);
      gc_put16(gripCanSnapshot+16, gripCanBoot);
      gc_put16(gripCanSnapshot+18, static_cast<uint16_t>(lroundf(maximumGapMm()*100)));
      gc_put32(gripCanSnapshot+20, gripSession); gc_put32(gripCanSnapshot+24,millis());
      gc_put32(gripCanSnapshot+28,gripCommands.valid?gripCommands.seq:0);
      gripCanSnapshot[32]=gripCommands.valid?gripCommands.result:0;
      gripCanSnapshot[33]=gripFault; gc_seal(sequence,gripCanSnapshot);
    }
    gripCanPart = 0;
  }
  if (gripAckPending) {
    twai_message_t ack = {}; ack.extd=1; ack.identifier=GC_ACK_ID; ack.data_length_code=8;
    memcpy(ack.data,gripAck,8);
    if (twai_transmit(&ack,0)==ESP_OK) gripAckPending=false;
    else ++gripCanTxFull;
    return;
  }
  // Bound work per loop and space fragments for STM32's three-slot RX FIFO.
  if (gripCanPart < GC_PARTS && millis()-gripCanLastTxMs >= 2U) {
    twai_message_t reply = {};
    reply.extd=1;reply.identifier=GC_REPLY_ID+gripCanPart;reply.data_length_code=8;
    gc_put32(reply.data,gripCanLastSeq);memcpy(reply.data+4,gripCanSnapshot+4*gripCanPart,4);
    if (twai_transmit(&reply,0)==ESP_OK) { ++gripCanPart;gripCanLastTxMs=millis(); }
    else ++gripCanTxFull;
  }
}

void pollFinalRadio() {
  static uint32_t lastSnapshot = 0;
  if (millis()-lastSnapshot >= 20U) {
    uint8_t status[40] = {};
    gc_put32(status, gripSession); gc_put32(status+4,millis());
    gc_put16(status+8,(uint16_t)lroundf(gapFromCanonicalAngle(currentAngleCanonicalRad)*100));
    gc_put16(status+10,7463);
    status[12]=driveEnabled; status[13]=motionActive; status[14]=sensor.errorCode();
    status[15]=gripOwned ? (uint8_t)gripOwner : 255;
    gc_put32(status+16,gripSensorMs ? millis()-gripSensorMs : 0xffffffffUL);
    gc_put32(status+20,controlMaxPeriodUs);
    gc_put16(status+24,(uint16_t)lroundf(gapFromCanonicalAngle(targetAngleCanonicalRad)*100));
    status[26]=targetReported; status[27]=gripFault;
    gc_put32(status+28,radioCommands[0].valid?radioCommands[0].seq:0); gc_put32(status+32,radioCommands[1].valid?radioCommands[1].seq:0);
    gc_put32(status+36,finalRadioDropped);
    finalRadioPublish(status);
    lastSnapshot=millis();
  }
  FinalRadioCommand request;
  if (finalRadioTake(request)) {
    GP_Command command;
    uint8_t ack[8] = {};
    if (gp_decode(request.raw,&command)) {
      handleGripperControl(request.raw,request.origin);
      memcpy(ack,gripAck,8);
    } else {
      gc_put32(ack,request.seq); ack[4]=GP_RANGE;
    }
    finalRadioComplete(request,ack);
  }
}

void printStatus(BufferedTelemetry &port = UsbOutput) {
  if (port.freeSpace() < 1200) return;
  const float errorRad =
    targetAngleCanonicalRad - currentAngleCanonicalRad;
  // Keep the status version prefix required by the deployed STM32 parser.
  port.print("version=v2.2.3-DENGFOC-STM32-RO enabled=");
  port.print(driveEnabled ? 1 : 0);
  port.print(" controller=DENG_V01_P_ADJ");
  port.print(" kp_v_per_deg=");
  port.print(POSITION_KP_V_PER_DEG, 3);
  port.print(" zero_electric_angle=");
  port.print(ZERO_ELECTRIC_ANGLE_RAD, 6);
  port.print(" sensor_direction=");
  port.print(SENSOR_DIRECTION);
  port.print(" trajectory_max_rad_s="); port.print(GM_MAX_SPEED,2);
  port.print(" trajectory_accel_rad_s2="); port.print(GM_ACCELERATION,2);
  port.print(" trajectory_velocity_rad_s="); port.print(trajectoryVelocityRadS,3);
  port.print(" endpoint_band_mm="); port.print(GM_END_BAND_MM,1);
  port.print(" slew_rad_s=");
  port.print(targetSlewRateRadS, 2);
  port.print(" closed_limit_rad=");
  port.print(closedLimitAngleRad, 6);
  port.print(" angle_rad=");
  port.print(currentAngleCanonicalRad, 6);
  port.print(" target_rad=");
  port.print(targetAngleCanonicalRad, 6);
  port.print(" trajectory_target_rad=");
  port.print(trajectoryTargetAngleRad, 6);
  port.print(" error_rad=");
  port.print(errorRad, 6);
  port.print(" gap_mm=");
  port.print(gapFromCanonicalAngle(currentAngleCanonicalRad), 2);
  port.print(" velocity_rad_s=");
  port.print(filteredVelocityRadS, 6);
  port.print(" uq_cmd_v=");
  port.print(uqCommandV, 3);
  port.print(" uq_limit_v=");
  port.print(UQ_LIMIT_V, 1);
  port.print(" target_tolerance_rad=");
  port.print(TARGET_TOLERANCE_RAD, 3);
  port.print(" boundary_monitor=");
  port.print(boundaryMonitorStateName(boundaryMonitorState));
  port.print(" boundary_action=MONITOR_ONLY");
  port.print(" outside_open_soft=");
  port.print(currentAngleCanonicalRad > OPEN_ANGLE_RAD ? 1 : 0);
  port.print(" outside_closed_soft=");
  port.print(currentAngleCanonicalRad < safeClosedAngle() ? 1 : 0);
  port.print(" i2c_error=");
  port.print(sensor.errorCode());
  port.print(" firmware_revision=v3.0-FINAL-INTERFACES pwm_bits=10");
  port.print(" en_gpio="); port.print(digitalRead(ENABLE_PIN));
  port.print(" pwm_a="); port.print(ledcRead(PWM_A_PIN));
  port.print(" pwm_b="); port.print(ledcRead(PWM_B_PIN));
  port.print(" pwm_c="); port.print(ledcRead(PWM_C_PIN));
  port.print(" pwm_hz="); port.print(ledcReadFreq(PWM_A_PIN));
  port.print(" control_max_period_us=");
  port.print(controlMaxPeriodUs);
  port.print(" telemetry_dropped=");
  port.print(UsbOutput.dropped + ArmOutput.dropped);
  twai_status_info_t canState = {};
  bool canStatusOK = gripCanReady && twai_get_status_info(&canState) == ESP_OK;
  port.print(" can_ready="); port.print(gripCanReady?1:0);
  port.print(" can_state="); port.print(canStatusOK?static_cast<int>(canState.state):-1);
  port.print(" can_requests=");port.print(gripCanRequests);
  port.print(" can_duplicates=");port.print(gripCanDuplicates);
  port.print(" can_invalid=");port.print(gripCanInvalid);
  port.print(" can_tx_full=");port.print(gripCanTxFull);
  port.print(" can_bus_off=");port.print(gripCanBusOff);
  port.print(" can_rx_missed=");port.print(canState.rx_missed_count);
  port.print(" can_tx_failed=");port.println(canState.tx_failed_count);
}

void printMagnetDiagnostics() {
  if (driveEnabled) {
    UsbOutput.println("ERR_DISABLE_DRIVE_BEFORE_MAGNET_DIAGNOSTICS");
    return;
  }
  const uint8_t registers[] = {0x0B, 0x1A, 0x1B, 0x1C};
  uint8_t values[4];
  for (size_t i = 0; i < 4; ++i) {
    Wire.beginTransmission(0x36); Wire.write(registers[i]);
    const uint8_t error = Wire.endTransmission(false);
    if (error || Wire.requestFrom(0x36, static_cast<uint8_t>(1)) != 1) {
      UsbOutput.print("MAGNET_DIAG_I2C_ERROR code="); UsbOutput.println(error);
      return;
    }
    values[i] = Wire.read();
  }
  UsbOutput.print("MAGNET_DIAG detected="); UsbOutput.print((values[0] & 0x20) != 0);
  UsbOutput.print(" weak="); UsbOutput.print((values[0] & 0x10) != 0);
  UsbOutput.print(" strong="); UsbOutput.print((values[0] & 0x08) != 0);
  UsbOutput.print(" agc="); UsbOutput.print(values[1]);
  UsbOutput.print(" magnitude="); UsbOutput.println(((values[2] & 0x0F) << 8) | values[3]);
}

void printHelp() {
  UsbOutput.println("Commands (send with newline):");
  UsbOutput.println("  E1       enable and lock the current safe position");
  UsbOutput.println("  E0 or X  disable motor immediately (no active hold)");
  UsbOutput.println("  O        move to safe-open endpoint");
  UsbOutput.println("  C        move to the saved closed boundary");
  UsbOutput.println("  G<mm>    command gap from captured closed boundary");
  UsbOutput.println("  R<x>     set/save target speed 0.10..2.00 rad/s");
  UsbOutput.println("  L<rad>   set/save closed boundary; drive must be disabled");
  UsbOutput.println("  B        capture/save current position as closed boundary; disabled only");
  UsbOutput.println("  S        print adjustable DengFOC P-loop status");
  UsbOutput.println("  M        read magnet diagnostics; disabled only");
  UsbOutput.println("  ?        print this help");
}

void executeCommand(char* line) {
  while (*line == ' ' || *line == '\t') ++line;
  const char command = line[0];
  const float value = atof(line + 1);
  if (strchr("EeXxOoCcGgRrLlBb", command) && command) {
    if (gripOwned) disableDrive("USB_TAKEOVER");
    gripOwned = false;
    gripSession = esp_random(); if (!gripSession) gripSession=1;
    gripCommands = {}; radioCommands[0] = {}; radioCommands[1] = {}; gripOwner = -1; gripAssembly = {}; gripCanCached = false;
  }

  if (command == 'E' || command == 'e') {
    if (value > 0.5f) enableDriveAtCurrentPosition();
    else disableDrive("USER_E0");
  } else if (command == 'X' || command == 'x') {
    disableDrive("USER_X");
  } else if (command == 'O' || command == 'o') {
    startMoveToCanonicalAngle(OPEN_ANGLE_RAD, "OPEN");
  } else if (command == 'C' || command == 'c') {
    startMoveToCanonicalAngle(safeClosedAngle(), "CLOSE_SAFE");
  } else if (command == 'G' || command == 'g') {
    commandGapMm(value);
  } else if (command == 'R' || command == 'r') {
    setTargetSlewRate(value);
  } else if (command == 'L' || command == 'l') {
    setClosedBoundary(value);
  } else if (command == 'B' || command == 'b') {
    captureClosedBoundary();
  } else if (command == 'S' || command == 's') {
    printStatus();
  } else if (command == 'M' || command == 'm') {
    printMagnetDiagnostics();
  } else if (command == '?') {
    printHelp();
  } else {
    UsbOutput.println("ERR_UNKNOWN_COMMAND");
  }
}

void readSerialCommands() {
  for (unsigned int n = 0; n < 64U && Serial.available() > 0; ++n) {
    const char value = Serial.read();
    if (value == '\n' || value == '\r') {
      if (serialLength > 0) {
        serialLine[serialLength] = '\0';
        executeCommand(serialLine);
        serialLength = 0;
      }
    } else if (serialLength < sizeof(serialLine) - 1) {
      serialLine[serialLength++] = value;
    }
  }
}

void setup() {
  pinMode(ENABLE_PIN, OUTPUT);
  digitalWrite(ENABLE_PIN, LOW);
  Serial.begin(115200);
  ArmLink.begin(115200, SERIAL_8N1, ARM_RX_PIN, ARM_TX_PIN);
  delay(800);
  UsbOutput.println();
  UsbOutput.println("ESP32 GRIPPER DENGFOC P FINAL v3.0-FINAL-INTERFACES");
  UsbOutput.println("Controller: DengFOC V0.1 pure P + adjustable target ramp");
  UsbOutput.println("BUS=12.0V Kp=0.133V/deg Uq_limit=6.0V");
  UsbOutput.println("Boundary mode: monitor-only, no boundary disable or latch");

  initGripperCAN();
  Wire.begin(AS5600_SDA_PIN, AS5600_SCL_PIN, 100000UL);
  Wire.setTimeOut(10);
  if (!sensor.begin(&Wire)) {
    UsbOutput.println("FATAL_AS5600_INIT");
    while (true) { UsbOutput.drain(); delay(1); }
  }
  if (!initializePwm()) {
    UsbOutput.println("FATAL_PWM_INIT");
    while (true) { UsbOutput.drain(); delay(1); }
  }

  loadConfiguration();

  if (!sensor.update()) {
    UsbOutput.println("FATAL_AS5600_FIRST_READ");
    while (true) { UsbOutput.drain(); delay(1); }
  }

  const float rawDirectedAngle = SENSOR_DIRECTION * sensor.continuousAngle();
  currentAngleCanonicalRad = canonicalizeAngle(rawDirectedAngle);
  branchOffsetRad = rawDirectedAngle - currentAngleCanonicalRad;
  targetAngleCanonicalRad = clampValue(
    currentAngleCanonicalRad,
    safeClosedAngle(),
    OPEN_ANGLE_RAD
  );
  trajectoryTargetAngleRad = targetAngleCanonicalRad;
  trajectoryVelocityRadS = 0.0f;
  previousVelocityFilterUs = micros();
  previousTrajectoryUs = previousVelocityFilterUs;

  UsbOutput.print("CALIBRATION_FIXED zero_electric_angle=");
  UsbOutput.print(ZERO_ELECTRIC_ANGLE_RAD, 6);
  UsbOutput.print(" sensor_direction=");
  UsbOutput.println(SENSOR_DIRECTION);
  UsbOutput.print("CONFIG closed_limit_rad=");
  UsbOutput.print(closedLimitAngleRad, 6);
  UsbOutput.print(" slew_rad_s=");
  UsbOutput.println(targetSlewRateRadS, 2);
  UsbOutput.print("STARTUP_POSITION angle_rad=");
  UsbOutput.println(currentAngleCanonicalRad, 6);

  updateBoundaryMonitor();
  disableDrive("STARTUP_COMM_ONLY");
  UsbOutput.println("STARTUP_DISABLED no automatic opening");
  printHelp();
  finalRadioBegin();
}

void readArmCommands() {
  static char input[16];
  static size_t used = 0;
  static bool discard = false;
  static uint32_t lastByteMs = 0;
  if ((used || discard) && millis() - lastByteMs > 1000U) {
    used = 0; discard = false;
  }
  for (unsigned int n = 0; n < 64U && ArmLink.available(); ++n) {
    char c = ArmLink.read(); lastByteMs = millis();
    if (c == '\r') continue;
    if (c == '\n') {
      input[used] = 0;
      if (!discard && strcmp(input, "S") == 0) printStatus(ArmOutput);
      else if (used || discard) ArmOutput.println("REJECT status-only UART");
      used = 0; discard = false;
    } else if (!discard) {
      if (c < 32 || c > 126 || used >= sizeof(input)-1) discard = true;
      else input[used++] = c;
    }
  }
}

void loop() {
  UsbOutput.drain();
  ArmOutput.drain();
  readArmCommands();
  pollGripperCAN();
  pollFinalRadio();
  static uint32_t lastSensorPollUs = 0;
  const uint32_t nowSensorUs = micros();
  if (nowSensorUs - lastSensorPollUs < 2000U) {
    readSerialCommands();
    delayMicroseconds(100);
    return;
  }
  if (lastSensorPollUs != 0 && nowSensorUs - lastSensorPollUs > controlMaxPeriodUs)
    controlMaxPeriodUs = nowSensorUs - lastSensorPollUs;
  lastSensorPollUs = nowSensorUs;
  if (!sensor.update()) {
    if (sensor.errorCount() >= MAX_CONSECUTIVE_I2C_ERRORS && driveEnabled) {
      UsbOutput.print("FAULT_AS5600 code="); UsbOutput.print(sensor.errorCode());
      UsbOutput.print(" consecutive="); UsbOutput.println(sensor.errorCount());
      disableDrive("AS5600_I2C_FAULT");
    }
    readSerialCommands();
    return;
  }

  gripSensorMs = millis();
  currentAngleCanonicalRad = canonicalAngleFromSensor();
  updateVelocityTelemetry();
  updateBoundaryMonitor();

  // Mechanical monitor edition: every boundary state is telemetry-only.
  // The clamped target and pure-P restoring action remain active at all times.
  if (driveEnabled) {
    runDengPositionLoop();
    updateArrivalAndProgress();
  }

  readSerialCommands();
}
