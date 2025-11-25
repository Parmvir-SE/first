// === ESP32 Crop Recommender — Top‑3 + 3 Choice Buttons + Feedback Learning ===
// Sensors: DHT11 (GPIO4), Soil AO (ADC GPIO34), EC analog (ADC GPIO35)
// Capture Button: GPIO18 (INPUT_PULLUP). Press to start 60s capture.
// Choice Buttons: GPIO21, GPIO22, GPIO23 (INPUT_PULLUP). Press to select Top‑3 crop.
//
// Pipeline:
// - 60s synchronized capture -> Vapor Pressure Deficit (VPD), Soil Moisture %, EC@25°C
// - Normalize to [0..1] (internally); detect season (NTP month if Wi‑Fi set; fallback to __DATE__)
// - Recommend Top‑3 with clear reasons
// - Show a farmer‑friendly summary (Temperature, Moisture %, EC25 + EC%, RH %, VPD kPa), then Top‑3
// - Wait for user choice (3 buttons); log to /choices.csv
// - Feedback learning: update per‑crop bias; save to /prefs.csv; bias nudges future scores

#include <Arduino.h>
#include "DHT.h"
#include <WiFi.h>
#include <time.h>
#include <math.h>
#include <string.h>
#include <FS.h>
#include <SPIFFS.h>

// -------- Wi‑Fi / NTP (empty → works offline with __DATE__) --------
#define WIFI_SSID "" // leave empty -> fallback to __DATE__
#define WIFI_PASS ""
#define TZ_INFO   "IST-5:30"
#define NTP_SERVER "pool.ntp.org"

// -------- Pins --------
#define DHTPIN 4      // DHT11 data
#define DHTTYPE DHT11
#define SOIL_AO_PIN 34 // Soil moisture analog (ADC1)
#define EC_PIN 35      // EC sense analog (ADC1)
#define BUTTON_CAPTURE_PIN 18 // Capture push button (INPUT_PULLUP)
#define BUTTON_CHOOSE_1_PIN 21
#define BUTTON_CHOOSE_2_PIN 22
#define BUTTON_CHOOSE_3_PIN 23

// -------- ADC / Capture --------
const int ADC_BITS = 12; // 0..4095
const unsigned long CAPTURE_MS = 60UL * 1000UL; // 1 minute
const uint16_t SAMPLE_PERIOD_MS = 200; // Soil/EC sample period
const uint16_t DHT_MIN_PERIOD_MS = 2000; // DHT11 minimum interval

// -------- Calibration constants (pre‑set) --------
const int   AIR_VALUE   = 4095;     // Dry reference (air)
const int   WATER_VALUE = 1241;     // Wet reference (saturated)
const float VCC         = 3.3f;     // ESP32 analog ref
const float RKNOWN      = 10000.0f; // 10kΩ divider resistor (at VCC side)
const float ALPHA       = 0.022f;   // EC temp coefficient (~2.2%/°C)
const float M_MAP       = 12008.580f; // EC25(dS/m) = M_MAP * invR25 + B_MAP
const float B_MAP       = -0.551378f; // intercept

// DHT object
DHT dht(DHTPIN, DHTTYPE);

// -------- Data structures --------
struct Snapshot {
  float temp_c;      // °C
  float humidity;    // % (Relative Humidity)
  float vpd_kpa;     // kPa (Vapor Pressure Deficit)
  float soil_pct;    // % (Soil Moisture)
  float ec25_ds_m;   // dS/m @25°C (Electrical Conductivity)
  int   season;      // 0,1,2 (0=Rabi, 1=Zaid, 2=Kharif)
};

struct NormFeatures {
  float temp_norm; // 0..1
  float hum_norm;  // 0..1
  float vpd_norm;  // 0..1
  float moist_norm;// 0..1
  float ec_norm;   // 0..1
  int   season;    // 0,1,2
};

struct CropResult {
  const char* name;
  float score;
  char  reason[128];
};

struct CropProfile {
  const char* name;
  bool allowRabi;
  bool allowZaid;
  bool allowKharif;
  float t_lo, t_hi; // temp band (norm)
  float h_lo, h_hi; // humidity band (norm)
  float v_lo, v_hi; // VPD band (norm)
  float m_lo, m_hi; // moisture band (norm)
  float ec_max;     // salinity tolerance (norm)
};

struct CaptureOut {
  Snapshot    snap;
  NormFeatures nf;
  bool        ok;
};

// -------- Crop table --------
static const CropProfile CROPS[] = {
  {"Rice (Paddy)", false, false, true,  0.55f,0.85f, 0.60f,0.95f, 0.00f,0.30f, 0.70f,1.00f, 0.30f},
  {"Wheat",         true, false, false, 0.30f,0.60f, 0.40f,0.80f, 0.15f,0.55f, 0.40f,0.65f, 0.25f},
  {"Maize",         false,false, true,  0.45f,0.80f, 0.40f,0.80f, 0.25f,0.65f, 0.45f,0.70f, 0.30f},
  {"Bajra",         false,false, true,  0.50f,0.85f, 0.35f,0.70f, 0.45f,0.90f, 0.30f,0.55f, 0.40f},
  {"Mustard",       true, false, false, 0.35f,0.65f, 0.40f,0.80f, 0.30f,0.65f, 0.35f,0.55f, 0.30f},
  {"Chickpea",      true, false, false, 0.30f,0.55f, 0.35f,0.75f, 0.25f,0.60f, 0.30f,0.50f, 0.25f},
  {"Cotton",        false,false, true,  0.55f,0.90f, 0.40f,0.75f, 0.40f,0.80f, 0.40f,0.65f, 0.30f},
  {"Okra (Bhindi)", false,false, true,  0.50f,0.85f, 0.45f,0.85f, 0.25f,0.60f, 0.45f,0.70f, 0.30f},
};
static const int N_CROPS = sizeof(CROPS)/sizeof(CROPS[0]);

// -------- Weights --------
static const float W_TEMP  = 0.22f;
static const float W_HUM   = 0.18f;
static const float W_VPD   = 0.22f;
static const float W_MOIST = 0.28f;
static const float W_EC    = 0.10f;

// -------- Feedback learning --------
static float gBias[N_CROPS];      // per‑crop personalization bias
static const float PREF_WEIGHT = 0.15f; // how much bias affects score
static const float LEARN_RATE  = 0.10f; // chosen crop bias step
static const float DECAY_RATE  = 0.02f; // non‑chosen decay step
static const float BIAS_CLIP   = 0.50f; // bias bounds [‑0.5..+0.5]

// -------- Files --------
static const char* CHOICES_CSV = "/choices.csv";
static const char* PREFS_CSV   = "/prefs.csv";

// -------- Prototypes --------
float clamp01(float x);
float fmax1(float a, float b);
float sat_kPa(float tC);
float vpd_kpa_from(float tC, float rh);
float adcToVoltage(int adc);
int   readADCavg(uint8_t pin, uint8_t ns);
float voltageToRsoil(float vout);
float invR_tempComp(float invR, float tC);
float ec25_from_invR25(float invR25);
float moisture_pct_from_raw(int raw);
int   season_from_month(int month);
float bandScore(float x, float lo, float hi, float softness);
void  addReason(char* buf, size_t cap, const char* txt);
CropResult evalCrop(CropProfile c, NormFeatures f);
void  recommendTop3(NormFeatures f, CropResult out[3]);
int   get_current_month_via_ntp();
String now_iso8601();
bool  waitForButtonPress(uint8_t pin);
CaptureOut captureOneMinute();
bool  initStorage();
bool  ensureChoicesHeader();
bool  appendChoiceCSV(const char* tsISO, Snapshot snap, NormFeatures nf, const CropResult top3[3], int chosenIdx);
int   waitForCropChoice(const CropResult top3[3]);

// Feedback prototypes
int   cropIndexByName(const char* name);
float getBiasForCrop(const char* name);
void  clampBiases();
void  applyFeedbackLearningGlobal(int globalIdx);   // <-- renamed & clarified
bool  savePrefsCSV();
bool  loadPrefsCSV();
bool  ensurePrefsCSV();

// Friendly farmer output
void  printFriendlySummaryAndTop3(const CaptureOut& out, const CropResult top3[3]);

// -------- Helpers --------
float clamp01(float x){ if (x<0) return 0; if (x>1) return 1; return x; }
float fmax1(float a, float b){ return (a > b) ? a : b; }
float sat_kPa(float tC) {
  return 0.6108f * expf((17.27f * tC) / (tC + 237.3f));
}
float vpd_kpa_from(float tC, float rh) {
  float es = sat_kPa(tC);
  float ea = (rh / 100.0f) * es;
  float vpd = es - ea;
  if (vpd < 0) vpd = 0.0f;
  return vpd;
}
float adcToVoltage(int adc) { return (adc / 4095.0f) * VCC; }
int readADCavg(uint8_t pin, uint8_t ns) {
  long acc = 0;
  for (uint8_t i=0; i<ns; ++i) { acc += analogRead(pin); delay(5); }
  int div = (ns > 0) ? ns : 1;
  return (int)(acc / div);
}

// NOTE: resistor is at VCC side (your hardware topology).
float voltageToRsoil(float vout) {
  float denom = VCC - vout;
  if (denom <= 0.0001f) return 1e12f;
  return (vout * RKNOWN) / denom;
}
float invR_tempComp(float invR, float tC) {
  return invR / (1.0f + ALPHA * (tC - 25.0f));
}
float ec25_from_invR25(float invR25) {
  float ec = M_MAP * invR25 + B_MAP;
  if (ec < 0) ec = 0;
  return ec;
}
float moisture_pct_from_raw(int raw) {
  if (AIR_VALUE == WATER_VALUE) return NAN;
  float pct = 100.0f * (float)(AIR_VALUE - raw) / (float)(AIR_VALUE - WATER_VALUE);
  if (pct < 0) pct = 0; if (pct > 100) pct = 100;
  return pct;
}

// -------- Season mapping --------
int season_from_month(int month) {
  if (month >= 10) return 0; // Rabi
  if (month <= 3)  return 0; // Rabi
  if (month >= 4) {
    if (month <= 6) return 1; // Zaid
  }
  return 2; // Kharif
}

// -------- Recommender --------
float bandScore(float x, float lo, float hi, float softness) {
  if (lo > hi) return 0.f;
  x = clamp01(x);
  if (x >= lo && x <= hi) return 1.f;
  float d = (x < lo) ? (lo - x) : (x - hi);
  float s = 1.f - d/softness;
  if (s < 0.f) s = 0.f;
  return s;
}

void addReason(char* buf, size_t cap, const char* txt) {
  if (!buf || cap < 2 || !txt) return;
  size_t cur = strlen(buf);
  if (cur > 0) {
    strncat(buf, "; ", cap - cur - 1);
  }
  strncat(buf, txt, cap - strlen(buf) - 1);
}

// Feedback helpers
int cropIndexByName(const char* name) {
  for (int i=0; i<N_CROPS; ++i) {
    if (strcmp(name, CROPS[i].name) == 0) return i;
  }
  return -1;
}
float getBiasForCrop(const char* name) {
  int idx = cropIndexByName(name);
  if (idx < 0) return 0.0f;
  return gBias[idx];
}
void clampBiases() {
  for (int i=0; i<N_CROPS; ++i) {
    if (gBias[i] >  BIAS_CLIP) gBias[i] =  BIAS_CLIP;
    if (gBias[i] < -BIAS_CLIP) gBias[i] = -BIAS_CLIP;
  }
}

void applyFeedbackLearningGlobal(int globalIdx) {
  // Positive reinforcement for chosen crop
  float curr = gBias[globalIdx];
  gBias[globalIdx] = curr + LEARN_RATE * (1.0f - curr); // toward +1

  // Gentle decay for others
  for (int j=0; j<N_CROPS; ++j) {
    if (j == globalIdx) continue;
    gBias[j] = gBias[j] - DECAY_RATE * (gBias[j]); // toward 0
  }

  clampBiases();

  // Print biases for transparency
  Serial.println(F("\n[FEEDBACK] Updated per-crop biases:"));
  for (int i=0; i<N_CROPS; ++i) {
    Serial.print(" - "); Serial.print(CROPS[i].name);
    Serial.print(": "); Serial.println(gBias[i], 3);
  }
}

CropResult evalCrop(CropProfile c, NormFeatures f) {
  CropResult r;
  r.name = c.name;
  r.score = 0.f;
  r.reason[0] = '\0';

  float sT = bandScore(f.temp_norm,  c.t_lo, c.t_hi, 0.15f);
  float sH = bandScore(f.hum_norm,   c.h_lo, c.h_hi, 0.15f);
  float sV = bandScore(f.vpd_norm,   c.v_lo, c.v_hi, 0.15f);
  float sM = bandScore(f.moist_norm, c.m_lo, c.m_hi, 0.15f);
  float sE = (f.ec_norm <= c.ec_max) ? 1.0f : fmax1(0.0f, 1.0f - (f.ec_norm - c.ec_max)/0.15f);

  float base = W_TEMP*sT + W_HUM*sH + W_VPD*sV + W_MOIST*sM + W_EC*sE;

  bool seasonOK = false;
  if (f.season == 0) seasonOK = c.allowRabi;
  else if (f.season == 1) seasonOK = c.allowZaid;
  else seasonOK = c.allowKharif;

  float seasonFactor = seasonOK ? 1.0f : 0.55f;
  if (f.ec_norm > (c.ec_max + 0.20f)) seasonFactor *= 0.70f;

  r.score = base * seasonFactor;

  // ---- Clear, farmer-friendly reasons (no "OK" words) ----
  if (!seasonOK) addReason(r.reason, sizeof(r.reason), "Out-of-season");

  // Soil moisture
  if (sM >= 0.9f) {
    addReason(r.reason, sizeof(r.reason), "Soil moisture ideal");
  } else {
    if (f.moist_norm < c.m_lo) addReason(r.reason, sizeof(r.reason), "Soil too dry");
    else if (f.moist_norm > c.m_hi) addReason(r.reason, sizeof(r.reason), "Soil too wet");
  }

  // Air dryness (VPD)
  if (sV >= 0.9f) {
    addReason(r.reason, sizeof(r.reason), "Air dryness ideal");
  } else {
    if (f.vpd_norm < c.v_lo) addReason(r.reason, sizeof(r.reason), "Air too humid");
    else if (f.vpd_norm > c.v_hi) addReason(r.reason, sizeof(r.reason), "Air too dry");
  }

  // Salinity
  if (f.ec_norm <= c.ec_max) addReason(r.reason, sizeof(r.reason), "Soil salinity within limit");
  else addReason(r.reason, sizeof(r.reason), "Soil salinity high");

  // Temperature & humidity wording
  if (sT >= 0.9f) addReason(r.reason, sizeof(r.reason), "Temperature ideal");
  if (sH >= 0.9f) addReason(r.reason, sizeof(r.reason), "Humidity ideal");
  // --------------------------------------------------------

  // Personalization bias
  float bias = getBiasForCrop(c.name);
  r.score = r.score + (PREF_WEIGHT * bias);

  return r;
}

void recommendTop3(NormFeatures f, CropResult out[3]) {
  CropResult all[N_CROPS];
  for (int i=0; i<N_CROPS; ++i) {
    all[i] = evalCrop(CROPS[i], f);
  }
  // Simple top-3 selection
  for (int k=0; k<3; ++k) {
    int best = k;
    for (int j=k+1; j<N_CROPS; ++j) {
      if (all[j].score > all[best].score) best = j;
    }
    CropResult tmp = all[k]; all[k] = all[best]; all[best] = tmp;
    out[k] = all[k];
  }
}

// -------- Month / Time helpers --------
int get_current_month_via_ntp() {
  bool haveCreds = (strlen(WIFI_SSID) > 0);
  if (haveCreds) {
    if (WiFi.status() != WL_CONNECTED) {
      WiFi.mode(WIFI_STA);
      WiFi.begin(WIFI_SSID, WIFI_PASS);
      unsigned long start = millis();
      while (WiFi.status() != WL_CONNECTED && (millis() - start) < 10000UL) { delay(200); }
    }
    configTzTime(TZ_INFO, NTP_SERVER);
    struct tm timeinfo;
    for (int i=0; i<20; ++i) {
      if (getLocalTime(&timeinfo, 200)) {
        return timeinfo.tm_mon + 1;
      }
      delay(200);
    }
  }
  // Fallback: __DATE__ ("Nov 5 2025")
  const char* dateStr = __DATE__;
  char mon3[4]; mon3[0]=dateStr[0]; mon3[1]=dateStr[1]; mon3[2]=dateStr[2]; mon3[3]=0;
  const char* names = "JanFebMarAprMayJunJulAugSepOctNovDec";
  const char* p = strstr(names, mon3);
  if (!p) return 11; // Nov
  int idx = (int)((p - names)/3);
  return idx + 1;
}

String now_iso8601() {
  struct tm timeinfo;
  char buf[40];
  bool haveCreds = (strlen(WIFI_SSID) > 0);
  if (haveCreds) {
    if (getLocalTime(&timeinfo, 200)) {
      strftime(buf, sizeof(buf), "%Y-%m-%dT%H:%M:%S%z", &timeinfo);
      return String(buf);
    }
  }
  return String(__DATE__) + "T" + String(__TIME__);
}

// -------- Button wait (INPUT_PULLUP) --------
bool waitForButtonPress(uint8_t pin) {
  const uint16_t debounceMs = 30;
  while (true) {
    if (digitalRead(pin) == LOW) {
      delay(debounceMs);
      if (digitalRead(pin) == LOW) return true;
    }
    delay(5);
  }
}

// -------- 60s capture (NO technical/normalized prints here) --------
CaptureOut captureOneMinute() {
  unsigned long startMs = millis();
  unsigned long lastSample = 0;
  unsigned long lastDht = 0;

  double sumTemp=0, sumHum=0, sumVpd=0, sumMoistPct=0, sumInvR25=0;
  uint32_t nDht=0, nSoil=0, nEC=0;
  float lastTempC = NAN;

  while (millis() - startMs < CAPTURE_MS) {
    // DHT every ~2s
    if (millis() - lastDht >= DHT_MIN_PERIOD_MS) {
      lastDht = millis();
      float tC = dht.readTemperature();
      float rh = dht.readHumidity();
      if (!isnan(tC) && !isnan(rh)) {
        lastTempC = tC;
        float vpd = vpd_kpa_from(tC, rh);
        sumTemp += tC; sumHum += rh; sumVpd += vpd; nDht++;
      }
    }
    // Soil & EC every SAMPLE_PERIOD_MS
    if (millis() - lastSample >= SAMPLE_PERIOD_MS) {
      lastSample = millis();
      // Soil
      int soilAdc = readADCavg(SOIL_AO_PIN, 4);
      float soilPct = moisture_pct_from_raw(soilAdc);
      if (!isnan(soilPct)) { sumMoistPct += soilPct; nSoil++; }
      // EC → invR → temp‑comp to invR25 → EC25(dS/m)
      int   ecAdc   = readADCavg(EC_PIN, 4);
      float vout_ec = adcToVoltage(ecAdc);
      float rsoil   = voltageToRsoil(vout_ec);
      float invR    = (rsoil <= 0.0f) ? 0.0f : (1.0f / rsoil);
      float tC_for_ec = isnan(lastTempC) ? 25.0f : lastTempC;
      float invR25  = invR_tempComp(invR, tC_for_ec);
      sumInvR25 += invR25; nEC++;
    }
    delay(5);
  }

  CaptureOut out;
  if (nDht == 0 || nSoil == 0 || nEC == 0) { out.ok = false; return out; }

  Snapshot snap;
  snap.temp_c    = (float)(sumTemp / nDht);
  snap.humidity  = (float)(sumHum / nDht);
  snap.vpd_kpa   = (float)(sumVpd / nDht);
  snap.soil_pct  = (float)(sumMoistPct / nSoil);
  float avgInvR25= (float)(sumInvR25 / nEC);
  snap.ec25_ds_m = ec25_from_invR25(avgInvR25);
  snap.season    = season_from_month(get_current_month_via_ntp());

  NormFeatures nf;
  nf.temp_norm  = clamp01(snap.temp_c / 50.0f);
  nf.hum_norm   = clamp01(snap.humidity / 100.0f);
  nf.vpd_norm   = clamp01(snap.vpd_kpa / 4.0f);
  nf.moist_norm = clamp01(snap.soil_pct / 100.0f);
  nf.ec_norm    = clamp01(snap.ec25_ds_m / 8.0f);
  nf.season     = snap.season;

  out.snap = snap;
  out.nf   = nf;
  out.ok   = true;
  return out;
}

// -------- SPIFFS storage --------
bool initStorage() {
  bool ok = SPIFFS.begin(true);
  return ok;
}

bool ensureChoicesHeader() {
  if (!SPIFFS.exists(CHOICES_CSV)) {
    File f = SPIFFS.open(CHOICES_CSV, FILE_WRITE);
    if (!f) return false;
    // crop1/crop2/crop3 stored as a single multi‑line CSV cell
    f.println("ts,tempC,RelativeHumidity%,VPD_kPa,Moist%,EC25_dS_m,season,crop1\\ncrop2\\ncrop3,chosen_idx,chosen_name");
    f.close();
  }
  return true;
}

bool appendChoiceCSV(const char* tsISO, Snapshot snap, NormFeatures nf, const CropResult top3[3], int chosenIdx) {
  File f = SPIFFS.open(CHOICES_CSV, FILE_APPEND);
  if (!f) return false;
  String crops = String(top3[0].name) + "\\n" + String(top3[1].name) + "\\n" + String(top3[2].name);
  String line = String(tsISO) + "," +
    String(snap.temp_c,2)     + "," + String(snap.humidity,2) + "," +
    String(snap.vpd_kpa,3)    + "," + String(snap.soil_pct,1) + "," +
    String(snap.ec25_ds_m,3)  + "," + String(nf.season)       + "," +
    crops + "," + String(chosenIdx) + "," + String(top3[chosenIdx].name);
  f.println(line);
  f.close();
  return true;
}

// -------- Feedback persistence (CSV) --------
bool savePrefsCSV() {
  File f = SPIFFS.open(PREFS_CSV, FILE_WRITE);
  if (!f) return false;
  f.println("crop,bias");
  for (int i=0; i<N_CROPS; ++i) {
    String line = String(CROPS[i].name) + "," + String(gBias[i], 6);
    f.println(line);
  }
  f.close();
  return true;
}

bool loadPrefsCSV() {
  if (!SPIFFS.exists(PREFS_CSV)) return false;
  File f = SPIFFS.open(PREFS_CSV, FILE_READ);
  if (!f) return false;
  for (int i=0; i<N_CROPS; ++i) gBias[i] = 0.0f; // init
  // skip header
  String header = f.readStringUntil('\n');
  while (f.available()) {
    String line = f.readStringUntil('\n');
    line.trim();
    if (line.length() == 0) continue;
    int comma = line.indexOf(',');
    if (comma <= 0) continue;
    String name = line.substring(0, comma);
    String biasStr = line.substring(comma + 1);
    biasStr.trim();
    float bias = biasStr.toFloat();
    for (int i=0; i<N_CROPS; ++i) {
      if (name.equals(String(CROPS[i].name))) {
        gBias[i] = bias;
        break;
      }
    }
  }
  f.close();
  clampBiases();
  return true;
}

bool ensurePrefsCSV() {
  if (!SPIFFS.exists(PREFS_CSV)) {
    for (int i=0; i<N_CROPS; ++i) gBias[i] = 0.0f;
    return savePrefsCSV();
  }
  return loadPrefsCSV();
}

// -------- Choice buttons --------
int waitForCropChoice(const CropResult top3[3]) {
  Serial.println(F("\nChoose a crop:"));
  Serial.print(F("[Button 1 @ GPIO21] ")); Serial.println(top3[0].name);
  Serial.print(F("[Button 2 @ GPIO22] ")); Serial.println(top3[1].name);
  Serial.print(F("[Button 3 @ GPIO23] ")); Serial.println(top3[2].name);
  const uint16_t debounceMs = 30;
  while (true) {
    if (digitalRead(BUTTON_CHOOSE_1_PIN) == LOW) {
      delay(debounceMs); if (digitalRead(BUTTON_CHOOSE_1_PIN) == LOW) return 0;
    }
    if (digitalRead(BUTTON_CHOOSE_2_PIN) == LOW) {
      delay(debounceMs); if (digitalRead(BUTTON_CHOOSE_2_PIN) == LOW) return 1;
    }
    if (digitalRead(BUTTON_CHOOSE_3_PIN) == LOW) {
      delay(debounceMs); if (digitalRead(BUTTON_CHOOSE_3_PIN) == LOW) return 2;
    }
    delay(5);
  }
}

// -------- Farmer‑friendly summary --------
void printFriendlySummaryAndTop3(const CaptureOut& out, const CropResult top3[3]) {
  const float temp_c = out.snap.temp_c;
  const float moist  = out.snap.soil_pct;
  const float ec25   = out.snap.ec25_ds_m;
  const float ec_pct = clamp01(out.nf.ec_norm) * 100.0f;

  Serial.println(F("\n=== Field Summary (Averaged 60s) ==="));
  Serial.print(F("Temperature: "));           Serial.print(temp_c, 1); Serial.println(F(" °C"));
  Serial.print(F("Soil Moisture: "));         Serial.print(moist, 1);  Serial.println(F(" %"));
  Serial.print(F("Electrical Conductivity @25°C: "));
  Serial.print(ec25, 3); Serial.print(F(" dS/m (EC%: "));
  Serial.print(ec_pct, 2); Serial.println(F("%)"));
  Serial.print(F("Relative Humidity: "));     Serial.print(out.snap.humidity, 1); Serial.println(F(" %"));
  Serial.print(F("Vapor Pressure Deficit: "));Serial.print(out.snap.vpd_kpa, 3);  Serial.println(F(" kPa"));
  Serial.print(F("Season: "));
  if (out.nf.season==0) Serial.println(F("Rabi"));
  else if (out.nf.season==1) Serial.println(F("Zaid"));
  else Serial.println(F("Kharif"));

  Serial.println(F("\n=== Top-3 Crop Suggestions ==="));
  for (int i=0; i<3; ++i) {
    Serial.print(i+1); Serial.print(F(") "));
    Serial.print(top3[i].name);
    Serial.print(F(" score=")); Serial.println(top3[i].score, 3);
    Serial.print(F(" Why: "));
    if (top3[i].reason[0] != '\0') Serial.println(top3[i].reason);
    else Serial.println(F("—"));
  }
}

// -------- Setup / Loop --------
void setup() {
  Serial.begin(115200);
  delay(200);
  dht.begin();
  analogReadResolution(ADC_BITS);
  analogSetPinAttenuation(SOIL_AO_PIN, ADC_11db);
  analogSetPinAttenuation(EC_PIN,      ADC_11db);
  pinMode(BUTTON_CAPTURE_PIN,  INPUT_PULLUP);
  pinMode(BUTTON_CHOOSE_1_PIN, INPUT_PULLUP);
  pinMode(BUTTON_CHOOSE_2_PIN, INPUT_PULLUP);
  pinMode(BUTTON_CHOOSE_3_PIN, INPUT_PULLUP);

  Serial.println();
  Serial.println(F("=== ESP32 Crop Recommender (Top‑3 + 3 Choice Buttons + Feedback) ==="));
  Serial.println(F("DHT11=GPIO4, Soil AO=GPIO34, EC=GPIO35, Capture=GPIO18"));
  Serial.println(F("Choice: Crop#1=GPIO21, Crop#2=GPIO22, Crop#3=GPIO23"));
  Serial.println(F("Press the capture button to start a 60-second capture..."));

  bool haveCreds = (strlen(WIFI_SSID) > 0);
  if (haveCreds) { WiFi.mode(WIFI_STA); WiFi.begin(WIFI_SSID, WIFI_PASS); }

  if (!initStorage()) {
    Serial.println(F("[WARN] SPIFFS init failed; logging/feedback disabled."));
  } else {
    bool okChoices = ensureChoicesHeader();
    bool okPrefs   = ensurePrefsCSV();
    Serial.println(okChoices ? F("[OK] /choices.csv ready") : F("[WARN] choices header fail"));
    Serial.println(okPrefs   ? F("[OK] /prefs.csv loaded")  : F("[WARN] prefs load fail; zeros"));
  }
}

void loop() {
  Serial.println(F("Press the capture button to start 60s capture..."));
  bool pressed = waitForButtonPress(BUTTON_CAPTURE_PIN);
  if (!pressed) return;

  Serial.println(F("[CAPTURE] Started 1-minute averaging..."));
  CaptureOut out = captureOneMinute();
  if (!out.ok) {
    Serial.println(F("[CAPTURE] Failed. Try again."));
    delay(1000);
    return;
  }

  CropResult top3[3];
  recommendTop3(out.nf, top3);

  // === Farmer view (summary first, then Top‑3) ===
  printFriendlySummaryAndTop3(out, top3);

  // Wait for user choice (top3 index 0..2)
  int chosenIdx = waitForCropChoice(top3);
  Serial.print(F("\n[CHOICE] You selected: ")); Serial.println(top3[chosenIdx].name);

  // Map to GLOBAL index and apply feedback learning
  int globalIdx = cropIndexByName(top3[chosenIdx].name);
  if (globalIdx >= 0) {
    applyFeedbackLearningGlobal(globalIdx);  // <-- bias fix
  } else {
    Serial.println(F("[FEEDBACK] ERROR: chosen crop not found in CROPS[]"));
  }

  // Log choice
  String tsISO = now_iso8601();
  bool logged = appendChoiceCSV(tsISO.c_str(), out.snap, out.nf, top3, chosenIdx);
  Serial.println(logged ? F("[LOG] Choice saved to /choices.csv") : F("[LOG] Failed to save choice"));

  // Persist preferences
  bool savedPrefs = savePrefsCSV();
  Serial.println(savedPrefs ? F("[PREFS] Saved /prefs.csv") : F("[PREFS] Save failed"));

  // No JSON printing (screen kept clean)
  Serial.println(F("\n-- Ready for next capture --\n"));
  delay(1500);
}