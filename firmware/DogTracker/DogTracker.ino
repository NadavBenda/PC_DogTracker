/*
 * DogTracker ESP32-S3 firmware
 * ------------------------------------------------------------------
 * - Continuously captures JPEG frames and saves them to the SD card as
 *   /<session>/dog_<millis>.jpg (compatible with the PC_DogTracker
 *   analysis tool's filename parser -- point it at one session folder,
 *   not the SD root). Baseline rate is CAPTURE_FPS; while the network
 *   window (below) is open, the rate drops to PREVIEW_CAPTURE_FPS to
 *   leave headroom for streaming instead of contending with it.
 * - At boot, tries to join one of WIFI_CANDIDATES (home Wi-Fi networks,
 *   in order) for up to NETWORK_WINDOW_MS. If one connects: gets real
 *   time via NTP automatically (renaming the session folder to a real
 *   "YYYYMMDD_HHMMSS"), and serves a live MJPEG preview + a client-side
 *   focus-score readout at http://<device-ip>/ so you can check camera
 *   placement and physical focus. After the window (or immediately, if
 *   no known network was reachable), Wi-Fi is shut down completely and
 *   only SD recording continues at the full baseline rate.
 * - The board has no RTC, so if no network/NTP is available that boot,
 *   the session folder keeps a boot-relative fallback name instead --
 *   made collision-proof across power cycles via a boot counter
 *   persisted on the SD card (millis() alone resets every reboot and
 *   isn't reliably unique if boot timing is consistent).
 * - Recording stops (cleanly) once the SD card runs out of space;
 *   existing frames are never overwritten or deleted.
 *
 * Hardware notes / things to verify for your specific board:
 *   - Camera pins below are taken from your working reference sketch.
 *   - SD card is wired for SD_MMC 1-bit mode on GPIO38 (CMD), GPIO39
 *     (CLK), GPIO40 (D0) -- confirm against your board's silkscreen/
 *     schematic; SD_MMC.begin() will fail loudly (see Serial output)
 *     if these are wrong -- it will not damage anything.
 *   - Status LED on GPIO2 is optional/cosmetic -- remove STATUS_LED_PIN
 *     if your board doesn't have one there or you want the pin free.
 *
 * Arduino IDE board settings (ESP32-S3 N16R8):
 *   Board: "ESP32S3 Dev Module"
 *   USB CDC On Boot: Enabled (for Serial over USB)
 *   Flash Size: 16MB
 *   Partition Scheme: "Huge APP (3MB No OTA/1MB SPIFFS)" or similar,
 *     with enough room for the sketch + camera/Wi-Fi/SD libraries
 *   PSRAM: "OPI PSRAM"
 * ------------------------------------------------------------------
 */

#include "esp_camera.h"
#include <WiFi.h>
#include "esp_http_server.h"
#include "SD_MMC.h"
#include "FS.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/semphr.h"
#include <time.h>
#include <sys/time.h>

// ======================================================
// Home Wi-Fi networks to try, in order, each for up to
// WIFI_CONNECT_TIMEOUT_MS. First one that connects wins. If none do,
// networking is skipped entirely for this boot -- recording still
// proceeds normally, just without a live preview or NTP time.
// ======================================================
struct WifiCandidate {
  const char *ssid;
  const char *password;
};
const WifiCandidate WIFI_CANDIDATES[] = {
  { "hevraty", "hevraty321" },
  { "Nitsan", "0503332262" },
};
const uint32_t WIFI_CONNECT_TIMEOUT_MS = 8000;
const uint32_t NETWORK_WINDOW_MS = 4UL * 60UL * 1000UL; // 4 minutes of preview/NTP once connected

// ======================================================
// Frame rates
// ======================================================
const uint32_t CAPTURE_FPS = 3;         // baseline, once the network window is closed (or never opened)
const uint32_t PREVIEW_CAPTURE_FPS = 2; // reduced rate while the network window is open, to leave
                                         // headroom for the camera+SD+Wi-Fi to not contend with each other

// ======================================================
// Time sync -- the board has no RTC. If a known Wi-Fi network is
// reachable at boot, NTP gives it real time automatically (see
// syncTimeViaNtp()); otherwise the session folder keeps its
// boot-relative fallback name for that recording. TZ_OFFSET_SECONDS
// only affects the human-readable folder name, not any recorded data.
// ======================================================
const long TZ_OFFSET_SECONDS = 2L * 3600L; // assumes Israel Standard Time (UTC+2); +3h in DST

// ======================================================
// Camera pins (ESP32-S3-CAM, from the known-working reference sketch)
// ======================================================
#define PWDN_GPIO_NUM     -1
#define RESET_GPIO_NUM    -1
#define XCLK_GPIO_NUM     15
#define SIOD_GPIO_NUM     4
#define SIOC_GPIO_NUM     5

#define Y9_GPIO_NUM       16
#define Y8_GPIO_NUM       17
#define Y7_GPIO_NUM       18
#define Y6_GPIO_NUM       12
#define Y5_GPIO_NUM       10
#define Y4_GPIO_NUM       8
#define Y3_GPIO_NUM       9
#define Y2_GPIO_NUM       11
#define VSYNC_GPIO_NUM    6
#define HREF_GPIO_NUM     7
#define PCLK_GPIO_NUM     13

// ======================================================
// SD card pins (SD_MMC 1-bit mode) -- confirm against your board
// ======================================================
#define SD_MMC_CLK_PIN 39
#define SD_MMC_CMD_PIN 38
#define SD_MMC_D0_PIN  40

// ======================================================
// Optional status LED (cosmetic only -- safe to remove)
// ======================================================
#define STATUS_LED_PIN 2

// ======================================================
// MJPEG stream protocol
// ======================================================
#define PART_BOUNDARY "123456789000000000000987654321"
static const char *_STREAM_CONTENT_TYPE = "multipart/x-mixed-replace;boundary=" PART_BOUNDARY;
static const char *_STREAM_BOUNDARY = "\r\n--" PART_BOUNDARY "\r\n";
static const char *_STREAM_PART = "Content-Type: image/jpeg\r\nContent-Length: %u\r\n\r\n";

httpd_handle_t stream_httpd = NULL;

// ======================================================
// Shared state between the capture task and the HTTP stream handler
// ======================================================
static SemaphoreHandle_t previewMutex;
static SemaphoreHandle_t previewReady;
static uint8_t *previewBuf = NULL;
static size_t previewLen = 0;
static size_t previewCapacity = 0;
static volatile uint32_t previewSeq = 0; // bumped each time previewBuf's content changes

static volatile bool networkWindowOpen = false;
static uint32_t networkWindowStart = 0;
static volatile bool sdFull = false;
static uint32_t frameCounter = 0;

// Recording stops once free space drops below this, so a last partial
// write can never corrupt the card or silently truncate a file.
const uint64_t SD_SAFETY_MARGIN_BYTES = 256UL * 1024UL;
// How often (in frames) to re-check free space -- doesn't need to be
// every frame, the safety margin above absorbs the extra ~5s of slack.
const uint32_t SD_SPACE_CHECK_EVERY_N_FRAMES = 25;

// ======================================================
// Current session folder -- frames go to "<currentSessionDir>/dog_<millis>.jpg".
// Starts as a boot-relative name; syncTimeViaNtp() switches it (for new
// frames only -- see its comment) to one named from the real start time
// if NTP succeeds. Guarded by sessionDirMutex since the capture task and
// setup()'s NTP sync run in different tasks.
// ======================================================
static SemaphoreHandle_t sessionDirMutex;
static char currentSessionDir[40];
static volatile bool timeSynced = false;

// ======================================================
// Camera init
// ======================================================
bool initCamera() {
  camera_config_t config;
  config.ledc_channel = LEDC_CHANNEL_0;
  config.ledc_timer   = LEDC_TIMER_0;
  config.pin_d0       = Y2_GPIO_NUM;
  config.pin_d1       = Y3_GPIO_NUM;
  config.pin_d2       = Y4_GPIO_NUM;
  config.pin_d3       = Y5_GPIO_NUM;
  config.pin_d4       = Y6_GPIO_NUM;
  config.pin_d5       = Y7_GPIO_NUM;
  config.pin_d6       = Y8_GPIO_NUM;
  config.pin_d7       = Y9_GPIO_NUM;
  config.pin_xclk     = XCLK_GPIO_NUM;
  config.pin_pclk     = PCLK_GPIO_NUM;
  config.pin_vsync    = VSYNC_GPIO_NUM;
  config.pin_href     = HREF_GPIO_NUM;
  config.pin_sccb_sda = SIOD_GPIO_NUM;
  config.pin_sccb_scl = SIOC_GPIO_NUM;
  config.pin_pwdn     = PWDN_GPIO_NUM;
  config.pin_reset    = RESET_GPIO_NUM;
  config.xclk_freq_hz = 20000000;
  config.pixel_format = PIXFORMAT_JPEG;

  if (psramFound()) {
    Serial.println("PSRAM found -- using VGA capture");
    config.frame_size   = FRAMESIZE_VGA; // 640x480
    config.jpeg_quality = 10;
    config.fb_count     = 3;
    config.grab_mode    = CAMERA_GRAB_LATEST;
    config.fb_location  = CAMERA_FB_IN_PSRAM;
  } else {
    Serial.println("WARNING: no PSRAM -- falling back to CIF capture, timings may not be sustainable");
    config.frame_size   = FRAMESIZE_CIF;
    config.jpeg_quality = 12;
    config.fb_count     = 1;
    config.grab_mode    = CAMERA_GRAB_LATEST;
    config.fb_location  = CAMERA_FB_IN_DRAM;
  }

  esp_err_t err = esp_camera_init(&config);
  if (err != ESP_OK) {
    Serial.printf("Camera init failed: 0x%x\n", err);
    return false;
  }
  return true;
}

// ======================================================
// SD card init
// ======================================================
bool initSD() {
  // Argument order is (clk, cmd, d0) -- easy to mix up with cmd/clk swapped.
  SD_MMC.setPins(SD_MMC_CLK_PIN, SD_MMC_CMD_PIN, SD_MMC_D0_PIN);
  if (!SD_MMC.begin("/sdcard", true /* 1-bit mode */)) {
    Serial.println("SD_MMC mount failed -- check CLK/CMD/D0 wiring and that the card is formatted FAT32.");
    return false;
  }
  if (SD_MMC.cardType() == CARD_NONE) {
    Serial.println("No SD card detected.");
    return false;
  }
  uint64_t totalMB = SD_MMC.totalBytes() / (1024 * 1024);
  uint64_t usedMB = SD_MMC.usedBytes() / (1024 * 1024);
  Serial.printf("SD card ready: %llu MB used / %llu MB total\n", usedMB, totalMB);
  return true;
}

// ======================================================
// Boot counter -- persisted on the SD card so the fallback session name
// is unique across power cycles, not just within one. millis() alone
// resets to 0 every reboot, so two boots with similar startup timing
// (very common -- the init sequence takes about as long every time)
// can otherwise land on the same boot-relative folder name.
// ======================================================
const char *BOOT_COUNTER_PATH = "/boot_count.txt";

uint32_t nextBootCounter() {
  uint32_t count = 0;
  File in = SD_MMC.open(BOOT_COUNTER_PATH, FILE_READ);
  if (in) {
    count = (uint32_t)in.parseInt();
    in.close();
  }
  count++;
  File out = SD_MMC.open(BOOT_COUNTER_PATH, FILE_WRITE);
  if (out) {
    out.print(count);
    out.close();
  } else {
    Serial.println("Warning: could not persist boot counter -- fallback names may collide across reboots.");
  }
  return count;
}

// ======================================================
// Session folder management
// ======================================================

// Creates `dir` on the SD card and (if it didn't already match) makes it
// the active session folder for all frames from this point on. Existing
// files already written to the previous session folder are left in place.
void switchSessionFolder(const char *dir) {
  if (xSemaphoreTake(sessionDirMutex, portMAX_DELAY) != pdTRUE) {
    return;
  }
  if (strcmp(currentSessionDir, dir) != 0) {
    SD_MMC.mkdir(dir);
    strncpy(currentSessionDir, dir, sizeof(currentSessionDir) - 1);
    currentSessionDir[sizeof(currentSessionDir) - 1] = '\0';
  }
  xSemaphoreGive(sessionDirMutex);
}

void startBootRelativeSession() {
  uint32_t bootNum = nextBootCounter();
  char dir[40];
  snprintf(dir, sizeof(dir), "/session_b%lu_%lu", (unsigned long)bootNum, (unsigned long)millis());
  switchSessionFolder(dir);
  Serial.printf("Session folder: %s (boot-relative; will rename if NTP syncs)\n", dir);
}

// Tries to get real time over NTP (only meaningful once Wi-Fi is up).
// On success, renames new frames' folder to a real "YYYYMMDD_HHMMSS"
// and returns true; on timeout, leaves the boot-relative name in place.
bool syncTimeViaNtp(uint32_t timeoutMs) {
  configTime(0, 0, "pool.ntp.org", "time.google.com", "time.nist.gov");

  uint32_t start = millis();
  time_t now = 0;
  while (millis() - start < timeoutMs) {
    time(&now);
    if (now > 1700000000) { // sanity floor (~Nov 2023) -- rejects the pre-sync default clock value
      break;
    }
    delay(200);
  }
  if (now <= 1700000000) {
    Serial.println("NTP sync timed out -- keeping boot-relative session folder name.");
    return false;
  }

  timeSynced = true;
  time_t localEpoch = now + TZ_OFFSET_SECONDS;
  struct tm timeinfo;
  gmtime_r(&localEpoch, &timeinfo); // epoch already shifted, so treat as UTC to avoid needing a TZ database

  char dir[40];
  strftime(dir, sizeof(dir), "/%Y%m%d_%H%M%S", &timeinfo);
  switchSessionFolder(dir);
  Serial.printf("NTP time synced -- new frames now saved under %s\n", dir);
  return true;
}

bool sdHasSpace() {
  uint64_t freeBytes = SD_MMC.totalBytes() - SD_MMC.usedBytes();
  return freeBytes > SD_SAFETY_MARGIN_BYTES;
}

// ======================================================
// Publish a frame to the preview stream (only called while the network
// window is open). previewSeq lets the stream handler tell whether the
// content actually changed since it last sent something, so it never
// re-sends a stale frame while waiting for the next real one.
// ======================================================
void publishPreviewFrame(camera_fb_t *fb) {
  if (xSemaphoreTake(previewMutex, pdMS_TO_TICKS(50)) != pdTRUE) {
    return; // stream handler is mid-read; skip this frame, no big deal at a couple FPS
  }
  if (previewCapacity < fb->len) {
    free(previewBuf);
    previewBuf = (uint8_t *)malloc(fb->len);
    previewCapacity = previewBuf ? fb->len : 0;
  }
  if (previewBuf) {
    memcpy(previewBuf, fb->buf, fb->len);
    previewLen = fb->len;
    previewSeq++;
  }
  xSemaphoreGive(previewMutex);
  xSemaphoreGive(previewReady);
}

// ======================================================
// Capture task: the sole owner of esp_camera_fb_get()/fb_return().
// Runs forever, saving every frame to SD at the current rate (slower
// while the network window is open, so streaming isn't starved), and
// -- only while that window is open -- feeding the preview stream too.
// ======================================================
void captureTask(void *pvParameters) {
  uint32_t nextFrameDue = millis();

  while (true) {
    uint32_t intervalMs = 1000 / (networkWindowOpen ? PREVIEW_CAPTURE_FPS : CAPTURE_FPS);

    if (sdFull) {
      vTaskDelay(pdMS_TO_TICKS(1000)); // stopped cleanly; nothing left to do
      continue;
    }

    if (frameCounter % SD_SPACE_CHECK_EVERY_N_FRAMES == 0 && !sdHasSpace()) {
      sdFull = true;
      Serial.println("SD card full -- recording stopped. Existing files are untouched.");
      continue;
    }

    camera_fb_t *fb = esp_camera_fb_get();
    if (!fb) {
      Serial.println("Camera capture failed");
      vTaskDelay(pdMS_TO_TICKS(50));
      continue;
    }

    char sessionDir[40];
    if (xSemaphoreTake(sessionDirMutex, portMAX_DELAY) == pdTRUE) {
      strncpy(sessionDir, currentSessionDir, sizeof(sessionDir));
      xSemaphoreGive(sessionDirMutex);
    }
    char filename[80];
    snprintf(filename, sizeof(filename), "%s/dog_%lu.jpg", sessionDir, (unsigned long)millis());
    File file = SD_MMC.open(filename, FILE_WRITE);
    if (file) {
      file.write(fb->buf, fb->len);
      file.close();
    } else {
      Serial.printf("Failed to open %s for writing\n", filename);
    }

    frameCounter++;
    if (networkWindowOpen) {
      publishPreviewFrame(fb);
    }

    esp_camera_fb_return(fb);

    nextFrameDue += intervalMs;
    int32_t delayMs = (int32_t)(nextFrameDue - millis());
    if (delayMs > 0) {
      vTaskDelay(pdMS_TO_TICKS(delayMs));
    } else {
      nextFrameDue = millis(); // capture+write ran over budget; resync instead of free-running
    }
  }
}

// ======================================================
// HTTP stream handler -- reads from the shared preview buffer only,
// never touches the camera driver directly. Skips re-sending a frame
// whose sequence number hasn't changed since the last one we sent.
// ======================================================
static esp_err_t stream_handler(httpd_req_t *req) {
  esp_err_t res = httpd_resp_set_type(req, _STREAM_CONTENT_TYPE);
  if (res != ESP_OK) {
    return res;
  }

  char part_buf[64];
  uint32_t lastSentSeq = 0;
  while (networkWindowOpen) {
    if (xSemaphoreTake(previewReady, pdMS_TO_TICKS(2000)) != pdTRUE) {
      continue; // no new frame yet, keep waiting while the window is open
    }

    if (xSemaphoreTake(previewMutex, portMAX_DELAY) != pdTRUE) {
      break;
    }
    uint32_t seq = previewSeq;
    size_t len = previewLen;
    uint8_t *copy = NULL;
    if (seq != lastSentSeq && len) {
      copy = (uint8_t *)malloc(len);
      if (copy) {
        memcpy(copy, previewBuf, len);
      }
    }
    xSemaphoreGive(previewMutex);

    if (!copy) {
      continue; // nothing new since the last frame we actually sent
    }
    lastSentSeq = seq;

    size_t hlen = snprintf(part_buf, sizeof(part_buf), _STREAM_PART, (unsigned)len);
    res = httpd_resp_send_chunk(req, part_buf, hlen);
    if (res == ESP_OK) {
      res = httpd_resp_send_chunk(req, (const char *)copy, len);
    }
    if (res == ESP_OK) {
      res = httpd_resp_send_chunk(req, _STREAM_BOUNDARY, strlen(_STREAM_BOUNDARY));
    }
    free(copy);

    if (res != ESP_OK) {
      break;
    }
  }
  return ESP_OK;
}

// GET / -- the live preview page. Draws the streamed frame onto a hidden
// canvas a couple times a second and computes a Laplacian-variance-style
// sharpness score client-side (cheap enough for a phone browser), so you
// can watch the number while turning the lens's physical focus ring.
static const char INDEX_HTML[] = R"HTML(<!doctype html><html><body style="margin:0;background:#000;color:#0f0;font-family:monospace">
<div id="score" style="position:absolute;top:4px;left:4px;font-size:20px;background:rgba(0,0,0,0.5);padding:4px 8px">focus score: --</div>
<img id="s" src="/stream" style="width:100%;display:block">
<canvas id="c" style="display:none"></canvas>
<script>
var img = document.getElementById('s');
var canvas = document.getElementById('c');
var ctx = canvas.getContext('2d', { willReadFrequently: true });
var scoreEl = document.getElementById('score');
function computeFocusScore() {
  if (!img.naturalWidth) { return; }
  var w = 160, h = Math.round(160 * img.naturalHeight / img.naturalWidth);
  canvas.width = w; canvas.height = h;
  ctx.drawImage(img, 0, 0, w, h);
  var data;
  try { data = ctx.getImageData(0, 0, w, h).data; } catch (e) { return; }
  var gray = new Float32Array(w * h);
  for (var i = 0; i < w * h; i++) {
    var o = i * 4;
    gray[i] = 0.299 * data[o] + 0.587 * data[o + 1] + 0.114 * data[o + 2];
  }
  var sum = 0, sumSq = 0, n = 0;
  for (var y = 1; y < h - 1; y++) {
    for (var x = 1; x < w - 1; x++) {
      var idx = y * w + x;
      var lap = -4 * gray[idx] + gray[idx - 1] + gray[idx + 1] + gray[idx - w] + gray[idx + w];
      sum += lap; sumSq += lap * lap; n++;
    }
  }
  var mean = sum / n;
  var variance = sumSq / n - mean * mean;
  scoreEl.textContent = 'focus score: ' + Math.round(variance) + ' (higher = sharper; turn the lens and watch this)';
}
setInterval(computeFocusScore, 500);
</script>
</body></html>
)HTML";

static esp_err_t index_handler(httpd_req_t *req) {
  httpd_resp_set_type(req, "text/html");
  httpd_resp_sendstr(req, INDEX_HTML);
  return ESP_OK;
}

void startCameraServer() {
  httpd_config_t config = HTTPD_DEFAULT_CONFIG();
  config.server_port = 80;

  httpd_uri_t stream_uri = {
    .uri = "/stream",
    .method = HTTP_GET,
    .handler = stream_handler,
    .user_ctx = NULL
  };
  httpd_uri_t index_uri = {
    .uri = "/",
    .method = HTTP_GET,
    .handler = index_handler,
    .user_ctx = NULL
  };

  if (httpd_start(&stream_httpd, &config) == ESP_OK) {
    httpd_register_uri_handler(stream_httpd, &stream_uri);
    httpd_register_uri_handler(stream_httpd, &index_uri);
  }
}

// Tries each known network in order, each for up to
// WIFI_CONNECT_TIMEOUT_MS. Returns true (and stays connected) on the
// first one that works; leaves Wi-Fi off if none do.
bool connectToKnownWifi() {
  WiFi.mode(WIFI_STA);
  size_t count = sizeof(WIFI_CANDIDATES) / sizeof(WIFI_CANDIDATES[0]);
  for (size_t i = 0; i < count; i++) {
    const WifiCandidate &candidate = WIFI_CANDIDATES[i];
    Serial.printf("Trying Wi-Fi \"%s\"...\n", candidate.ssid);
    WiFi.begin(candidate.ssid, candidate.password);

    uint32_t start = millis();
    while (WiFi.status() != WL_CONNECTED && millis() - start < WIFI_CONNECT_TIMEOUT_MS) {
      delay(250);
    }
    if (WiFi.status() == WL_CONNECTED) {
      Serial.printf("Connected to \"%s\", IP: %s\n", candidate.ssid, WiFi.localIP().toString().c_str());
      return true;
    }
    WiFi.disconnect(true);
    delay(100);
  }
  Serial.println("No known Wi-Fi network reachable -- skipping preview/NTP, recording at full rate.");
  return false;
}

// ======================================================
// Shut Wi-Fi + the HTTP server down once the network window elapses
// ======================================================
void closeNetworkWindow() {
  networkWindowOpen = false;
  xSemaphoreGive(previewReady); // wake the stream handler so it notices the window closed

  if (stream_httpd) {
    httpd_stop(stream_httpd);
    stream_httpd = NULL;
  }
  WiFi.disconnect(true);
  WiFi.mode(WIFI_OFF);
  Serial.println("Network window elapsed -- Wi-Fi and preview stream shut down. SD recording continues at full rate.");
}

void setup() {
  Serial.begin(115200);
  // Native USB CDC takes a moment to enumerate after reset -- wait for it,
  // but time out and carry on regardless so a field-deployed unit (no PC
  // ever attached) never hangs here waiting for a terminal that won't come.
  uint32_t serialWaitStart = millis();
  while (!Serial && millis() - serialWaitStart < 3000) {
    delay(10);
  }
  delay(200); // let the host terminal finish attaching before the first line
  Serial.println();

  pinMode(STATUS_LED_PIN, OUTPUT);
  digitalWrite(STATUS_LED_PIN, HIGH); // solid on while the network window is open

  if (!initCamera()) {
    Serial.println("Halting: camera init failed.");
    while (true) { delay(1000); }
  }

  if (!initSD()) {
    Serial.println("Halting: SD card init failed.");
    while (true) { delay(1000); }
  }

  previewMutex = xSemaphoreCreateMutex();
  previewReady = xSemaphoreCreateBinary();
  sessionDirMutex = xSemaphoreCreateMutex();
  startBootRelativeSession();

  if (connectToKnownWifi()) {
    syncTimeViaNtp(5000);
    startCameraServer();
    networkWindowOpen = true;
    networkWindowStart = millis();
    Serial.print("Preview + focus score: http://");
    Serial.print(WiFi.localIP());
    Serial.println("/");
  } else {
    networkWindowOpen = false;
    digitalWrite(STATUS_LED_PIN, LOW);
  }

  xTaskCreatePinnedToCore(captureTask, "captureTask", 8192, NULL, 1, NULL, 1);
}

void loop() {
  if (networkWindowOpen && millis() - networkWindowStart >= NETWORK_WINDOW_MS) {
    closeNetworkWindow();
    digitalWrite(STATUS_LED_PIN, LOW);
  }

  // Slow heartbeat blink once recording-only (no network window active)
  // to show the board is alive; fast blink instead if the SD card filled up.
  static uint32_t lastBlink = 0;
  if (!networkWindowOpen) {
    uint32_t period = sdFull ? 200 : 2000;
    if (millis() - lastBlink >= period) {
      lastBlink = millis();
      digitalWrite(STATUS_LED_PIN, !digitalRead(STATUS_LED_PIN));
    }
  }

  delay(50);
}
