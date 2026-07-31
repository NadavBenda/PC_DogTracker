/*
 * DogTracker ESP32-S3 firmware
 * ------------------------------------------------------------------
 * - Continuously captures JPEG frames at ~5 FPS and saves them to the
 *   SD card as /dog_<millis>.jpg (compatible with the PC_DogTracker
 *   analysis tool's filename parser).
 * - For the first AP_WINDOW_MS milliseconds (default 2 minutes) only,
 *   opens a Wi-Fi access point named "WatchDog" and serves a slow
 *   (default 1 FPS) MJPEG preview at http://192.168.4.1/stream so you
 *   can confirm camera placement/focus from a phone or laptop. After
 *   that window the AP and HTTP server are shut down completely and
 *   only SD recording continues.
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

// ======================================================
// Wi-Fi access point (first AP_WINDOW_MS only)
// ======================================================
const char *AP_SSID = "WatchDog";
const char *AP_PASSWORD = "12345678"; // WPA2 minimum is 8 characters
const uint32_t AP_WINDOW_MS = 2UL * 60UL * 1000UL; // 2 minutes

// ======================================================
// Frame rates
// ======================================================
const uint32_t CAPTURE_FPS = 5;
const uint32_t CAPTURE_INTERVAL_MS = 1000 / CAPTURE_FPS;
const uint32_t PREVIEW_FPS_DIVISOR = 5; // 5 FPS / 5 = 1 FPS preview stream

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

static volatile bool apWindowOpen = true;
static volatile bool sdFull = false;
static uint32_t frameCounter = 0;

// Recording stops once free space drops below this, so a last partial
// write can never corrupt the card or silently truncate a file.
const uint64_t SD_SAFETY_MARGIN_BYTES = 256UL * 1024UL;
// How often (in frames) to re-check free space -- doesn't need to be
// every frame, the safety margin above absorbs the extra ~5s of slack.
const uint32_t SD_SPACE_CHECK_EVERY_N_FRAMES = 25;

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
    Serial.println("WARNING: no PSRAM -- falling back to CIF capture, 5 FPS may not be sustainable");
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

bool sdHasSpace() {
  uint64_t freeBytes = SD_MMC.totalBytes() - SD_MMC.usedBytes();
  return freeBytes > SD_SAFETY_MARGIN_BYTES;
}

// ======================================================
// Publish a frame to the preview stream (only called during the AP window)
// ======================================================
void publishPreviewFrame(camera_fb_t *fb) {
  if (xSemaphoreTake(previewMutex, pdMS_TO_TICKS(50)) != pdTRUE) {
    return; // stream handler is mid-read; skip this frame, no big deal at 1 FPS
  }
  if (previewCapacity < fb->len) {
    free(previewBuf);
    previewBuf = (uint8_t *)malloc(fb->len);
    previewCapacity = previewBuf ? fb->len : 0;
  }
  if (previewBuf) {
    memcpy(previewBuf, fb->buf, fb->len);
    previewLen = fb->len;
  }
  xSemaphoreGive(previewMutex);
  xSemaphoreGive(previewReady);
}

// ======================================================
// Capture task: the sole owner of esp_camera_fb_get()/fb_return().
// Runs forever at CAPTURE_FPS, saving every frame to SD, and -- only
// while the AP window is open -- feeding a slower preview stream too.
// ======================================================
void captureTask(void *pvParameters) {
  uint32_t nextFrameDue = millis();

  while (true) {
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

    char filename[48];
    snprintf(filename, sizeof(filename), "/dog_%lu.jpg", (unsigned long)millis());
    File file = SD_MMC.open(filename, FILE_WRITE);
    if (file) {
      file.write(fb->buf, fb->len);
      file.close();
    } else {
      Serial.printf("Failed to open %s for writing\n", filename);
    }

    frameCounter++;
    if (apWindowOpen && (frameCounter % PREVIEW_FPS_DIVISOR == 0)) {
      publishPreviewFrame(fb);
    }

    esp_camera_fb_return(fb);

    nextFrameDue += CAPTURE_INTERVAL_MS;
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
// never touches the camera driver directly.
// ======================================================
static esp_err_t stream_handler(httpd_req_t *req) {
  esp_err_t res = httpd_resp_set_type(req, _STREAM_CONTENT_TYPE);
  if (res != ESP_OK) {
    return res;
  }

  char part_buf[64];
  while (apWindowOpen) {
    if (xSemaphoreTake(previewReady, pdMS_TO_TICKS(2000)) != pdTRUE) {
      continue; // no new frame yet, keep waiting while the window is open
    }

    if (xSemaphoreTake(previewMutex, portMAX_DELAY) != pdTRUE) {
      break;
    }
    size_t len = previewLen;
    uint8_t *copy = len ? (uint8_t *)malloc(len) : NULL;
    if (copy) {
      memcpy(copy, previewBuf, len);
    }
    xSemaphoreGive(previewMutex);

    if (!copy) {
      continue;
    }

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

void startCameraServer() {
  httpd_config_t config = HTTPD_DEFAULT_CONFIG();
  config.server_port = 80;

  httpd_uri_t stream_uri = {
    .uri = "/stream",
    .method = HTTP_GET,
    .handler = stream_handler,
    .user_ctx = NULL
  };

  if (httpd_start(&stream_httpd, &config) == ESP_OK) {
    httpd_register_uri_handler(stream_httpd, &stream_uri);
  }
}

// ======================================================
// Shut the AP + HTTP server down once the preview window elapses
// ======================================================
void closeApWindow() {
  apWindowOpen = false;
  xSemaphoreGive(previewReady); // wake the stream handler so it notices the window closed

  if (stream_httpd) {
    httpd_stop(stream_httpd);
    stream_httpd = NULL;
  }
  WiFi.softAPdisconnect(true);
  WiFi.mode(WIFI_OFF);
  Serial.println("AP window elapsed -- Wi-Fi and preview stream shut down. SD recording continues.");
}

void setup() {
  Serial.begin(115200);
  Serial.println();

  pinMode(STATUS_LED_PIN, OUTPUT);
  digitalWrite(STATUS_LED_PIN, HIGH); // solid on while the AP window is open

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

  WiFi.softAP(AP_SSID, AP_PASSWORD);
  Serial.print("Access point \"");
  Serial.print(AP_SSID);
  Serial.print("\" up at ");
  Serial.println(WiFi.softAPIP());
  startCameraServer();
  Serial.println("Preview stream: http://192.168.4.1/stream");

  xTaskCreatePinnedToCore(captureTask, "captureTask", 8192, NULL, 1, NULL, 1);
}

void loop() {
  if (apWindowOpen && millis() >= AP_WINDOW_MS) {
    closeApWindow();
    digitalWrite(STATUS_LED_PIN, LOW);
  }

  // Slow heartbeat blink once recording-only (post-AP) to show the board
  // is alive; fast blink instead if the SD card has filled up.
  static uint32_t lastBlink = 0;
  if (!apWindowOpen) {
    uint32_t period = sdFull ? 200 : 2000;
    if (millis() - lastBlink >= period) {
      lastBlink = millis();
      digitalWrite(STATUS_LED_PIN, !digitalRead(STATUS_LED_PIN));
    }
  }

  delay(50);
}
