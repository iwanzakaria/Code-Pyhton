# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

The main program is a single-file Python application, `CCTV_COUNT_v10_INIT_GERAK.py`. It counts cars and motorcycles that cross user-drawn zones in a CCTV stream for Bapenda, using YOLOv8 (Ultralytics) with ByteTrack. Code, comments, console output and generated files are written in **Indonesian**, so keep new code and messages in that style.

`testkoneksi.py` is a separate, scheduled script. It loads `log_kendaraan_terhitung.csv` and `log_capture_kendaraan.xlsx` into Oracle using `oracledb` in thick mode with Instant Client 19c, plus `pandas`, `openpyxl` and `schedule`. If you change the CSV or Excel log schema in the counter, update this script's parsers as well.

The module docstring at the top of the counter is its user manual. It still refers to the script as `traffic_counter_v5.py`; the real filename is `CCTV_COUNT_v10_INIT_GERAK.py`.

Several people commit to this repository directly; there is a GitHub remote with a teammate's fork. Runtime outputs are committed too: logs, captures, reports and `.pt` weights. Before editing, check `git log` and the current code, because the logic changes between sessions.

## Running

There is no build system, requirements file, test suite or linter config. The counter depends on `opencv-python`, `numpy`, `requests` and `ultralytics`. `XlsxWriter` is optional; without it the Excel capture log is skipped and a warning is printed.

```
python CCTV_COUNT_v10_INIT_GERAK.py                 # prompts for stream URL, then metadata (NOP, CCTV_ID, NAMA_OP, ALAMAT_OP), then ROI drawing
python CCTV_COUNT_v10_INIT_GERAK.py "<url|uuid>"    # use the given stream source
python CCTV_COUNT_v10_INIT_GERAK.py --report [id]   # regenerate today's HTML report(s) from the CSV log without opening a camera
```

A run is interactive: it needs a GUI (`cv2.imshow`), stdin input and a live stream. The model is set by `MODEL_NAME`, currently `yolov8m.pt`. To check that the file still parses without running it, use `python -m py_compile CCTV_COUNT_v10_INIT_GERAK.py`.

## Architecture

**Process and thread layout.** Each stage runs separately so the OpenCV window never shows "Not Responding":
- `stream_worker` runs in its own process. It opens the stream (Jasnita API token, RTSP, RTMP or HTTP), refreshes the Jasnita token every `TOKEN_REFRESH_SEC`, retries with backoff, and logs connect and disconnect events to `log_koneksi_stream.csv`.
- `inference_worker` runs in its own process. It runs YOLO `model.track()` with ByteTrack, using a config written to `bytetrack_traffic.yaml` from `TRACKER_CFG`, and sends back only plain numpy arrays.
- `run()` is the main process and GUI loop. It handles per-track state, counting decisions, drawing and keyboard input.
- `writer_worker` is a thread that does all disk I/O: capture JPGs, CSV logs, the Excel workbook and the HTML report. The GUI loop sends tuples (`"log_event"`, `"summary"`, `"report"`) to it through `write_queue`, and `None` stops it. Never do disk I/O directly in the GUI loop.
- The inter-process queues are `maxsize=1` and use `_push_latest`, so stale frames are dropped rather than queued.

**Counting logic.** All of it lives inline in `run()` and is backed by `TrackState`. The user draws three polygons:
- yellow: initialization
- green: the motorcycle counting line
- red: the car, bus and truck counting line

Vehicles must go in the order object → yellow → green/red. The camera feeds have **low FPS**, so vehicles can move a long way between detections. Many of the checks below interpolate along the path between two frames for that reason.

1. **Pre-yellow history.** `pre` holds detections that have not touched yellow yet. `pre_observe` records them; they get no `TrackState`. When one touches yellow, `seed_from_pre` gives the new track its earlier positions, class votes and previous bbox. `pre_target` remembers which counting areas the vehicle's feet touched *before* yellow; those classes are never counted, because the order was reversed.
2. **Initialization.** `update_yellow_init` stores the yellow entry point in foot (ground) coordinates. Contact comes from the feet or wheels, the bottom `INIT_FOOT_BAND_FRAC` band of the bbox, or the foot path between frames. If only the upper body overlaps yellow, the vehicle needs `INIT_BODY_MIN_HITS` observations plus forward motion. A parked vehicle is not initialized.
3. **Parked and stationary filtering.** This uses `update_motion_state`, `park_memory` (parked locations kept independent of ByteTrack IDs, `PARK_MEMORY_*`), `motion_confirmed`, and an in-loop spread re-check.
4. **Count.** `directional_motion_ok` must pass (consistent motion along the yellow→target direction, `DIRECTIONAL_MOTION_*`, `MIN_YELLOW_TO_TARGET_SEC`). Then `vehicle_touches_count_line` must hit the target polygon edge, and travel since the yellow entry must be at least `MIN_TRAVEL_PX`. Each tracker ID counts at most once (`counted_ids`), and `find_relink` carries state across ByteTrack ID switches.

Debug mode (key `d`) shows PARKIR, GERAK and INIT status labels on each vehicle. Counters reset at midnight.

**Tuning.** Behavior is controlled by the module-level constants in the "KONFIGURASI" section at the top of the file. Prefer changing those over hardcoding values in the logic.

**Outputs** go next to the script (`BASE_DIR`):
- `log_kendaraan_terhitung.csv`: the main event log, using the Bapenda field schema from `ensure_event_log_schema`.
- `_capture_index.csv`
- `log_capture_kendaraan.xlsx`: images are embedded with XlsxWriter.
- `ringkasan_hitungan.csv`
- `capture/<CCTV_ID>/<class>/<date>/*.jpg`
- `laporan/laporan_<CCTV_ID>_<date>.html`

## Notes

- Jasnita credentials are hardcoded at the top of the file as `JASNITA_USER` / `JASNITA_PASS`. Only `JASNITA_URL` reads from the environment.
- Keep `mp.freeze_support()` and the `if __name__ == "__main__"` guard. Windows multiprocessing uses spawn, so worker functions must stay at module top level and be picklable.
- `OPENCV_FFMPEG_CAPTURE_OPTIONS` is set before `import cv2` on purpose. Do not reorder these imports.
