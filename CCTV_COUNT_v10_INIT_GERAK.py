"""
CCTV Bapenda - AI Traffic Counting v5
-------------------------------------
Saat program mulai: (1) masukkan URL stream, (2) gambar 3 area langsung di layar.
Area SELALU direset dan digambar ulang setiap program dijalankan (tidak disimpan):
    1) AREA INISIALISASI (KUNING)
    2) AREA COUNTING MOTOR (HIJAU)
    3) AREA COUNTING MOBIL (MERAH)
Aturan:
  * Kendaraan harus lebih dulu menyentuh area KUNING (inisialisasi). Inisialisasi sah bila KAKI/roda (bidang
    tanah) menyentuh atau melintasi kuning -- termasuk lompatan antar-frame di FPS rendah --, atau bila hanya
    BADAN yang menyentuh kuning tetapi kendaraan terbukti bergerak maju ke arah hijau/merah. Kendaraan yang
    kakinya sudah menyentuh hijau/merah SEBELUM kuning (arah balik) tidak dihitung untuk kelas tersebut.
  * Counting tidak memakai bagian dalam/border area sebagai indikator. Motor dihitung saat BADAN kendaraan
    menyentuh/menyeberangi GARIS HIJAU; mobil/bus/truk saat RODA atau BADAN kendaraan
    menyentuh/menyeberangi GARIS MERAH.
  * Deteksi YOLO berjalan di PROSES TERPISAH dari jendela tampilan, supaya jendela tidak "Not Responding"
    walau ada banyak kendaraan sekaligus / deteksi sedang berat.
  * Setiap ID hanya dihitung SATU KALI selama program berjalan.
  * Arah masuk otomatis: motor = pusat kuning -> pusat hijau, mobil = pusat kuning -> pusat merah.
  * Gambar area merah agar mencakup titik tempat mobil berhenti/parkir (titik kaki = titik magenta
    di bawah bounding box) dan JANGAN menyentuh badan jalan raya.

Cara pakai:
  python traffic_counter_v5.py                      -> program menanyakan URL stream (Enter = URL default)
  python traffic_counter_v5.py "<URL stream>"        -> langsung pakai URL tersebut
URL yang diterima:
  - https://panel.jastrak.id/dashboard/video-stream?id=<UUID>   (ID diambil dari parameter id)
  - <UUID> saja
  - rtsp:// , rtmp:// , atau http(s):// stream langsung (tanpa login Jasnita)

Saat menggambar ROI:
  Klik kiri = tambah titik | Klik kanan / Backspace = undo titik
  ENTER = selesai satu area (minimal 3 titik) | C = ulang area ini
  F = ambil frame baru | Q / Esc = batal
  Setelah 3 area lengkap: ENTER = mulai | R = ulang semua | M = ulang area merah saja

Output (disimpan di folder yang sama dengan script):
  capture/<CCTV_ID>/<mobil|motor>/<tanggal>/HHMMSS_ID<n>_track<id>.jpg   foto tiap kendaraan terhitung
  log_kendaraan_terhitung.csv               log utama format Excel Bapenda (ID/NOP/CCTV_ID/NAMA_OP/.../VENDOR)
  ringkasan_hitungan.csv                    id CCTV / URL stream, jumlah mobil & motor (tiap 60 dtk jika berubah,
                                            dan 1 baris saat program berhenti)
  log_koneksi_stream.csv                    riwayat sambung/terputus/refresh stream + durasi -> utk cari pola jam sering putus
  laporan/laporan_<id_cctv>_<tanggal>.html  dashboard HTML per CCTV (dibuat ulang tiap ringkasan & saat keluar)

Laporan HTML (tanpa menjalankan kamera):
  python traffic_counter_v5.py --report                -> buat laporan utk semua ID CCTV yang ada di log hari ini
  python traffic_counter_v5.py --report "<id_cctv>"     -> buat laporan utk satu ID CCTV saja

Saat counting:
  q = keluar | d = tampilkan semua deteksi (debug) | r = edit/gambar ulang ROI
  z = +1 motor MANUAL (ground truth) | x = +1 mobil MANUAL (ground truth)
"""
import base64
import csv
import html
import multiprocessing as mp
import os
import re
import sys
import threading
import time
from collections import Counter, deque
from pathlib import Path
from queue import Empty, Queue
from urllib.parse import parse_qs, urlparse

# Opsi FFmpeg harus dipasang sebelum backend FFmpeg OpenCV dipakai.
os.environ.setdefault(
    "OPENCV_FFMPEG_CAPTURE_OPTIONS",
    "reconnect;1|"
    "reconnect_streamed;1|"
    "reconnect_at_eof;1|"
    "reconnect_on_network_error;1|"
    "reconnect_delay_max;5|"
    "rw_timeout;15000000",
)

import cv2
import numpy as np
import requests

# ==========================================
# 0. KONFIGURASI
# ==========================================
JASNITA_USER = 'sby@jasnita.co.id'
JASNITA_PASS = 'surabaya2025!!'
JASNITA_URL = os.getenv("JASNITA_URL", "https://panel.jastrak.id/api")

OPEN_TIMEOUT_MS = 15000        # batas waktu MEMBUKA koneksi stream
READ_TIMEOUT_MS = 15000        # batas waktu MEMBACA setiap frame
TOKEN_REFRESH_SEC = 240        # token Jasnita berlaku 300 dtk -> sambung ulang di 240 dtk (sebelum kedaluwarsa)

# Retry pembukaan stream. Token yang sama dicoba beberapa kali dulu sebelum meminta token baru,
# agar API Jasnita tidak dihantam login berulang saat koneksi kamera sedang lambat.
STREAM_OPEN_ATTEMPTS = 3
STREAM_OPEN_DELAY_SEC = 1.0
STREAM_OPEN_BACKOFF_MAX = 6.0
STREAM_TOKEN_RETRIES = 3
STREAM_RETRY_DELAY_SEC = 3.0
STREAM_HTTP_PROBE_TIMEOUT = (5, 8)
STREAM_USER_AGENT = "Mozilla/5.0 CCTV-Bapenda-TrafficCounter/5"
# URL default jika Anda hanya menekan Enter saat diminta input URL stream
DEFAULT_STREAM_URL = 'https://panel.jastrak.id/dashboard/video-stream?id=a02e5078-f6fb-463b-98bc-02fc3ce8be28'
TIMEOUT = 30

FRAME_W, FRAME_H = 1024, 576

# Model & deteksi
MODEL_NAME = "yolov8m.pt"      # model medium: kompromi akurasi/FPS yang masih realistis untuk CCTV
IMGSZ = 896                     # naikkan ukuran inferensi agar motor kecil/jauh lebih mudah terbaca
DET_CONF = 0.05                 # beri ByteTrack kandidat low-score; keputusan akhir tetap lewat ROI + tracking
MAX_DET = 200                   # jangan membuang motor hanya karena jumlah objek dalam satu frame cukup banyak
TARGET_CLASSES = [2, 3, 5, 7]  # 2 car, 3 motorcycle, 5 bus, 7 truck

# Logika counting
MIN_TRAVEL_PX = 12              # hanya sebagai filter arah; indikator counting UTAMA adalah sentuhan garis ROI
MIN_TRACK_FRAMES = 2            # motor yang singkat terlihat tetap bisa diinisialisasi
MIN_YELLOW_HITS = 1              # satu bukti kontak badan/anchor dengan area kuning sudah cukup
# Inisialisasi kuning berbasis BIDANG TANAH (kaki/roda), bukan sekadar bbox: bagian atas bbox kendaraan yang
# berada di belakang/di jalan sebelah sering 'menyentuh' kuning secara perspektif padahal kendaraannya tidak.
INIT_FOOT_BAND_FRAC = 0.25       # pita kaki = 25% bawah bbox; menyentuh kuning = kontak tanah sah
INIT_BODY_MIN_HITS = 2           # hanya BADAN yang menyentuh kuning (kaki tidak) -> minimal 2 observasi ...
INIT_BODY_FWD_PX = 30.0          # ... DAN sudah maju searah kuning -> hijau/merah minimal max(30 px, ...
INIT_BODY_FWD_RATIO = 0.50       # ... 50% lebar bbox) sejak pertama terlihat (kendaraan keluar dari halangan)
PRE_TRACK_KEEP_SEC = 3.0         # riwayat deteksi SEBELUM menyentuh kuning disimpan selama ini (detik)
MIN_CLASS_CONFIRM_FRAMES = 2     # class minimal 2 observasi sebelum counting
LOST_AFTER = 1.2                 # jangan terlalu cepat melepas track saat inference tidak setiap frame
RELINK_TIME = 4.5                # pertahankan kandidat lebih lama saat ID ByteTrack berganti
RELINK_DIST = 140                # toleransi perpindahan posisi saat ID berubah

# Counting berbasis GARIS, bukan area bagian dalam ROI.
LINE_TOUCH_TOLERANCE_PX = 5.0    # toleransi tipis untuk perbedaan bounding box/pixel CCTV
LINE_CROSS_SAMPLES = 20          # sampling lintasan antar-frame agar garis tipis tidak terlewati
WHEEL_INSET = 0.20                # posisi roda: kiri/tengah/kanan di bawah bounding box
MIN_ENTRY_VECTOR_PX = 40          # jarak minimum antar pusat area untuk menentukan arah masuk

# Filter otomatis kendaraan DIAM/PARKIR. Tanda X pada contoh capture hanya anotasi evaluasi;
# TIDAK ada tombol X di aplikasi. Objek parkir/stasioner tidak boleh dihitung atau memicu capture.
PARK_WINDOW_SEC = 3.0
PARK_MIN_SAMPLES = 4
PARK_MIN_RADIUS_PX = 12.0
PARK_RADIUS_RATIO = 0.18
PARK_RELEASE_PX = 45.0
PARK_RELEASE_RATIO = 0.60
MOTION_CONFIRM_MIN_SAMPLES = 4
MOTION_CONFIRM_MIN_STEP_PX = 5.0
MOTION_CONFIRM_RATIO = 0.20
PARK_MEMORY_SEC = 1800.0
PARK_MEMORY_IOU_MIN = 0.30
PARK_MEMORY_DIST_RATIO = 0.75
PARK_MEMORY_MIN_STABLE_SEC = 1.5
# Gerakan harus nyata dan konsisten menuju garis counting, bukan jitter bounding-box.
DIRECTIONAL_MOTION_WINDOW_SEC = 2.5
DIRECTIONAL_MOTION_MIN_NET_PX = 30.0
DIRECTIONAL_MOTION_MIN_STEP_PX = 5.0
DIRECTIONAL_MOTION_MIN_POSITIVE_RATIO = 0.75
DIRECTIONAL_MOTION_MIN_POSITIVE_STEPS = 3
MIN_YELLOW_TO_TARGET_SEC = 1.0

# ROI
COLOR_YELLOW = (0, 255, 255)
COLOR_GREEN = (0, 255, 0)
COLOR_RED = (0, 0, 255)
ROI_STEPS = [
    ("AREA INISIALISASI (KUNING)", COLOR_YELLOW),
    ("AREA COUNTING MOTOR (HIJAU)", COLOR_GREEN),
    ("AREA COUNTING MOBIL (MERAH)", COLOR_RED),
]

# Output (semua disimpan di folder yang sama dengan script)
BASE_DIR = Path(__file__).resolve().parent
CSV_DELIMITER = ","                                          # Excel versi Indonesia biasanya butuh ";"
EVENT_LOG_FILE = BASE_DIR / "log_kendaraan_terhitung.csv"    # format log utama untuk Excel sesuai field Bapenda
CAPTURE_INDEX_FILE = BASE_DIR / "_capture_index.csv"       # index internal capture -> ID log, tidak dipakai sebagai log utama
CAPTURE_EXCEL_FILE = BASE_DIR / "log_capture_kendaraan.xlsx" # Excel capture dengan gambar tertanam
CAPTURE_EXCEL_SHEET = "CAPTURE"
CAPTURE_EXCEL_IMAGE_W = 320
CAPTURE_EXCEL_IMAGE_H = 180
SUMMARY_FILE = BASE_DIR / "ringkasan_hitungan.csv"           # ringkasan jumlah mobil & motor per sumber stream
NETWORK_LOG_FILE = BASE_DIR / "log_koneksi_stream.csv"       # riwayat sambung/putus stream, utk diagnosis pola putus
SUMMARY_INTERVAL_SEC = 60                                    # ringkasan berkala (hanya jika angka berubah)
CAPTURE_DIR = BASE_DIR / "capture"                           # capture/<id_cctv>/<kelas>/<tanggal>/HHMMSS_idN.jpg
# PENTING: sebelumnya hanya "motor" -> itu sebabnya mobil yang terhitung tidak pernah ter-capture.
CAPTURE_CLASSES = ("motor", "car")                           # kelas yang di-capture; kosongkan salah satu utk mematikannya
CAPTURE_FULL_FRAME = False                                   # True = simpan juga frame penuh + bounding box
CAPTURE_SCALE = 2.2                                          # pembesaran dasar crop
CAPTURE_MIN_LONG_SIDE = 640                                  # sisi terpanjang minimum setelah resize (px)
CAPTURE_UPSCALE_MAX = 3.5                                    # batas maksimum pembesaran
CAPTURE_JPEG_QUALITY = 97                                    # kualitas JPG capture (0-100)
CAPTURE_SHARPEN_AMOUNT = 0.65                                # sharpening ringan
CAPTURE_PAD_RATIO = 0.18                                     # ruang tambahan di sekitar kendaraan
REPORT_THUMB_W = 150                                         # lebar thumbnail HTML
REPORT_THUMB_H = 108                                         # tinggi thumbnail HTML
REPORT_DIR = BASE_DIR / "laporan"                            # laporan/laporan_<id_cctv>_<tanggal>.html

TRACKER_CFG = """tracker_type: bytetrack
track_high_thresh: 0.20
track_low_thresh: 0.05
new_track_thresh: 0.25
track_buffer: 90
match_thresh: 0.85
fuse_score: True
"""

LABEL = {"car": "Mobil", "motor": "Motor"}
LOG_JENIS = {"car": "MOBIL", "motor": "MOTOR"}
LOG_VENDOR = "BAPENDA"


# ==========================================
# 1. API & PEKERJA JARINGAN
# ==========================================
def login(session: requests.Session) -> str:
    res = session.post(
        f"{JASNITA_URL}/login",
        json={"email": JASNITA_USER, "password": JASNITA_PASS},
        timeout=TIMEOUT,
    )
    res.raise_for_status()
    token = res.json().get("token")
    if not token:
        raise RuntimeError("Login gagal: token kosong")
    return token


def get_stream_url(display_id: str) -> str:
    session = requests.Session()
    session.headers["Authorization"] = f"Bearer {login(session)}"
    res = session.post(f"{JASNITA_URL}/video-stream-url", json={"id": display_id}, timeout=TIMEOUT)
    res.raise_for_status()
    data = res.json()
    stream_token = data.get("token")
    if not stream_token:
        raise RuntimeError("Stream URL gagal: token kosong")
    print(f"[INFO-NETWORK] Token stream didapat. Masa aktif: {data.get('expires_in')} detik")
    return f"{JASNITA_URL}/video-stream?token={stream_token}"


class _StreamRefresh(Exception):
    """Bukan error: dipakai utk menutup & membuka ulang stream sebelum token 300 dtk kedaluwarsa."""


def _push_latest(q, frame):
    if q.full():
        try:
            q.get_nowait()
        except Exception:
            pass
    try:
        q.put_nowait(frame)
    except Exception:
        pass


def log_network_event(label, event, detail="", connected_since=None, disconnected_since=None):
    """Catat riwayat sambung/putus stream ke CSV, supaya pola putusnya (jam berapa, seberapa sering,
    berapa lama tiap kali putus) bisa dianalisis belakangan tanpa harus menonton terminal terus-menerus."""
    now = time.time()
    durasi_konek = f"{now - connected_since:.1f}" if connected_since else ""
    downtime = f"{now - disconnected_since:.1f}" if disconnected_since else ""
    append_csv(
        NETWORK_LOG_FILE,
        ["waktu", "sumber", "kejadian", "durasi_terhubung_detik", "downtime_sebelumnya_detik", "detail"],
        [fmt_time(now), label, event, durasi_konek, downtime, detail],
    )


def _set_capture_timeouts(cap):
    """Set timeout/buffer bila backend OpenCV mendukung property tersebut."""
    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    except Exception:
        pass
    for prop_name, value in (
        ("CAP_PROP_OPEN_TIMEOUT_MSEC", OPEN_TIMEOUT_MS),
        ("CAP_PROP_READ_TIMEOUT_MSEC", READ_TIMEOUT_MS),
    ):
        if hasattr(cv2, prop_name):
            try:
                cap.set(getattr(cv2, prop_name), value)
            except Exception:
                pass


def _open_capture_once(url):
    """Buka stream dengan timeout melalui constructor jika tersedia; fallback ke set()."""
    params = []
    if hasattr(cv2, "CAP_PROP_OPEN_TIMEOUT_MSEC"):
        params += [cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, OPEN_TIMEOUT_MS]
    if hasattr(cv2, "CAP_PROP_READ_TIMEOUT_MSEC"):
        params += [cv2.CAP_PROP_READ_TIMEOUT_MSEC, READ_TIMEOUT_MS]

    last_error = None
    # Coba FFmpeg dulu karena stream Jasnita/HTTP biasanya paling konsisten lewat FFmpeg.
    backends = [cv2.CAP_FFMPEG]
    if hasattr(cv2, "CAP_ANY") and cv2.CAP_ANY not in backends:
        backends.append(cv2.CAP_ANY)

    for api in backends:
        cap = None
        try:
            # Beberapa build OpenCV tidak menerima parameter constructor. Coba versi lengkap dulu.
            if params:
                try:
                    cap = cv2.VideoCapture(url, api, params)
                except (TypeError, cv2.error):
                    cap = cv2.VideoCapture(url, api)
            else:
                cap = cv2.VideoCapture(url, api)

            _set_capture_timeouts(cap)
            if cap.isOpened():
                return cap
            last_error = f"backend={api} tidak berhasil membuka URL"
        except Exception as exc:
            last_error = f"backend={api}: {exc}"
        finally:
            if cap is not None and not cap.isOpened():
                try:
                    cap.release()
                except Exception:
                    pass

    raise RuntimeError(last_error or "OpenCV tidak dapat membuka stream")


def _probe_http_stream(url):
    """Probe ringan untuk membedakan masalah token/API HTTP vs masalah decoder OpenCV.
    Hanya dipanggil setelah pembukaan OpenCV gagal, jadi tidak menambah beban pada koneksi normal.
    """
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        return ""
    try:
        with requests.get(
            url,
            stream=True,
            timeout=STREAM_HTTP_PROBE_TIMEOUT,
            headers={"User-Agent": STREAM_USER_AGENT},
            allow_redirects=True,
        ) as res:
            ctype = (res.headers.get("Content-Type") or "").split(";", 1)[0]
            return f"HTTP {res.status_code}, content-type={ctype or '-'}, final_url={res.url[:120]}"
    except requests.RequestException as exc:
        return f"probe HTTP gagal: {exc}"


def open_stream_with_retries(url):
    """Buka stream beberapa kali memakai URL/token yang sama sebelum menyerah."""
    last_detail = ""
    for attempt in range(1, STREAM_OPEN_ATTEMPTS + 1):
        cap = None
        try:
            cap = _open_capture_once(url)
            if cap is not None and cap.isOpened():
                return cap, ""
        except Exception as exc:
            last_detail = str(exc)
        finally:
            if cap is not None and not cap.isOpened():
                try:
                    cap.release()
                except Exception:
                    pass

        if attempt < STREAM_OPEN_ATTEMPTS:
            delay = min(STREAM_OPEN_BACKOFF_MAX, STREAM_OPEN_DELAY_SEC * (2 ** (attempt - 1)))
            print(f"[WARN-NETWORK] Open stream gagal (attempt {attempt}/{STREAM_OPEN_ATTEMPTS}) -> retry {delay:.1f}s")
            time.sleep(delay)

    probe = _probe_http_stream(url)
    if probe:
        last_detail = f"{last_detail}; {probe}" if last_detail else probe
    return None, last_detail or "OpenCV gagal membuka stream"


def stream_worker(source, frame_queue):
    kind, value = source
    label = describe_source(source)["sumber"]
    failures = 0
    token_url = None
    token_obtained_at = 0.0
    disconnected_at = None
    connected_once = False

    while True:
        cap = None
        opened_at = 0.0
        try:
            # Untuk Jasnita, satu token dipakai ulang untuk beberapa percobaan open.
            # Token baru hanya diminta setelah retry token yang sama benar-benar gagal.
            if kind == "jasnita":
                token_expired = token_url is None or (time.time() - token_obtained_at) >= (TOKEN_REFRESH_SEC - 15)
                if token_expired:
                    token_url = get_stream_url(value)
                    token_obtained_at = time.time()
            else:
                token_url = value
                token_obtained_at = time.time()

            cap, detail = open_stream_with_retries(token_url)
            if cap is None:
                failures += 1
                # Open failure belum berarti stream sebelumnya putus. Jangan mengisi downtime di sini
                # karena pada startup memang belum pernah connected.
                print(f"[WARN-NETWORK] Gagal membuka stream: {detail} (siklus {failures})")

                # Bila token Jasnita sudah beberapa kali gagal, paksa refresh token sebelum backoff panjang.
                if kind == "jasnita" and failures >= STREAM_TOKEN_RETRIES:
                    print("[WARN-NETWORK] Token/endpoint gagal dibuka beberapa kali -> minta token baru.")
                    token_url = None
                    token_obtained_at = 0.0
                    failures = 0
                    wait = STREAM_RETRY_DELAY_SEC
                else:
                    wait = min(STREAM_OPEN_BACKOFF_MAX, STREAM_RETRY_DELAY_SEC * max(1, failures))

                log_network_event(label, "gagal_buka", detail=detail)
                time.sleep(wait)
                continue

            failures = 0
            opened_at = time.time()
            connected_once = True
            log_network_event(label, "terhubung", disconnected_since=disconnected_at)
            disconnected_at = None
            print("[INFO-NETWORK] Stream berhasil dibuka.")

            while True:
                # Refresh token hanya setelah stream baru berhasil dibuka; bila refresh gagal,
                # stream lama tidak langsung dibuang sebelum ada kesempatan retry.
                if kind == "jasnita" and time.time() - token_obtained_at >= TOKEN_REFRESH_SEC:
                    raise _StreamRefresh()

                ok, frame = cap.read()
                if not ok:
                    raise RuntimeError("Stream terputus / frame tidak terbaca")
                failures = 0
                _push_latest(frame_queue, frame)

        except _StreamRefresh:
            disconnected_at = time.time() if connected_once else None
            log_network_event(label, "refresh_terjadwal", connected_since=opened_at)
            print("[INFO-NETWORK] Refresh token terjadwal -> membuka koneksi baru ...")
            # Paksa token baru pada iterasi berikutnya. Tidak melakukan sleep panjang.
            token_url = None
            token_obtained_at = 0.0
        except Exception as e:
            failures += 1
            disconnected_at = time.time() if connected_once else None
            log_network_event(label, "terputus", detail=str(e), connected_since=opened_at if opened_at else None)
            wait = min(15.0, max(1.0, 2.0 * failures))
            print(f"[WARN-NETWORK] {e} (reconnect {failures}) - ulang dalam {wait:.1f}s")
            time.sleep(wait)
        finally:
            if cap is not None:
                try:
                    cap.release()
                except Exception:
                    pass


def inference_worker(frame_queue, result_queue, tracker_cfg_path):
    """
    PROSES TERPISAH: menjalankan YOLO + ByteTrack. Sengaja dipisah dari proses tampilan (run()) supaya
    jendela cv2.imshow tidak pernah menunggu YOLO -> tidak "Not Responding" walau deteksi berat/banyak objek.
    Hasil deteksi dikirim sebagai data polos (bukan objek Ultralytics) agar bisa lewat multiprocessing.Queue.
    """
    from ultralytics import YOLO
    print(f"[INFO] Memuat model {MODEL_NAME} ...")
    model = YOLO(MODEL_NAME)
    print("[INFO] Model siap. Deteksi berjalan di proses terpisah dari tampilan.")
    while True:
        try:
            raw = frame_queue.get(timeout=5.0)
        except Empty:
            continue
        raw = cv2.resize(raw, (FRAME_W, FRAME_H))
        results = model.track(raw, persist=True, classes=TARGET_CLASSES, tracker=tracker_cfg_path,
                              imgsz=IMGSZ, conf=DET_CONF, max_det=MAX_DET, verbose=False)
        boxes = results[0].boxes
        if boxes is not None and boxes.id is not None:
            payload = {
                "xyxy": boxes.xyxy.cpu().numpy(),
                "ids": boxes.id.int().cpu().numpy(),
                "cls": boxes.cls.int().cpu().numpy(),
                "conf": boxes.conf.cpu().numpy(),
            }
        else:
            payload = None
        _push_latest(result_queue, (raw, payload, time.time()))


UUID_RE = re.compile(r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}")


def parse_stream_input(text):
    """Return ("jasnita", display_id) atau ("direct", url); None jika tidak valid."""
    text = (text or "").strip().strip('"').strip("'")
    if not text:
        return None
    parsed = urlparse(text)
    qid = (parse_qs(parsed.query).get("id") or [""])[0]
    if UUID_RE.fullmatch(qid):
        return ("jasnita", qid)
    if UUID_RE.fullmatch(text):
        return ("jasnita", text)
    if parsed.scheme in ("rtsp", "rtmp", "http", "https") and parsed.netloc:
        if "jastrak.id" in parsed.netloc:
            return None  # halaman panel Jasnita tanpa parameter id
        return ("direct", text)
    return None


def ask_stream_source():
    """URL dari argumen command line, atau ditanyakan saat program mulai."""
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    text = args[0] if args else None
    while True:
        if text is None:
            try:
                text = input(f"\nMasukkan URL stream CCTV\n(Enter = default: {DEFAULT_STREAM_URL})\n> ")
            except EOFError:
                print("Tidak ada input. Jalankan dari terminal atau beri URL sebagai argumen.")
                sys.exit(1)
            if not text.strip():
                text = DEFAULT_STREAM_URL
        source = parse_stream_input(text)
        if source:
            return source
        print("[ERROR] URL tidak valid. Contoh: https://panel.jastrak.id/dashboard/video-stream?id=<UUID>")
        text = None


def grab_frame(frame_queue):
    """Ambil satu frame dari stream (menunggu sampai ada)."""
    while True:
        try:
            frame = frame_queue.get(timeout=5.0)
        except Empty:
            print("[INFO] Menunggu frame dari stream ...")
            continue
        return cv2.resize(frame, (FRAME_W, FRAME_H))


# ==========================================
# 2. GEOMETRI & STATE TRACK
# ==========================================
def to_poly(pts):
    return np.array(pts, np.int32).reshape((-1, 1, 2))


def inside(poly, pt):
    return cv2.pointPolygonTest(poly, (float(pt[0]), float(pt[1])), False) >= 0


def segment_hits(poly, p0, p1, samples=12):
    """True jika ruas p0->p1 melewati poligon (menangani kendaraan yang 'melompati' zona tipis)."""
    for t in np.linspace(0.0, 1.0, samples):
        if inside(poly, (p0[0] + (p1[0] - p0[0]) * t, p0[1] + (p1[1] - p0[1]) * t)):
            return True
    return False


def touches(poly, prev_pt, pt):
    """Kontak/gerak melalui POLYGON (dipakai untuk kompatibilitas fungsi lama)."""
    return inside(poly, pt) or (prev_pt is not None and segment_hits(poly, prev_pt, pt))


def bbox_anchor_points(box):
    """Titik jangkar pada badan kendaraan, terutama bagian bawah."""
    x1, y1, x2, y2 = [float(v) for v in box]
    w = max(1.0, x2 - x1)
    h = max(1.0, y2 - y1)
    xm = (x1 + x2) / 2.0
    return [
        (xm, y2),
        (x1 + 0.25 * w, y2),
        (x2 - 0.25 * w, y2),
        (x1, y2),
        (x2, y2),
        (xm, y1 + 0.55 * h),
        (xm, y1 + 0.35 * h),
        (x1 + 0.20 * w, y1 + 0.60 * h),
        (x2 - 0.20 * w, y1 + 0.60 * h),
    ]


def bbox_edges(box):
    """Empat sisi bounding box sebagai ruas garis."""
    x1, y1, x2, y2 = [float(v) for v in box]
    return [
        ((x1, y1), (x2, y1)),
        ((x2, y1), (x2, y2)),
        ((x2, y2), (x1, y2)),
        ((x1, y2), (x1, y1)),
    ]


def point_segment_distance(p, a, b):
    """Jarak titik ke ruas garis."""
    p = np.asarray(p, dtype=float)
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    ab = b - a
    den = float(np.dot(ab, ab))
    if den <= 1e-9:
        return float(np.linalg.norm(p - a))
    t = float(np.dot(p - a, ab) / den)
    t = max(0.0, min(1.0, t))
    q = a + t * ab
    return float(np.linalg.norm(p - q))


def orientation(a, b, c):
    return ((float(b[0]) - float(a[0])) * (float(c[1]) - float(a[1]))
            - (float(b[1]) - float(a[1])) * (float(c[0]) - float(a[0])))


def on_segment(a, b, p, eps=1e-6):
    return (min(a[0], b[0]) - eps <= p[0] <= max(a[0], b[0]) + eps
            and min(a[1], b[1]) - eps <= p[1] <= max(a[1], b[1]) + eps)


def segments_intersect(a, b, c, d, eps=1e-6):
    """Intersection ruas garis 2D, termasuk kasus garis saling menyentuh."""
    o1, o2 = orientation(a, b, c), orientation(a, b, d)
    o3, o4 = orientation(c, d, a), orientation(c, d, b)
    if ((o1 > eps and o2 < -eps) or (o1 < -eps and o2 > eps)) and ((o3 > eps and o4 < -eps) or (o3 < -eps and o4 > eps)):
        return True
    if abs(o1) <= eps and on_segment(a, b, c):
        return True
    if abs(o2) <= eps and on_segment(a, b, d):
        return True
    if abs(o3) <= eps and on_segment(c, d, a):
        return True
    if abs(o4) <= eps and on_segment(c, d, b):
        return True
    return False


def poly_edges(poly):
    pts = np.asarray(poly).reshape(-1, 2).astype(float)
    if len(pts) < 2:
        return []
    return [(tuple(pts[i]), tuple(pts[(i + 1) % len(pts)])) for i in range(len(pts))]


def point_near_poly_boundary(poly, pt, tolerance=LINE_TOUCH_TOLERANCE_PX):
    """True hanya jika titik dekat GARIS batas polygon; tidak menganggap seluruh area sebagai hit."""
    return any(point_segment_distance(pt, a, b) <= tolerance for a, b in poly_edges(poly))


def segment_hits_poly_boundary(poly, p0, p1, samples=LINE_CROSS_SAMPLES):
    """True jika gerakan titik menyentuh/menyeberangi GARIS polygon."""
    edges = poly_edges(poly)
    if not edges:
        return False
    if any(point_near_poly_boundary(poly, p, LINE_TOUCH_TOLERANCE_PX) for p in (p0, p1)):
        return True
    if any(segments_intersect(p0, p1, a, b) for a, b in edges):
        return True
    # Fallback untuk video dengan gerak cepat / bounding box yang bergeser besar.
    for t in np.linspace(0.0, 1.0, max(2, int(samples))):
        p = (p0[0] + (p1[0] - p0[0]) * t, p0[1] + (p1[1] - p0[1]) * t)
        if point_near_poly_boundary(poly, p, LINE_TOUCH_TOLERANCE_PX):
            return True
    return False


def bbox_overlaps_polygon(poly, box, prev_box=None):
    """Deteksi bahwa BADAN bounding box menyentuh/beririsan area polygon (untuk INISIALISASI)."""
    edges = poly_edges(poly)
    x1, y1, x2, y2 = [float(v) for v in box]
    corners = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
    if any(inside(poly, p) for p in corners):
        return True
    poly_pts = np.asarray(poly).reshape(-1, 2)
    if any(x1 <= float(px) <= x2 and y1 <= float(py) <= y2 for px, py in poly_pts):
        return True
    box_edges = bbox_edges(box)
    if any(segments_intersect(a, b, c, d) for a, b in box_edges for c, d in edges):
        return True
    # Jika box bergerak menyeberangi ROI antara dua hasil deteksi, tetap anggap kontak terjadi.
    if prev_box is not None:
        prev_edges = bbox_edges(prev_box)
        for (a0, a1), (b0, b1) in zip(prev_edges, box_edges):
            if any(segment_hits_poly_boundary(poly, p0, p1) for p0, p1 in ((a0, a1), (b0, b1))):
                return True
        prev_c = bbox_anchor_points(prev_box)
        cur_c = bbox_anchor_points(box)
        if any(segments_intersect(p0, p1, e0, e1) for p0, p1 in zip(prev_c, cur_c) for e0, e1 in edges):
            return True
    return False


def vehicle_touches_count_line(poly, box, prev_box=None, wheels=None, prev_wheels=None):
    """Counting AKTIF hanya ketika RODA atau BADAN kendaraan menyentuh garis ROI.

    Berbeda dari `inside()`: kendaraan yang sudah berada di dalam polygon tetapi tidak menyentuh
    garis tidak dihitung. Gerakan antar-frame yang menyeberangi garis juga terdeteksi.
    """
    # 1) Roda/titik bawah: paling kuat untuk mobil.
    wheels = wheels or wheel_points(float(box[0]), float(box[2]), float(box[3]))
    if any(point_near_poly_boundary(poly, p) for p in wheels):
        return True
    if prev_wheels is not None:
        for p0, p1 in zip(prev_wheels, wheels):
            if segment_hits_poly_boundary(poly, p0, p1):
                return True

    # 2) Seluruh badan/bounding box: juga berlaku untuk motor, mobil, bus, truk.
    edges = poly_edges(poly)
    if any(segments_intersect(a, b, c, d) for a, b in bbox_edges(box) for c, d in edges):
        return True

    # 3) Bila bbox melewati garis di antara dua frame, tangkap lintasannya.
    if prev_box is not None:
        prev_edges = bbox_edges(prev_box)
        cur_edges = bbox_edges(box)
        for (p0a, p0b), (p1a, p1b) in zip(prev_edges, cur_edges):
            if any(segment_hits_poly_boundary(poly, q0, q1) for q0, q1 in ((p0a, p1a), (p0b, p1b))):
                return True
    return False


def touches_bbox(poly, box, prev_box=None):
    """Kontak badan kendaraan dengan area polygon, untuk inisialisasi yellow.
    Return (hit, anchor).
    """
    cur_pts = bbox_anchor_points(box)
    hit = bbox_overlaps_polygon(poly, box, prev_box)
    if not hit:
        return False, None
    # Pilih anchor yang benar-benar berada di dalam area; fallback ke bottom-center.
    for p in cur_pts:
        if inside(poly, p):
            return True, p
    return True, cur_pts[0]


def wheel_points(x1, x2, y2):
    """Tiga titik perkiraan roda kiri/tengah/kanan pada garis bawah bounding box."""
    w = x2 - x1
    return [(x1 + WHEEL_INSET * w, y2), ((x1 + x2) / 2.0, y2), (x2 - WHEEL_INSET * w, y2)]


def touches_any(poly, prev_pts, pts):
    # Kompatibilitas: fungsi lama sekarang memakai GARIS batas polygon.
    return any(segment_hits_poly_boundary(poly, prev_pts[i] if prev_pts else p, p)
               for i, p in enumerate(pts))


def compute_entry_dir(from_pts, to_pts, fallback=None):
    """Arah masuk = vektor satuan dari pusat area 'from' ke pusat area 'to'.
    Jika kedua pusat terlalu berdekatan (area saling menimpa), pakai fallback (default: ke bawah layar)."""
    v = np.mean(np.array(to_pts, float), axis=0) - np.mean(np.array(from_pts, float), axis=0)
    n = float(np.linalg.norm(v))
    if n >= MIN_ENTRY_VECTOR_PX:
        return v / n
    return fallback if fallback is not None else np.array([0.0, 1.0])


class TrackState:
    def __init__(self, now):
        self.frames = 0
        self.last_seen = now
        self.last_pt = None
        self.last_wheels = None
        self.last_box = None
        self.last_group = None
        self.yellow_pt = None      # titik pertama/bukti terbaik menyentuh area kuning
        # Titik dan waktu kontak kaki kendaraan dengan area KUNING.
        # Dipakai directional_motion_ok() untuk memastikan gerak terjadi setelah inisialisasi.
        self.yellow_foot_pt = None
        self.yellow_foot_t = None
        self.yellow_hits = 0
        self.initialized = False
        self.counted = False
        self.votes = Counter()
        self.hist = deque(maxlen=64)
        self.parked = False
        self.park_anchor = None
        self.motion_confirmed = False
        self.park_memory_id = None
        self.origin_pt = None         # posisi kaki saat pertama terlihat (termasuk riwayat sebelum kuning)
        self.origin_t = None
        self.yellow_body_hits = 0     # observasi di mana HANYA badan (bukan kaki) menyentuh kuning
        self.init_mode = None         # "kaki" / "badan+gerak" -- untuk label debug
        self.pre_target = set()       # kelas yang kakinya menyentuh hijau/merah SEBELUM kuning (arah balik)


def yellow_ground_contact(poly, box, prev_pt, pt):
    """Titik kaki (koordinat tanah) saat MASUK area kuning, atau None bila hanya badan atas yang menyentuh.

    Lintasan kaki antar-frame diperiksa lebih dulu: di FPS rendah kendaraan bisa melompati area kuning dalam satu
    langkah, dan titik masuk pada lintasan lebih akurat daripada posisi kaki sekarang."""
    x1, y1, x2, y2 = [float(v) for v in box]
    if prev_pt is not None:
        for t in np.linspace(0.0, 1.0, max(2, LINE_CROSS_SAMPLES)):
            p = (prev_pt[0] + (pt[0] - prev_pt[0]) * t, prev_pt[1] + (pt[1] - prev_pt[1]) * t)
            if inside(poly, p):
                return p
    if any(inside(poly, w) or point_near_poly_boundary(poly, w) for w in wheel_points(x1, x2, y2)):
        return pt
    band = (x1, y2 - INIT_FOOT_BAND_FRAC * max(1.0, y2 - y1), x2, y2)
    if bbox_overlaps_polygon(poly, band):
        return pt
    return None


def pre_observe(pre, tid, group, pt, box, now, green_roi, red_roi):
    """Riwayat ringan deteksi yang BELUM punya track (belum menyentuh kuning): posisi, kelas, dan apakah kakinya
    sudah menyentuh hijau/merah lebih dulu. Dipakai seed_from_pre() saat objek akhirnya menyentuh kuning."""
    p = pre.get(tid)
    if p is None:
        p = {"hist": deque(maxlen=16), "votes": Counter(), "target": set(), "pt": None, "box": None, "wheels": None}
        pre[tid] = p
    wheels = wheel_points(float(box[0]), float(box[2]), float(box[3]))
    for g_name, g_roi in (("motor", green_roi), ("car", red_roi)):
        if any(inside(g_roi, w) for w in wheels) or (p["pt"] is not None and segment_hits(g_roi, p["pt"], pt)):
            p["target"].add(g_name)
    p["hist"].append((now, float(pt[0]), float(pt[1])))
    p["votes"][group] += 1
    p["pt"], p["box"], p["wheels"], p["t"] = pt, tuple(float(v) for v in box), wheels, now


def seed_from_pre(st, p):
    """Track baru mewarisi riwayat sebelum kuning: lintasan kaki masuk kuning bisa dihitung dari posisi sebelumnya
    (FPS rendah), gerak/arah punya sampel lebih banyak, dan observasi sebelumnya ikut memenuhi MIN_TRACK_FRAMES."""
    if not p or not p["hist"]:
        return
    st.hist.extend(p["hist"])
    st.votes.update(p["votes"])
    st.last_pt, st.last_box, st.last_wheels = p["pt"], p["box"], p["wheels"]
    t0, x0, y0 = p["hist"][0]
    st.origin_pt, st.origin_t = (x0, y0), t0
    st.frames = 1
    st.pre_target = set(p["target"])


def update_yellow_init(st, pt, box, prev_pt, prev_box, now, yellow_roi, entry_dirs, group):
    """Catat bukti kontak kuning. Return True bila objek kini punya titik masuk kuning yang SAH:
    (a) kaki/roda/pita bawah bbox menyentuh kuning, atau lintasan kaki antar-frame melewatinya; ATAU
    (b) hanya badan yang menyentuh kuning >= INIT_BODY_MIN_HITS kali DAN objek sudah maju searah kuning -> target."""
    if st.origin_pt is None:
        st.origin_pt, st.origin_t = (float(pt[0]), float(pt[1])), now
    yellow_hit, _ = touches_bbox(yellow_roi, box, prev_box)
    if yellow_hit:
        st.yellow_hits += 1
        if st.yellow_foot_pt is None:
            contact = yellow_ground_contact(yellow_roi, box, prev_pt, pt)
            if contact is not None:
                st.yellow_pt = st.yellow_foot_pt = (float(contact[0]), float(contact[1]))
                st.yellow_foot_t = now
                st.init_mode = "kaki"
            else:
                st.yellow_body_hits += 1
    if st.yellow_foot_pt is None and st.yellow_body_hits >= INIT_BODY_MIN_HITS and not st.parked:
        g = st.votes.most_common(1)[0][0] if st.votes else group
        bw = max(1.0, float(box[2]) - float(box[0]))
        d = entry_dirs[g]
        fwd = (pt[0] - st.origin_pt[0]) * float(d[0]) + (pt[1] - st.origin_pt[1]) * float(d[1])
        if fwd >= max(INIT_BODY_FWD_PX, INIT_BODY_FWD_RATIO * bw):
            st.yellow_pt = st.yellow_foot_pt = st.origin_pt
            st.yellow_foot_t = st.origin_t
            st.init_mode = "badan+gerak"
    return st.yellow_foot_pt is not None


def _box_center(box):
    return ((float(box[0]) + float(box[2])) / 2.0,
            (float(box[1]) + float(box[3])) / 2.0)


def box_iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 1e-9 else 0.0


def directional_motion_ok(st, direction, now, box):
    """Validasi gerak nyata menuju arah counting. Menghindari parkir/jitter bbox."""
    # Kompatibilitas terhadap TrackState lama yang belum memiliki atribut ini.
    yellow_foot_pt = getattr(st, "yellow_foot_pt", None)
    yellow_foot_t = getattr(st, "yellow_foot_t", None)
    if yellow_foot_pt is None or yellow_foot_t is None:
        return False
    if now - yellow_foot_t < MIN_YELLOW_TO_TARGET_SEC:
        return False

    bw = max(1.0, float(box[2]) - float(box[0]))
    recent = [h for h in st.hist if now - h[0] <= DIRECTIONAL_MOTION_WINDOW_SEC]
    if len(recent) < DIRECTIONAL_MOTION_MIN_POSITIVE_STEPS + 1:
        return False

    dx = recent[-1][1] - recent[0][1]
    dy = recent[-1][2] - recent[0][2]
    net_forward = dx * float(direction[0]) + dy * float(direction[1])
    if net_forward < max(DIRECTIONAL_MOTION_MIN_NET_PX, 0.30 * bw):
        return False

    meaningful = 0
    positive = 0
    for a, b in zip(recent, recent[1:]):
        sx = b[1] - a[1]
        sy = b[2] - a[2]
        step = float(np.hypot(sx, sy))
        proj = sx * float(direction[0]) + sy * float(direction[1])
        if step >= DIRECTIONAL_MOTION_MIN_STEP_PX:
            meaningful += 1
            if proj > 0:
                positive += 1

    if meaningful < DIRECTIONAL_MOTION_MIN_POSITIVE_STEPS:
        return False
    return positive / float(meaningful) >= DIRECTIONAL_MOTION_MIN_POSITIVE_RATIO


def update_motion_state(st, pt, box, now):
    """Klasifikasi PARKIR vs GERAK menggunakan beberapa frame, bukan 1 frame/jitter bbox."""
    st.hist.append((now, float(pt[0]), float(pt[1])))
    bw = max(1.0, float(box[2]) - float(box[0]))

    recent = [h for h in st.hist if now - h[0] <= PARK_WINDOW_SEC]
    if (not st.parked and len(recent) >= PARK_MIN_SAMPLES
            and now - recent[0][0] >= PARK_WINDOW_SEC * 0.80):
        xs = np.array([h[1] for h in recent], dtype=float)
        ys = np.array([h[2] for h in recent], dtype=float)
        cx, cy = float(xs.mean()), float(ys.mean())
        radius = float(np.percentile(np.hypot(xs - cx, ys - cy), 85))
        if radius <= max(PARK_MIN_RADIUS_PX, PARK_RADIUS_RATIO * bw):
            st.parked = True
            st.park_anchor = (cx, cy)
            st.motion_confirmed = False

    if st.parked and st.park_anchor is not None:
        d = float(np.hypot(pt[0] - st.park_anchor[0], pt[1] - st.park_anchor[1]))
        if d > max(PARK_RELEASE_PX, PARK_RELEASE_RATIO * bw):
            st.parked = False
            st.park_anchor = None
            st.hist.clear()
            st.hist.append((now, float(pt[0]), float(pt[1])))

    recent_move = list(st.hist)[-MOTION_CONFIRM_MIN_SAMPLES:]
    if len(recent_move) >= MOTION_CONFIRM_MIN_SAMPLES:
        endpoint = float(np.hypot(
            recent_move[-1][1] - recent_move[0][1],
            recent_move[-1][2] - recent_move[0][2]))
        path = 0.0
        moving_steps = 0
        step_thr = max(MOTION_CONFIRM_MIN_STEP_PX, 0.08 * bw)
        for a, b in zip(recent_move, recent_move[1:]):
            step = float(np.hypot(b[1] - a[1], b[2] - a[2]))
            path += step
            if step >= step_thr:
                moving_steps += 1
        motion_thr = max(20.0, MOTION_CONFIRM_RATIO * bw)
        if ((endpoint >= motion_thr and moving_steps >= 2)
                or (path >= 1.20 * motion_thr and endpoint >= 0.60 * motion_thr
                    and moving_steps >= 2)):
            st.motion_confirmed = True


def _park_memory_match(memory, group, box, now):
    """Cocokkan deteksi baru dengan lokasi kendaraan yang sebelumnya sudah terbukti parkir."""
    cx, cy = _box_center(box)
    bw = max(1.0, float(box[2]) - float(box[0]))
    best = None
    best_score = 1e9
    for m in memory:
        if m.get("group") != group:
            continue
        if now - m.get("last_seen", 0.0) > PARK_MEMORY_SEC:
            continue
        mb = m.get("box")
        if mb is None:
            continue
        iou = box_iou(box, mb)
        mcx, mcy = m["center"]
        dist = float(np.hypot(cx - mcx, cy - mcy))
        limit = max(25.0, PARK_MEMORY_DIST_RATIO * bw)
        if iou < PARK_MEMORY_IOU_MIN and dist > limit:
            continue
        score = (1.0 - iou) * 100.0 + dist
        if score < best_score:
            best = m
            best_score = score
    return best


def remember_parked(memory, group, box, now):
    """Simpan/update lokasi kendaraan parkir agar ID tracker baru tidak menghitung ulang objek yang sama."""
    cx, cy = _box_center(box)
    bw = max(1.0, float(box[2]) - float(box[0]))
    for m in memory:
        if m.get("group") != group or now - m.get("last_seen", 0.0) > PARK_MEMORY_SEC:
            continue
        mb = m.get("box")
        if mb is None:
            continue
        iou = box_iou(box, mb)
        mcx, mcy = m["center"]
        dist = float(np.hypot(cx - mcx, cy - mcy))
        if iou >= PARK_MEMORY_IOU_MIN or dist <= max(25.0, PARK_MEMORY_DIST_RATIO * bw):
            m["center"] = (0.8 * mcx + 0.2 * cx, 0.8 * mcy + 0.2 * cy)
            m["box"] = tuple(float(v) for v in box)
            m["last_seen"] = now
            return m
    m = {
        "id": len(memory) + 1,
        "group": group,
        "center": (cx, cy),
        "box": tuple(float(v) for v in box),
        "first_seen": now,
        "last_seen": now,
    }
    memory.append(m)
    return m


def prune_park_memory(memory, now):
    memory[:] = [m for m in memory if now - m.get("last_seen", 0.0) <= PARK_MEMORY_SEC]


def find_relink(lost, pt, group, now):
    """Cari track lama yang paling mungkin menjadi ID baru.

    Perubahan penting: perbedaan class sementara (motor terbaca car atau sebaliknya) tidak lagi
    langsung menggagalkan relink. Kedekatan posisi tetap menjadi syarat utama, sehingga kontinuitas
    track motor lebih terjaga ketika ByteTrack mengganti ID.
    """
    best_id, best_score = None, float(RELINK_DIST)
    for old_id, st in lost.items():
        if now - st.last_seen > RELINK_TIME or st.last_pt is None:
            continue
        d = float(np.hypot(pt[0] - st.last_pt[0], pt[1] - st.last_pt[1]))
        if d > RELINK_DIST:
            continue
        class_penalty = 0.0 if st.last_group == group else min(25.0, RELINK_DIST * 0.20)
        score = d + class_penalty
        if score < best_score:
            best_id, best_score = old_id, score
    return best_id


# ==========================================
# 3. ROI: SIMPAN / MUAT / GAMBAR DI LAYAR
# ==========================================
def build_zones(zones):
    yellow, green, red = zones
    dir_motor = compute_entry_dir(yellow, green)
    dir_car = compute_entry_dir(yellow, red, fallback=dir_motor)
    return to_poly(yellow), to_poly(green), to_poly(red), {"motor": dir_motor, "car": dir_car}


def draw_zone(vis, pts, color):
    poly = to_poly(pts)
    overlay = vis.copy()
    cv2.fillPoly(overlay, [poly], color)
    cv2.addWeighted(overlay, 0.18, vis, 0.82, 0, vis)
    cv2.polylines(vis, [poly], True, color, 2)


def draw_overlay(frame, lines, x, y, w, alpha=0.55):
    h = 25 * len(lines) + 15
    overlay = frame.copy()
    cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    for i, line in enumerate(lines):
        cv2.putText(frame, line, (x + 12, y + 28 + i * 25), cv2.FONT_HERSHEY_SIMPLEX,
                    0.52, (255, 255, 255), 1, cv2.LINE_AA)


def key_char(key):
    return chr(key).lower() if 0 <= key < 255 else ""


def draw_rois(frame_queue, initial=None):
    """
    Gambar area di atas snapshot stream, berurutan: kuning (inisialisasi), hijau (counting motor),
    merah (counting mobil). initial = list poligon tersimpan (2 atau 3 area).
    Return list 3 poligon, atau None jika batal.
    """
    n_zones = len(ROI_STEPS)
    frame = grab_frame(frame_queue)
    done = [[list(p) for p in z] for z in initial] if initial else []
    pts, cursor = [], [None]
    win = "Gambar ROI - CCTV Bapenda"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, FRAME_W, FRAME_H)

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(done) < n_zones:
                pts.append([x, y])
        elif event == cv2.EVENT_RBUTTONDOWN:
            if pts:
                pts.pop()
        elif event == cv2.EVENT_MOUSEMOVE:
            cursor[0] = (x, y)

    cv2.setMouseCallback(win, on_mouse)
    result = None
    while True:
        try:
            if cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
                break
        except cv2.error:
            break

        vis = frame.copy()
        for zone_pts, (_, color) in zip(done, ROI_STEPS):
            draw_zone(vis, zone_pts, color)

        if len(done) < n_zones:
            title, color = ROI_STEPS[len(done)]
            if pts:
                cv2.polylines(vis, [to_poly(pts)], False, color, 2)
                for p in pts:
                    cv2.circle(vis, tuple(p), 4, (255, 0, 255), -1)
                if cursor[0] is not None:
                    cv2.line(vis, tuple(pts[-1]), cursor[0], color, 1)
            lines = [
                f"GAMBAR {title}  ({len(done) + 1}/{n_zones})",
                "Klik kiri: tambah titik | Klik kanan/Backspace: undo",
                "ENTER: selesai (min 3 titik) | C: ulang | F: frame baru | Q: batal",
            ]
        else:
            lines = [
                "ROI LENGKAP (kuning + hijau + merah)",
                "ENTER: mulai | R: ulang semua | M: ulang area merah saja",
                "Saran: KUNING di jalur masuk, HIJAU/MERAH 60-120 px setelahnya; jangan tumpang tindih",
                "Q / Esc: batal",
            ]
        draw_overlay(vis, lines, 10, 10, 610)
        cv2.imshow(win, vis)

        key = cv2.waitKey(20) & 0xFF
        ch = key_char(key)
        if key == 27 or ch == "q":
            break
        if len(done) == n_zones:
            if key in (13, 10, 32):
                result = done
                break
            if ch == "r":
                done.clear()
                pts.clear()
            elif ch == "m":
                done.pop()
                pts.clear()
        else:
            if key in (13, 10, 32) and len(pts) >= 3:
                done.append(list(pts))
                pts.clear()
            elif key == 8 and pts:
                pts.pop()
            elif ch == "c":
                pts.clear()
        if ch == "f":
            frame = grab_frame(frame_queue)

    try:
        cv2.destroyWindow(win)
    except cv2.error:
        pass
    return result


# ==========================================
# 4. LOG & AKURASI
# ==========================================
def fmt_time(t):
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(t))


def describe_source(source):
    """Info sumber untuk CSV. URL langsung dibersihkan dari user:password dan query/token."""
    kind, value = source
    if kind == "jasnita":
        label = value                                  # id CCTV (display id)
    else:
        parsed = urlparse(value)
        host = parsed.hostname or ""
        if parsed.port:
            host += f":{parsed.port}"
        label = f"{parsed.scheme}://{host}{parsed.path}"
    return {"tipe": kind, "sumber": label, "sesi": fmt_time(time.time())}


def append_csv(path, header, row):
    """Tambah satu baris; tulis header jika file baru. Tidak menghentikan program jika file terkunci (Excel)."""
    try:
        is_new = (not path.exists()) or path.stat().st_size == 0
        with open(path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f, delimiter=CSV_DELIMITER)
            if is_new:
                writer.writerow(header)
            writer.writerow(row)
    except OSError as e:
        print(f"[WARN] Gagal menulis {path.name}: {e} (tutup file jika sedang dibuka di Excel)")


def write_jpg(path, img, quality=CAPTURE_JPEG_QUALITY):
    try:
        quality = max(1, min(100, int(quality)))
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
        if not ok:
            return False
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(buf.tobytes())
        return True
    except OSError as e:
        print(f"[WARN] Gagal menyimpan gambar {path.name}: {e}")
        return False


def enhance_capture_image(img):
    """Perbesar crop CCTV dan beri sharpening ringan tanpa menambah detail palsu."""
    if img is None or img.size == 0:
        return img

    h, w = img.shape[:2]
    long_side = max(h, w)
    if long_side <= 0:
        return img

    requested_scale = max(1.0, float(CAPTURE_SCALE))
    min_side_scale = float(CAPTURE_MIN_LONG_SIDE) / float(long_side)
    scale = min(float(CAPTURE_UPSCALE_MAX), max(requested_scale, min_side_scale))

    if scale > 1.01:
        nw = max(1, int(round(w * scale)))
        nh = max(1, int(round(h * scale)))
        img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_CUBIC)

    amount = max(0.0, min(1.0, float(CAPTURE_SHARPEN_AMOUNT)))
    if amount > 0.0:
        blurred = cv2.GaussianBlur(img, (0, 0), sigmaX=1.1)
        img = cv2.addWeighted(img, 1.0 + amount, blurred, -amount, 0)

    return img

def safe_slug(text):
    """Ubah id CCTV / URL menjadi nama folder yang aman di semua OS."""
    text = re.sub(r"^[a-zA-Z][a-zA-Z0-9+.-]*://", "", text or "")   # buang skema (rtsp://, https://, ...)
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", text).strip("_")
    return text[:80] or "unknown"


def ensure_event_log_schema():
    """Pastikan log utama memakai format 14 kolom yang diminta.

    Jika ada CSV lama dari versi sebelumnya, file lama dibackup otomatis agar tidak tercampur
    dengan format baru. Log baru tetap bernama log_kendaraan_terhitung.csv.
    """
    expected = [
        "ID", "NOP", "CCTV_ID", "NAMA_OP", "ALAMAT_OP", "WILAYAH_PAJAK",
        "WAKTU_MASUK", "JENIS_KEND", "PLAT_NO", "WAKTU_KELUAR",
        "DIRECTION", "LOG", "IMAGE_URL", "VENDOR"
    ]
    if not EVENT_LOG_FILE.exists() or EVENT_LOG_FILE.stat().st_size == 0:
        return

    try:
        with open(EVENT_LOG_FILE, newline="", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter=CSV_DELIMITER)
            current = next(reader, [])
    except (OSError, StopIteration):
        return

    if current == expected:
        return

    stamp = time.strftime("%Y%m%d_%H%M%S")
    backup = EVENT_LOG_FILE.with_name(f"{EVENT_LOG_FILE.stem}_legacy_{stamp}{EVENT_LOG_FILE.suffix}")
    try:
        EVENT_LOG_FILE.rename(backup)
        print(f"[INFO] Format log lama ditemukan. Backup: {backup.name}")
    except OSError as e:
        raise RuntimeError(
            f"Format {EVENT_LOG_FILE.name} lama tidak sesuai dan tidak bisa dibackup: {e}. "
            "Tutup file CSV jika sedang dibuka di Excel."
        ) from e

def _next_event_id(ctx):
    """Generate nomor ID log berurutan. Nilai terakhir dibaca sekali per sesi writer."""
    if "_next_log_id" not in ctx:
        last_id = 0
        if EVENT_LOG_FILE.exists():
            try:
                with open(EVENT_LOG_FILE, newline="", encoding="utf-8") as f:
                    reader = csv.DictReader(f, delimiter=CSV_DELIMITER)
                    for r in reader:
                        try:
                            last_id = max(last_id, int(str(r.get("ID", "")).strip()))
                        except (TypeError, ValueError):
                            # Kompatibilitas dengan log lama: abaikan ID yang bukan numerik.
                            try:
                                last_id = max(last_id, int(str(r.get("id", "")).strip()))
                            except (TypeError, ValueError):
                                continue
            except OSError as e:
                print(f"[WARN] Gagal membaca nomor ID log: {e}")
        ctx["_next_log_id"] = last_id + 1
    event_id = int(ctx["_next_log_id"])
    ctx["_next_log_id"] = event_id + 1
    return event_id


def save_capture(raw, box, tid, group, now, ctx, log_id):
    """Simpan foto kendaraan yang terhitung dan kembalikan path relatifnya."""
    if group not in CAPTURE_CLASSES:
        return ""
    x1, y1, x2, y2 = [int(v) for v in box]
    box_w = max(1, x2 - x1)
    box_h = max(1, y2 - y1)
    pad = max(24, int(CAPTURE_PAD_RATIO * max(box_w, box_h)))
    crop = raw[max(0, y1 - pad):min(FRAME_H, y2 + pad), max(0, x1 - pad):min(FRAME_W, x2 + pad)]
    if crop.size == 0:
        return ""

    crop = enhance_capture_image(crop)

    lt = time.localtime(now)
    capture_cctv = safe_slug(ctx.get("CCTV_ID") or ctx.get("sumber") or "unknown")
    folder = CAPTURE_DIR / capture_cctv / group / time.strftime("%Y-%m-%d", lt)
    stem = f"{time.strftime('%H%M%S', lt)}_ID{log_id}_track{tid}"
    path = folder / f"{stem}.jpg"
    if not write_jpg(path, crop, CAPTURE_JPEG_QUALITY):
        return ""
    if CAPTURE_FULL_FRAME:
        full = raw.copy()
        cv2.rectangle(full, (x1, y1), (x2, y2), (0, 255, 0), 2)
        write_jpg(folder / f"{stem}_full.jpg", full, CAPTURE_JPEG_QUALITY)
    return path.relative_to(BASE_DIR).as_posix()



def _read_existing_capture_excel_rows():
    """Bangun ulang daftar data Excel dari log CSV + index capture yang sudah ada."""
    rows = []
    if not CAPTURE_INDEX_FILE.exists():
        return rows

    nops = {}
    if EVENT_LOG_FILE.exists():
        try:
            with open(EVENT_LOG_FILE, newline="", encoding="utf-8") as f:
                for r in csv.DictReader(f, delimiter=CSV_DELIMITER):
                    rid = str(r.get("ID", "")).strip()
                    if rid:
                        nops[rid] = {
                            "NOP": r.get("NOP", ""),
                            "CCTV_ID": r.get("CCTV_ID", ""),
                        }
        except OSError as e:
            print(f"[WARN] Gagal membaca {EVENT_LOG_FILE.name} untuk Excel: {e}")

    try:
        with open(CAPTURE_INDEX_FILE, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f, delimiter=CSV_DELIMITER):
                rid = str(r.get("ID", "")).strip()
                image_path = (r.get("FILE_GAMBAR", "") or "").strip()
                if not rid or not image_path:
                    continue
                image_file = BASE_DIR / image_path
                if not image_file.exists() or not image_file.is_file():
                    continue
                meta = nops.get(rid, {})
                rows.append({
                    "ID": rid,
                    "NOP": meta.get("NOP", ""),
                    "CCTV_ID": r.get("CCTV_ID", "") or meta.get("CCTV_ID", ""),
                    "image_path": image_path,
                })
    except OSError as e:
        print(f"[WARN] Gagal membaca {CAPTURE_INDEX_FILE.name} untuk Excel: {e}")
    return rows


def append_capture_to_excel(log_id, ctx, image_path):
    """
    Tambahkan hasil capture ke log_capture_kendaraan.xlsx.

    Format kolom:
      ID | NOP | CCTV_ID | IMAGE_DATA

    IMAGE_DATA ditanam sebagai gambar JPG di kolom D. Implementasi memakai XlsxWriter
    agar tidak bergantung pada artifact_tool daemon yang dapat menyebabkan file Excel
    gagal dibuat pada komputer tempat program counting dijalankan.
    """
    if not image_path:
        return

    image_file = BASE_DIR / image_path
    if not image_file.exists() or not image_file.is_file():
        print(f"[WARN] Capture untuk Excel tidak ditemukan: {image_path}")
        return

    try:
        import xlsxwriter
    except ImportError:
        print(
            "[WARN] Modul XlsxWriter belum tersedia. Install sekali dengan: "
            "python -m pip install XlsxWriter"
        )
        return

    # Cache hanya untuk sesi berjalan agar tidak membaca ulang seluruh CSV setiap capture.
    cache = ctx.get("_capture_excel_rows")
    if cache is None:
        cache = _read_existing_capture_excel_rows()
        ctx["_capture_excel_rows"] = cache

    record = {
        "ID": str(log_id),
        "NOP": str(ctx.get("NOP", "")),
        "CCTV_ID": str(ctx.get("CCTV_ID", "")),
        "image_path": image_path,
    }

    # Jangan masukkan ID yang sama dua kali.
    if not any(str(r.get("ID")) == str(log_id) for r in cache):
        cache.append(record)

    tmp_path = CAPTURE_EXCEL_FILE.with_name(CAPTURE_EXCEL_FILE.stem + "_tmp.xlsx")
    try:
        workbook = xlsxwriter.Workbook(str(tmp_path))
        worksheet = workbook.add_worksheet(CAPTURE_EXCEL_SHEET)

        header_fmt = workbook.add_format({
            "bold": True,
            "font_color": "white",
            "bg_color": "#1F4E78",
            "align": "center",
            "valign": "vcenter",
            "border": 1,
        })
        text_fmt = workbook.add_format({
            "align": "center",
            "valign": "vcenter",
            "border": 1,
        })
        image_fmt = workbook.add_format({
            "align": "center",
            "valign": "vcenter",
            "border": 1,
        })

        worksheet.set_column("A:A", 12)
        worksheet.set_column("B:B", 24)
        worksheet.set_column("C:C", 20)
        worksheet.set_column("D:D", 46)
        worksheet.set_row(0, 24)
        worksheet.freeze_panes(1, 0)
        worksheet.autofilter(0, 0, max(1, len(cache)), 3)

        headers = ["ID", "NOP", "CCTV_ID", "IMAGE_DATA"]
        for col, value in enumerate(headers):
            worksheet.write(0, col, value, header_fmt)

        for row_idx, row in enumerate(cache, start=1):
            worksheet.set_row(row_idx, CAPTURE_EXCEL_IMAGE_H * 0.75)
            worksheet.write(row_idx, 0, row["ID"], text_fmt)
            worksheet.write(row_idx, 1, row["NOP"], text_fmt)
            worksheet.write(row_idx, 2, row["CCTV_ID"], text_fmt)
            worksheet.write_blank(row_idx, 3, None, image_fmt)

            img_path = BASE_DIR / row["image_path"]
            if not img_path.exists() or not img_path.is_file():
                worksheet.write(row_idx, 3, "[capture tidak ditemukan]", image_fmt)
                continue

            # Tentukan ukuran asli dengan OpenCV yang memang sudah menjadi dependency aplikasi.
            img = cv2.imread(str(img_path))
            if img is not None:
                h, w = img.shape[:2]
            else:
                w, h = CAPTURE_EXCEL_IMAGE_W, CAPTURE_EXCEL_IMAGE_H

            scale = min(
                float(CAPTURE_EXCEL_IMAGE_W) / max(1, w),
                float(CAPTURE_EXCEL_IMAGE_H) / max(1, h),
            )
            scale = max(0.05, min(1.0, scale))
            worksheet.insert_image(
                row_idx,
                3,
                str(img_path),
                {
                    "x_scale": scale,
                    "y_scale": scale,
                    "x_offset": 3,
                    "y_offset": 3,
                    "description": f"Capture kendaraan ID {row['ID']}",
                },
            )

        worksheet.write_comment(0, 3, "Gambar capture kendaraan tertanam di sel/kolom ini.")
        workbook.close()

        # Ganti file secara atomik supaya Excel tidak pernah melihat file setengah jadi.
        os.replace(str(tmp_path), str(CAPTURE_EXCEL_FILE))
    except Exception as e:
        try:
            if tmp_path.exists():
                tmp_path.unlink()
        except OSError:
            pass
        print(f"[WARN] Gagal membuat {CAPTURE_EXCEL_FILE.name}: {e}")


def log_event(raw, box, tid, group, now, ctx, counts):
    """Tulis log utama dengan format kolom persis sesuai kebutuhan integrasi Bapenda."""
    log_id = _next_event_id(ctx)
    image_path = save_capture(raw, box, tid, group, now, ctx, log_id)
    waktu = fmt_time(now)

    row = [
        log_id,
        ctx.get("NOP", ""),
        ctx.get("CCTV_ID", ""),
        ctx.get("NAMA_OP", ""),
        ctx.get("ALAMAT_OP", ""),
        "-",                       # WILAYAH_PAJAK
        waktu,                      # WAKTU_MASUK
        LOG_JENIS.get(group, "MOTOR"),
        "-",                       # PLAT_NO
        waktu,                      # WAKTU_KELUAR = WAKTU_MASUK
        "IN",                      # DIRECTION
        "SUKSES",                  # LOG
        "-",                       # IMAGE_URL
        LOG_VENDOR,
    ]
    append_csv(
        EVENT_LOG_FILE,
        [
            "ID", "NOP", "CCTV_ID", "NAMA_OP", "ALAMAT_OP", "WILAYAH_PAJAK",
            "WAKTU_MASUK", "JENIS_KEND", "PLAT_NO", "WAKTU_KELUAR",
            "DIRECTION", "LOG", "IMAGE_URL", "VENDOR"
        ],
        row,
    )

    # Index internal agar HTML report tetap dapat membuka capture walaupun IMAGE_URL pada
    # log utama memang harus selalu bernilai '-'.
    append_csv(
        CAPTURE_INDEX_FILE,
        ["ID", "CCTV_ID", "WAKTU_MASUK", "JENIS_KEND", "FILE_GAMBAR"],
        [log_id, ctx.get("CCTV_ID", ""), waktu, LOG_JENIS.get(group, "MOTOR"), image_path],
    )

    # Tambahan baru: simpan capture yang sama ke Excel sebagai gambar embedded.
    append_capture_to_excel(log_id, ctx, image_path)


def write_summary(ctx, counts, truth, now, note):
    """Simpan ringkasan hitungan jumlah kendaraan."""
    # Kolom tambahan lama dipertahankan kosong agar CSV lama tetap kompatibel.
    append_csv(
        SUMMARY_FILE,
        [
            "sesi_mulai", "waktu", "tipe_sumber", "id_cctv_atau_url",
            "jumlah_mobil", "jumlah_motor", "total", "manual_mobil", "manual_motor",
            "estimasi_lolos_mobil", "estimasi_lolos_motor", "akurasi_estimasi_persen",
            "stream_uptime_persen", "stream_putus_tak_terduga", "stream_refresh_terjadwal",
            "skor_keandalan_persen", "keterangan"
        ],
        [
            ctx["sesi"], fmt_time(now), ctx["tipe"], ctx["sumber"],
            counts["car"], counts["motor"], counts["car"] + counts["motor"],
            truth["car"], truth["motor"],
            "", "", "", "", "", "", "", note
        ],
    )


# ==========================================
# 4b. LAPORAN HTML (dashboard per ID CCTV)
# ==========================================
KELAS_LABEL = {"car": "Mobil", "motor": "Motor"}
KELAS_WARNA = {"car": "#2563eb", "motor": "#16a34a"}


def capture_file_exists(file_gambar):
    """Cek apakah file capture yang tercatat di CSV masih benar-benar ada di folder capture."""
    file_gambar = (file_gambar or "").strip()
    if not file_gambar:
        return False

    try:
        path = (BASE_DIR / file_gambar).resolve()
        capture_root = CAPTURE_DIR.resolve()
        # Pastikan file yang dicek memang berada di dalam folder capture.
        path.relative_to(capture_root)
        return path.is_file()
    except (OSError, ValueError):
        return False


def _read_capture_index():
    """Bangun map ID -> file capture dari index internal."""
    index = {}
    if not CAPTURE_INDEX_FILE.exists():
        return index
    try:
        with open(CAPTURE_INDEX_FILE, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f, delimiter=CSV_DELIMITER):
                rid = str(r.get("ID", "")).strip()
                if rid:
                    index[rid] = r.get("FILE_GAMBAR", "") or ""
    except OSError as e:
        print(f"[WARN] Gagal membaca {CAPTURE_INDEX_FILE.name}: {e}")
    return index


def read_events_for_report(cctv_id, day):
    """Baca log baru berbasis CCTV_ID atau log lama, lalu hanya tampilkan capture yang masih ada."""
    rows = []
    if not EVENT_LOG_FILE.exists():
        return rows

    capture_index = _read_capture_index()
    try:
        with open(EVENT_LOG_FILE, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter=CSV_DELIMITER)
            headers = set(reader.fieldnames or [])
            is_new_format = "CCTV_ID" in headers and "WAKTU_MASUK" in headers

            for r in reader:
                if is_new_format:
                    cid = (r.get("CCTV_ID") or "").strip()
                    waktu = (r.get("WAKTU_MASUK") or "").strip()
                    jenis = (r.get("JENIS_KEND") or "").strip().upper()
                    log_id = str(r.get("ID", "")).strip()
                    if cid != cctv_id or not waktu.startswith(day):
                        continue
                    group = "car" if jenis == "MOBIL" else "motor" if jenis == "MOTOR" else ""
                    if not group:
                        continue
                    rel = capture_index.get(log_id, "")
                    if not capture_file_exists(rel):
                        continue
                    # Normalisasi field internal agar renderer lama tetap sederhana.
                    r["waktu"] = waktu
                    r["kelas"] = group
                    r["file_gambar"] = rel
                    r["track_id"] = log_id
                else:
                    # Kompatibilitas log versi sebelumnya.
                    if r.get("id_cctv_atau_url") != cctv_id or not r.get("waktu", "").startswith(day):
                        continue
                    if not capture_file_exists(r.get("file_gambar")):
                        continue
                rows.append(r)
    except OSError as e:
        print(f"[WARN] Gagal membaca {EVENT_LOG_FILE.name}: {e}")
        return rows

    seen, unique = set(), []
    for r in rows:
        key = r.get("ID") if "ID" in r else (r.get("track_id"), r.get("kelas"), r.get("waktu"))
        if key in seen:
            continue
        seen.add(key)
        unique.append(r)
    unique.sort(key=lambda r: r.get("waktu", r.get("WAKTU_MASUK", "")), reverse=True)
    return unique


def list_known_cctv_ids(day=None):
    """Ambil CCTV_ID dari log baru atau id_cctv_atau_url dari log lama."""
    ids = []
    if not EVENT_LOG_FILE.exists():
        return ids
    seen = set()
    try:
        with open(EVENT_LOG_FILE, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter=CSV_DELIMITER)
            headers = set(reader.fieldnames or [])
            is_new_format = "CCTV_ID" in headers
            for r in reader:
                if is_new_format:
                    cid = (r.get("CCTV_ID") or "").strip()
                    when = (r.get("WAKTU_MASUK") or "").strip()
                else:
                    cid = (r.get("id_cctv_atau_url") or "").strip()
                    when = (r.get("waktu") or "").strip()
                if not cid or cid in seen:
                    continue
                if day and not when.startswith(day):
                    continue
                seen.add(cid)
                ids.append(cid)
    except OSError as e:
        print(f"[WARN] Gagal membaca {EVENT_LOG_FILE.name}: {e}")
    return ids


def hourly_counts(rows):
    buckets = {h: {"car": 0, "motor": 0} for h in range(24)}
    for r in rows:
        try:
            h = int(r["waktu"][11:13])
        except (ValueError, IndexError, KeyError):
            continue
        k = r.get("kelas")
        if k in buckets.get(h, {}):
            buckets[h][k] += 1
    return buckets


def build_svg_chart(buckets, width=760, height=200):
    pad_l, pad_r, pad_t, pad_b = 34, 8, 10, 22
    max_v = max(1, max(max(b["car"], b["motor"]) for b in buckets.values()))
    step = max(1, -(-max_v // 4))  # ceil(max_v/4), min 1

    def x(h):
        return pad_l + h * (width - pad_l - pad_r) / 23.0

    def y(v):
        return height - pad_b - (v / max_v) * (height - pad_t - pad_b)

    def line(kelas, color):
        pts = " ".join(f"{x(h):.1f},{y(buckets[h][kelas]):.1f}" for h in range(24))
        return f'<polyline fill="none" stroke="{color}" stroke-width="2.5" points="{pts}" />'

    grid, v = [], 0
    while v <= max_v:
        grid.append(f'<line x1="{pad_l}" y1="{y(v):.1f}" x2="{width - pad_r}" y2="{y(v):.1f}" '
                    f'stroke="#e5e7eb" stroke-width="1"/>')
        grid.append(f'<text x="2" y="{y(v) + 4:.1f}" font-size="10" fill="#6b7280">{v}</text>')
        v += step
    labels = "".join(
        f'<text x="{x(h):.1f}" y="{height - 6}" font-size="10" fill="#6b7280" text-anchor="middle">{h:02d}</text>'
        for h in range(0, 24, 2)
    )
    return (f'<svg viewBox="0 0 {width} {height}" width="100%" height="{height}" '
            f'xmlns="http://www.w3.org/2000/svg">' + "".join(grid) + labels +
            line("car", KELAS_WARNA["car"]) + line("motor", KELAS_WARNA["motor"]) + "</svg>")


def render_report_html(cctv_id, day, rows):
    total = {"car": sum(1 for r in rows if r["kelas"] == "car"),
             "motor": sum(1 for r in rows if r["kelas"] == "motor")}
    buckets = hourly_counts(rows)
    busiest = max(buckets.items(), key=lambda kv: kv[1]["car"] + kv[1]["motor"])
    n_hours_active = sum(1 for b in buckets.values() if b["car"] or b["motor"]) or 1
    avg_motor_per_jam = total["motor"] / n_hours_active
    avg_mobil_per_jam = total["car"] / n_hours_active
    chart = build_svg_chart(buckets)
    def cell_img(r):
        rel = (r.get("file_gambar") or "").strip()
        if rel and capture_file_exists(rel):
            src = html.escape(f"../{rel}", quote=True)
            return (f'<a class="capture-link" href="{src}" '
                    f'onclick="openCapture(event, this.href)" title="Klik untuk memperbesar">'
                    f'<img src="{src}" alt="capture kendaraan" loading="lazy" decoding="async" draggable="false"></a>')
        return '<div class="no-img">-</div>'


    def cell_row(r):
        kelas = r.get("kelas", "")
        badge = KELAS_LABEL.get(kelas, kelas)
        color = KELAS_WARNA.get(kelas, "#6b7280")
        waktu = html.escape(r.get("waktu", "")[11:19])
        tid = html.escape(str(r.get("track_id", "")))
        return (f'<tr><td class="thumb">{cell_img(r)}</td>'
                f'<td>{waktu}</td>'
                f'<td><span class="badge" style="background:{color}1a;color:{color}">{html.escape(badge)}</span></td>'
                f'<td class="muted">#{tid}</td></tr>')

    rows_html = "".join(cell_row(r) for r in rows) or (
        '<tr><td colspan="4" class="muted center">Belum ada kendaraan terhitung pada tanggal ini.</td></tr>')

    return f"""<!DOCTYPE html>
<html lang="id">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Laporan Traffic Counting - {html.escape(cctv_id)} - {day}</title>
<style>
  :root {{ color-scheme: light; }}
  * {{ box-sizing: border-box; }}
  body {{ margin: 0; font-family: -apple-system, "Segoe UI", Roboto, Arial, sans-serif;
         background: #f3f4f6; color: #111827; }}
  .wrap {{ max-width: 1080px; margin: 0 auto; padding: 24px 20px 60px; }}
  header {{ margin-bottom: 20px; }}
  header h1 {{ font-size: 20px; margin: 0 0 4px; }}
  header .sub {{ color: #6b7280; font-size: 13px; }}
  .cards {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr));
           gap: 14px; margin-bottom: 22px; }}
  .card {{ background: #fff; border-radius: 12px; padding: 16px; box-shadow: 0 1px 2px rgba(0,0,0,.06); }}
  .card .label {{ font-size: 12px; color: #6b7280; margin-bottom: 6px; }}
  .card .value {{ font-size: 26px; font-weight: 700; }}
  .card.mobil .value {{ color: {KELAS_WARNA["car"]}; }}
  .card.motor .value {{ color: {KELAS_WARNA["motor"]}; }}
  .panel {{ background: #fff; border-radius: 12px; padding: 18px; box-shadow: 0 1px 2px rgba(0,0,0,.06);
           margin-bottom: 22px; }}
  .panel h2 {{ font-size: 14px; margin: 0 0 12px; color: #374151; }}
  .legend {{ display: flex; gap: 16px; font-size: 12px; color: #374151; margin-top: 6px; }}
  .legend span::before {{ content: ""; display: inline-block; width: 10px; height: 10px; border-radius: 50%;
                         margin-right: 5px; vertical-align: -1px; }}
  .legend .l-mobil::before {{ background: {KELAS_WARNA["car"]}; }}
  .legend .l-motor::before {{ background: {KELAS_WARNA["motor"]}; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
  th {{ text-align: left; padding: 8px 10px; background: #f9fafb; color: #6b7280;
       font-weight: 600; border-bottom: 1px solid #e5e7eb; }}
  td {{ padding: 8px 10px; border-bottom: 1px solid #f3f4f6; vertical-align: middle; }}
  tr:hover td {{ background: #f9fafb; }}
  td.thumb {{ width: {REPORT_THUMB_W + 14}px; }}
  .capture-link {{ display: inline-flex; width: {REPORT_THUMB_W}px; height: {REPORT_THUMB_H}px; align-items: center; justify-content: center;
                  border-radius: 10px; overflow: hidden; background: #111827; cursor: zoom-in;
                  box-shadow: 0 1px 3px rgba(0,0,0,.12); text-decoration: none; }}
  td.thumb img {{ width: {REPORT_THUMB_W}px; height: {REPORT_THUMB_H}px; object-fit: contain; display: block;
                 cursor: zoom-in; }}
  .capture-link:hover {{ box-shadow: 0 0 0 2px #2563eb55; transform: translateY(-1px); }}
  .no-img {{ width: {REPORT_THUMB_W}px; height: {REPORT_THUMB_H}px; border-radius: 10px; background: #f3f4f6; color: #9ca3af;
            display: flex; align-items: center; justify-content: center; font-size: 12px; }}

  .image-modal {{ position: fixed; inset: 0; z-index: 9999; display: none; padding: 24px;
                  background: rgba(0,0,0,.78); align-items: center; justify-content: center; cursor: zoom-out; }}
  .image-modal.show {{ display: flex; }}
  .image-modal-content {{ position: relative; max-width: 96vw; max-height: 94vh; display: flex;
                          align-items: center; justify-content: center; cursor: default; }}
  .image-modal-content img {{ max-width: 96vw; max-height: 94vh; width: auto; height: auto;
                              object-fit: contain; border-radius: 8px; box-shadow: 0 8px 40px rgba(0,0,0,.45);
                              background: #111827; display: block; }}
  .image-modal-close {{ position: fixed; top: 16px; right: 20px; width: 42px; height: 42px;
                        border: 0; border-radius: 50%; background: rgba(255,255,255,.92); color: #111827;
                        font-size: 28px; line-height: 42px; text-align: center; cursor: pointer;
                        box-shadow: 0 2px 10px rgba(0,0,0,.2); }}
  .image-modal-hint {{ position: fixed; left: 50%; bottom: 16px; transform: translateX(-50%);
                       color: #fff; font-size: 12px; background: rgba(0,0,0,.42); padding: 6px 10px;
                       border-radius: 999px; }}
  body.modal-open {{ overflow: hidden; }}
  .badge {{ padding: 3px 10px; border-radius: 999px; font-size: 12px; font-weight: 600; }}
  .muted {{ color: #6b7280; }}
  .center {{ text-align: center; padding: 24px 0; }}
  .detail {{ font-size: 13px; color: #374151; line-height: 1.9; }}
</style>
</head>
<body>
<div class="wrap">
  <header>
    <h1>Laporan Traffic Counting</h1>
    <div class="sub">ID CCTV: <strong>{html.escape(cctv_id)}</strong> &middot; Tanggal: {day}
      &middot; Dibuat: {fmt_time(time.time())}</div>
  </header>

  <div class="cards">
    <div class="card mobil"><div class="label">Mobil Hari Ini</div><div class="value">{total["car"]}</div></div>
    <div class="card motor"><div class="label">Motor Hari Ini</div><div class="value">{total["motor"]}</div></div>
    <div class="card"><div class="label">Total Kendaraan</div><div class="value">{total["car"] + total["motor"]}</div></div>
  </div>

  <div class="panel">
    <h2>Kendaraan per Jam</h2>
    {chart}
    <div class="legend"><span class="l-mobil">Mobil</span><span class="l-motor">Motor</span></div>
    <div class="detail" style="margin-top:14px">
      Jam tersibuk: {busiest[0]:02d}:00 - {busiest[0]:02d}:59
      ({busiest[1]["car"] + busiest[1]["motor"]} kendaraan)<br>
      Rata-rata motor per jam aktif: {avg_motor_per_jam:.1f}<br>
      Rata-rata mobil per jam aktif: {avg_mobil_per_jam:.1f}
    </div>
  </div>

  <div class="panel">
    <h2>Aktivitas Hari Ini ({len(rows)} kendaraan, capture terbaru di atas, tanpa duplikat)</h2>
    <table>
      <thead><tr><th>Capture</th><th>Waktu</th><th>Jenis</th><th>ID Track</th></tr></thead>
      <tbody>{rows_html}</tbody>
    </table>
  </div>
</div>

<div id="captureModal" class="image-modal" onclick="closeCapture(event)" aria-hidden="true">
  <button class="image-modal-close" type="button" onclick="closeCapture()" aria-label="Tutup">&times;</button>
  <div class="image-modal-content">
    <img id="captureModalImg" src="" alt="Capture kendaraan diperbesar">
  </div>
  <div class="image-modal-hint">Klik di luar gambar atau tekan Esc untuk menutup</div>
</div>

<script>
function openCapture(event, src) {{
  if (event) event.preventDefault();
  const modal = document.getElementById('captureModal');
  const img = document.getElementById('captureModalImg');
  img.src = src;
  modal.classList.add('show');
  modal.setAttribute('aria-hidden', 'false');
  document.body.classList.add('modal-open');
}}

function closeCapture(event) {{
  const modal = document.getElementById('captureModal');
  if (event && event.target !== modal) return;
  modal.classList.remove('show');
  modal.setAttribute('aria-hidden', 'true');
  document.body.classList.remove('modal-open');
  document.getElementById('captureModalImg').src = '';
}}

document.addEventListener('keydown', function(event) {{
  if (event.key === 'Escape') closeCapture();
}});
</script>
</body>
</html>"""


def generate_html_report(cctv_id, day=None):
    """Buat/timpa 1 file laporan.html untuk satu ID CCTV pada satu tanggal. Return path file."""
    day = day or time.strftime("%Y-%m-%d")
    rows = read_events_for_report(cctv_id, day)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    path = REPORT_DIR / f"laporan_{safe_slug(cctv_id)}_{day.replace('-', '')}.html"
    path.write_text(render_report_html(cctv_id, day, rows), encoding="utf-8")
    return path


def accuracy(auto, truth):
    if truth <= 0:
        return None
    return 100.0 * max(0.0, 1.0 - abs(auto - truth) / truth)


# ==========================================
# 5. PROSES UTAMA
# ==========================================
def writer_worker(q, ctx):
    """
    THREAD TERPISAH dari GUI: menjalankan semua operasi disk yang berat/lambat -- simpan foto capture,
    tulis CSV, buat laporan HTML -- supaya loop tampilan (run()) tidak pernah menunggu disk. Ini yang
    sebelumnya jadi penyebab jendela freeze tepat saat ada kendaraan yang baru terhitung: log_event()
    dulu dipanggil langsung di thread GUI. Sentinel None dipakai utk menghentikan thread ini dgn rapi.
    """
    while True:
        item = q.get()
        try:
            if item is None:
                return
            kind = item[0]
            if kind == "log_event":
                _, raw, box, tid, group, now, counts_snapshot = item
                log_event(raw, box, tid, group, now, ctx, counts_snapshot)
            elif kind == "summary":
                _, counts_snapshot, truth_snapshot, now, note = item
                write_summary(ctx, counts_snapshot, truth_snapshot, now, note)
            elif kind == "report":
                _, day = item
                generate_html_report(ctx["CCTV_ID"], day)
        except Exception as e:
            # Jangan sampai thread ini mati gara-gara 1 penulisan gagal (mis. file lagi dibuka di Excel);
            # cukup catat, sisanya tetap lanjut -- konsisten dgn append_csv/write_jpg yg juga toleran begitu.
            print(f"[WARN] Penulis background gagal ({item[0] if item else '?'}): {e}")
        finally:
            q.task_done()


def run(frame_queue, counts, truth, session):
    ctx = session["ctx"]
    # --- Gambar / konfirmasi ROI di layar ---
    print("[INFO] Gambar ulang semua area (kuning, hijau, merah) ...")
    roi = draw_rois(frame_queue)
    if roi is None:
        print("[INFO] Pengaturan ROI dibatalkan.")
        return
    yellow_roi, green_roi, red_roi, entry_dirs = build_zones(roi)

    tracker_path = Path(__file__).with_name("bytetrack_traffic.yaml")
    tracker_path.write_text(TRACKER_CFG)

    # Semua penulisan disk (capture foto, CSV, laporan HTML) dikerjakan thread terpisah, BUKAN di GUI loop --
    # ini yang bikin jendela tidak lagi freeze tepat saat kendaraan baru terhitung.
    write_queue = Queue()
    writer_thread = threading.Thread(target=writer_worker, args=(write_queue, ctx), daemon=True)
    writer_thread.start()

    # YOLO berjalan di proses terpisah (inference_worker) supaya jendela tampilan di bawah ini
    # tidak pernah menunggu proses deteksi yang berat -> tidak "Not Responding".
    result_queue = mp.Queue(maxsize=1)
    infer_proc = mp.Process(target=inference_worker, args=(frame_queue, result_queue, str(tracker_path)),
                             daemon=True)
    infer_proc.start()

    tracks, lost = {}, {}
    counted_ids = set()   # ID yang sudah terhitung (tidak akan dihitung lagi)
    park_memory = []      # lokasi kendaraan parkir persisten, independen dari ID ByteTrack
    pre = {}              # riwayat deteksi yang belum menyentuh kuning (lihat pre_observe)
    total_init = 0
    fps, prev_t = 0.0, time.time()
    last_summary, last_written = time.time(), (0, 0)
    debug = False

    current_day = time.strftime("%Y-%m-%d")

    win = "CCTV Bapenda - AI Traffic Counting v5"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, FRAME_W, FRAME_H)
    print("[INFO] Menunggu model YOLO siap di proses terpisah ...")
    print("[INFO] Berjalan. q=keluar, d=debug, r=edit ROI, z=+motor manual, x=+mobil manual")
    session["counting"] = True

    try:
        while True:
            try:
                raw, payload, _ = result_queue.get(timeout=5.0)
            except Empty:
                # Tetap proses tombol & pompa jendela walau belum ada hasil deteksi baru,
                # supaya jendela tetap dianggap "responding" oleh Windows.
                key_char(cv2.waitKey(30) & 0xFF)
                continue

            now = time.time()
            fps = 0.9 * fps + 0.1 * (1.0 / max(now - prev_t, 1e-3))
            prev_t = now

            today = time.strftime("%Y-%m-%d", time.localtime(now))
            if today != current_day:
                prev_day = current_day
                write_queue.put(("summary", dict(counts), dict(truth), now, "ganti_hari"))
                write_queue.put(("report", prev_day))
                log_network_event(ctx["sumber"], "ganti_hari",
                                   detail=f"reset counter dari {prev_day} ke {today}")
                counts["car"] = counts["motor"] = 0
                truth["car"] = truth["motor"] = 0
                total_init = 0
                tracks.clear()
                lost.clear()
                counted_ids.clear()
                park_memory.clear()
                pre.clear()
                last_written = (0, 0)
                current_day = today
                print(f"[INFO] Tanggal berganti ke {today} -> semua hitungan (mobil/motor) direset ke 0.")

            vis = raw.copy()
            cv2.polylines(vis, [yellow_roi], True, COLOR_YELLOW, 2)
            cv2.polylines(vis, [green_roi], True, COLOR_GREEN, 2)
            cv2.polylines(vis, [red_roi], True, COLOR_RED, 2)

            seen = set()
            if payload is not None:
                xyxy = payload["xyxy"]
                ids = payload["ids"].tolist()
                clss = payload["cls"].tolist()
                confs = payload["conf"].tolist()

                for box, tid, cls_id, conf in zip(xyxy, ids, clss, confs):
                    group = "motor" if cls_id == 3 else "car"
                    x1, y1, x2, y2 = box
                    pt = ((float(x1) + float(x2)) / 2.0, float(y2))  # titik kaki = bidang tanah
                    seen.add(tid)

                    st = tracks.get(tid)

                    # Kendaraan baru HANYA mulai dibuat ketika badan kendaraan menyentuh
                    # area KUNING. Namun sebelum menolak objek di luar kuning, coba relink dulu:
                    # bila ini sebenarnya kendaraan lama yang baru mendapat ID berbeda setelah keluar
                    # dari kuning, track harus tetap dilanjutkan sampai zona counting.
                    # bbox sebelumnya (riwayat pra-kuning) ikut dipakai: di FPS rendah kendaraan bisa melompati
                    # area kuning di antara dua deteksi tanpa bbox-nya pernah tepat berada di atas kuning.
                    p_prev = pre.get(tid)
                    yellow_hit_new, yellow_anchor_new = touches_bbox(
                        yellow_roi, box, p_prev["box"] if p_prev is not None else None)
                    if st is None:
                        st = lost.pop(tid, None)                       # ID lama muncul kembali
                        if st is None:
                            old = find_relink(lost, pt, group, now)    # ID baru dari track lama
                            st = lost.pop(old) if old is not None else None

                        # Belum punya track lama dan belum menyentuh kuning -> jangan buat state baru,
                        # tetapi simpan riwayatnya (posisi sebelum kuning = bukti arah objek -> kuning).
                        if st is None and not yellow_hit_new:
                            pre_observe(pre, tid, group, pt, box, now, green_roi, red_roi)
                            continue

                        if st is None:
                            st = TrackState(now)
                            seed_from_pre(st, pre.pop(tid, None))
                        else:
                            pre.pop(tid, None)
                        tracks[tid] = st
                        if tid in counted_ids:
                            st.counted = True          # ID yang sudah pernah terhitung tidak dihitung lagi
                        elif st.counted:
                            counted_ids.add(tid)       # state hasil relink yang sudah terhitung

                    st.frames += 1
                    # X pada gambar referensi hanya anotasi evaluasi: kendaraan parkir/diam dideteksi otomatis.
                    update_motion_state(st, pt, box, now)
                    pm = _park_memory_match(park_memory, group, tuple(float(v) for v in box), now)
                    if pm is not None and not st.motion_confirmed:
                        st.parked = True
                        st.park_anchor = pm["center"]
                        st.park_memory_id = pm["id"]
                    if st.parked:
                        pm2 = remember_parked(park_memory, group, box, now)
                        st.park_memory_id = pm2["id"]
                    prune_park_memory(park_memory, now)
                    # Voting kelas sederhana tetapi lebih stabil: jangan percaya 1 frame saja.
                    st.votes[group] += 1
                    st.last_group = group
                    prev_pt = st.last_pt
                    prev_box = st.last_box

                    # --- Inisialisasi (kuning): kontak KAKI/tanah, atau badan + gerak maju (lihat update_yellow_init) ---
                    # Titik masuk disimpan dalam koordinat KAKI agar jarak tempuh (pt - yellow_pt) tidak bias;
                    # dulu titiknya bisa berupa anchor di tengah badan sehingga 'travel' melenceng ~0.6 x tinggi bbox.
                    yellow_ok = update_yellow_init(st, pt, box, prev_pt, prev_box, now, yellow_roi, entry_dirs, group)

                    # Tetap harus punya minimal 2 observasi (riwayat pra-kuning ikut dihitung) dan tidak sedang
                    # parkir, sehingga deteksi satu-frame / kendaraan diam di kuning tidak langsung sah.
                    if (not st.initialized and yellow_ok and not st.parked
                            and st.frames >= MIN_TRACK_FRAMES
                            and st.yellow_hits >= MIN_YELLOW_HITS):
                        st.initialized = True
                        total_init += 1

                    # --- Counting: INDIKATOR HANYA GARIS ROI ---
                    # Motor: badan kendaraan menyentuh/menyeberangi GARIS HIJAU.
                    # Mobil/bus/truk: roda ATAU badan kendaraan menyentuh/menyeberangi GARIS MERAH.
                    wheels = wheel_points(float(x1), float(x2), float(y2))
                    vote_frames = sum(st.votes.values())
                    # Recheck lokal tepat sebelum counting: jika posisi sebenarnya masih berputar di
                    # sekitar lokasi yang sama, paksa PARKIR walaupun PARK_WINDOW belum selesai penuh.
                    recent_now = [h for h in st.hist if now - h[0] <= DIRECTIONAL_MOTION_WINDOW_SEC]
                    if len(recent_now) >= 4:
                        rx = np.array([h[1] for h in recent_now], dtype=float)
                        ry = np.array([h[2] for h in recent_now], dtype=float)
                        spread = float(np.hypot(rx.max() - rx.min(), ry.max() - ry.min()))
                        local_thr = max(PARK_MIN_RADIUS_PX, 0.20 * max(1.0, float(x2 - x1)))
                        if spread <= local_thr:
                            st.motion_confirmed = False
                            st.parked = True
                            st.park_anchor = (float(rx.mean()), float(ry.mean()))
                            pm3 = remember_parked(park_memory, group, box, now)
                            st.park_memory_id = pm3["id"]

                    candidate_group = st.votes.most_common(1)[0][0] if st.votes else group
                    directional_ok = (
                        not st.parked
                        and st.motion_confirmed
                        and directional_motion_ok(st, entry_dirs[candidate_group], now, box)
                    )
                    if (st.initialized and not st.counted and tid not in counted_ids
                            and not st.parked and st.motion_confirmed and directional_ok
                            and vote_frames >= MIN_CLASS_CONFIRM_FRAMES):
                        cur_group = st.votes.most_common(1)[0][0]
                        target_roi = green_roi if cur_group == "motor" else red_roi
                        hit = vehicle_touches_count_line(
                            target_roi, box, prev_box=prev_box,
                            wheels=wheels, prev_wheels=st.last_wheels
                        )
                        # Kaki sudah menyentuh area counting kelas ini SEBELUM kuning -> arah balik, bukan
                        # urutan objek -> kuning -> hijau/merah.
                        if hit and cur_group not in st.pre_target:
                            # Travel hanya menjadi filter arah minimum. Kendaraan TIDAK harus masuk
                            # ke dalam area polygon; menyentuh garis sudah cukup sebagai trigger.
                            travel = float(np.dot((pt[0] - st.yellow_pt[0], pt[1] - st.yellow_pt[1]),
                                                  entry_dirs[cur_group]))
                            if travel >= MIN_TRAVEL_PX:
                                st.counted = True
                                counted_ids.add(tid)
                                counts[cur_group] += 1
                                write_queue.put(("log_event", raw, tuple(float(v) for v in box),
                                                  tid, cur_group, now, dict(counts)))

                    st.last_pt = pt
                    st.last_wheels = wheels
                    st.last_box = tuple(float(v) for v in box)
                    st.last_seen = now

                    if (st.initialized and not st.parked) or debug:
                        color = COLOR_GREEN if st.counted else (COLOR_YELLOW if st.initialized else (160, 160, 160))
                        if debug and st.parked:
                            color = (255, 128, 0)
                        label_extra = " PARKIR" if (debug and st.parked) else (" GERAK" if debug and st.motion_confirmed else "")
                        if debug:
                            label_extra += f" INIT:{st.init_mode}" if st.initialized else " belum-init"
                        cv2.rectangle(vis, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                        cv2.putText(vis, f"#{tid} {LABEL[group]} {conf:.2f}{label_extra}", (int(x1), int(y1) - 5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                        for wp in (wheels if group == "car" else [pt]):
                            cv2.circle(vis, (int(wp[0]), int(wp[1])), 4, (255, 0, 255), -1)
                        # Marker kecil kontak garis target saat debug/track sudah diinisialisasi.
                        target_roi = green_roi if group == "motor" else red_roi
                        for bp in bbox_anchor_points(box)[:5]:
                            if point_near_poly_boundary(target_roi, bp):
                                cv2.circle(vis, (int(bp[0]), int(bp[1])), 5, (0, 165, 255), -1)

            # Track yang tidak terlihat -> pindah ke 'lost' (untuk relink) atau dibuang
            for tid in list(tracks):
                st = tracks[tid]
                if tid not in seen and now - st.last_seen > LOST_AFTER:
                    del tracks[tid]
                    if st.yellow_hits > 0:
                        lost[tid] = st
            for tid in list(lost):
                st = lost[tid]
                if now - st.last_seen > RELINK_TIME:
                    # Track yang terlalu lama hilang dibuang dari daftar relink.
                    del lost[tid]
            for tid in [k for k, p in pre.items() if now - p["t"] > PRE_TRACK_KEEP_SEC]:
                del pre[tid]

            pending = sum(1 for t in seen if t in tracks and tracks[t].initialized and not tracks[t].counted
                          and not tracks[t].parked and tracks[t].motion_confirmed)
            lines = [
                "--- TRAFFIC ANALYTICS ---",
                f"Mobil (Car)       : {counts['car']}",
                f"Motorcycle        : {counts['motor']}",
                f"Total Vehicle     : {counts['car'] + counts['motor']}",
                "-------------------------",
                f"Lewat zona kuning : {total_init}",
                f"Sedang transit    : {pending}",
                f"Parkir/diam       : {sum(1 for t in seen if t in tracks and tracks[t].parked)}",
                f"FPS               : {fps:.1f}",
]
            if truth["car"] or truth["motor"]:
                ac, am = accuracy(counts["car"], truth["car"]), accuracy(counts["motor"], truth["motor"])
                lines += [
                    "--- vs HITUNGAN MANUAL ---",
                    f"Manual Mobil/Motor: {truth['car']}/{truth['motor']}",
                    f"Akurasi Mobil     : {'n/a' if ac is None else f'{ac:.1f}%'}",
                    f"Akurasi Motor     : {'n/a' if am is None else f'{am:.1f}%'}",
                ]
            if now - last_summary >= SUMMARY_INTERVAL_SEC:
                last_summary = now
                current = (counts["car"], counts["motor"])
                if current != last_written:
                    write_queue.put(("summary", dict(counts), dict(truth), now, "berkala"))
                    write_queue.put(("report", current_day))
                    last_written = current
            draw_overlay(vis, lines, FRAME_W - 320, 10, 310)
            cv2.imshow(win, vis)

            key = cv2.waitKey(1) & 0xFF
            ch = key_char(key)
            if ch == "q":
                break
            elif ch == "d":
                debug = not debug
            elif ch == "z":
                truth["motor"] += 1
            elif ch == "x":
                truth["car"] += 1
            elif ch == "r":
                new_roi = draw_rois(frame_queue, roi)
                if new_roi is not None:
                    roi = new_roi
                    yellow_roi, green_roi, red_roi, entry_dirs = build_zones(roi)
                    tracks.clear()
                    lost.clear()
                    pre.clear()
                prev_t = time.time()
    finally:
        # Tunggu semua tulisan yg masih di antrean selesai dulu, baru tutup thread-nya dgn rapi.
        write_queue.join()
        write_queue.put(None)
        writer_thread.join(timeout=5)
        # Ringkasan akhir ditulis LANGSUNG (bukan lewat antrean) -- satu-satunya penulisan blocking yg
        # tersisa, sengaja, karena hanya terjadi sekali saat program benar-benar berhenti, bukan saat
        # kendaraan lewat, jadi tidak lagi menyebabkan freeze.
        write_summary(ctx, counts, truth, time.time(), "akhir")
        report_path = generate_html_report(ctx["CCTV_ID"], current_day)
        print(f"[HASIL] Mobil={counts['car']} Motor={counts['motor']} | "
              f"ringkasan: {SUMMARY_FILE.name} | laporan: {report_path}")
        infer_proc.terminate()
        infer_proc.join()




def ask_log_metadata():
    """Minta metadata lokasi/objek sekali di awal sesi counting."""
    fields = [
        ("NOP", "Masukkan NOP"),
        ("CCTV_ID", "Masukkan CCTV_ID"),
        ("NAMA_OP", "Masukkan NAMA_OP"),
        ("ALAMAT_OP", "Masukkan ALAMAT_OP"),
    ]
    data = {}
    for key, prompt in fields:
        while True:
            try:
                value = input(f"{prompt}: ").strip()
            except EOFError:
                print("[ERROR] Input metadata tidak tersedia.")
                sys.exit(1)
            if value:
                data[key] = value
                break
            print(f"[ERROR] {key} wajib diisi.")
    print(
        f"[INFO] Metadata log: NOP={data['NOP']} | CCTV_ID={data['CCTV_ID']} | "
        f"NAMA_OP={data['NAMA_OP']} | ALAMAT_OP={data['ALAMAT_OP']}"
    )
    return data


def main():
    if "--report" in sys.argv:
        cli_report()
        return
    source = ask_stream_source()
    if source[0] == "jasnita" and not (JASNITA_USER and JASNITA_PASS):
        print("Kredensial JASNITA_USER / JASNITA_PASS belum diisi.")
        sys.exit(1)
    print(f"[INFO] Sumber stream: {'Jasnita display ' if source[0] == 'jasnita' else ''}{source[1]}")

    ensure_event_log_schema()
    metadata = ask_log_metadata()
    counts = {"car": 0, "motor": 0}
    truth = {"car": 0, "motor": 0}
    ctx = describe_source(source)
    ctx.update(metadata)
    session = {"ctx": ctx, "counting": False}

    frame_queue = mp.Queue(maxsize=1)
    proc = mp.Process(target=stream_worker, args=(source, frame_queue), daemon=True)
    proc.start()
    try:
        run(frame_queue, counts, truth, session)
    except KeyboardInterrupt:
        print("[INFO] Dihentikan (Ctrl+C).")
    finally:
        # Ringkasan akhir & laporan HTML sudah ditulis di dalam run() sendiri (lihat finally di sana),
        # supaya urutannya benar: tunggu antrean writer kosong dulu baru tulis angka final.
        proc.terminate()
        proc.join()
        cv2.destroyAllWindows()


def cli_report():
    """python traffic_counter_v5.py --report ["<id_cctv>"]  -- buat laporan tanpa membuka kamera."""
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    cctv_arg = args[0].strip().strip('"').strip("'") if args else None
    ids = [cctv_arg] if cctv_arg else list_known_cctv_ids(day=time.strftime("%Y-%m-%d"))
    if not ids:
        print(f"[INFO] Tidak ada data hari ini di {EVENT_LOG_FILE.name}. "
              f"Jalankan dulu counting-nya, atau sebutkan ID CCTV secara manual.")
        return
    for cid in ids:
        path = generate_html_report(cid)
        print(f"[INFO] Laporan dibuat: {path}")


if __name__ == "__main__":
    mp.freeze_support()
    main()
