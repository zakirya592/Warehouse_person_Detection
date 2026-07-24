import os
import threading
import time

import cv2
from flask import Flask, Response, jsonify, request
from flask_cors import CORS

from NVRConnect import (
    ACTIVE_CHANNELS,
    CAMERA_CONFIGS,
    NVR_IP,
    PersonTracker,
    open_camera,
    process_frame,
)
from detection_alert_db import get_all_alerts
from notification_logging import setup_notification_logging

os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")
os.environ.setdefault("OPENCV_FFMPEG_LOGLEVEL", "-8")

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})
PORT = 5051

_frame_lock = threading.Lock()
_latest_frames = {}
_stream_running = False


def _connect_cameras():
    caps = []
    for i, config in enumerate(CAMERA_CONFIGS, start=1):
        cap, name = open_camera(config)
        if cap is not None:
            caps.append((cap, name, i))
    return caps


def _camera_loop():
    global _latest_frames, _stream_running

    caps = _connect_cameras()
    if not caps:
        print("No cameras connected. Stream will be unavailable.")
        _stream_running = False
        return

    _stream_running = True
    frame_counters = {camera_id: 0 for _, _, camera_id in caps}
    person_trackers = {camera_id: PersonTracker() for _, _, camera_id in caps}

    print(f"Live detection stream started with {len(caps)} camera(s).")

    while _stream_running:
        for cap, name, camera_id in caps:
            success, frame = cap.read()
            if success:
                frame_counters[camera_id] += 1
                processed = process_frame(
                    frame, name, frame_counters[camera_id], person_trackers[camera_id]
                )
                with _frame_lock:
                    _latest_frames[camera_id] = cv2.resize(processed, (1280, 720))
            else:
                print(f"Failed to read from {name}")

    for cap, _, _ in caps:
        cap.release()


def _generate_mjpeg(camera_id=None):
    while True:
        with _frame_lock:
            if camera_id is not None:
                frame = None if camera_id not in _latest_frames else _latest_frames[camera_id].copy()
            elif _latest_frames:
                frames = sorted(_latest_frames.items())
                if len(frames) == 1:
                    frame = frames[0][1].copy()
                else:
                    f1 = cv2.resize(frames[0][1], (640, 360))
                    f2 = cv2.resize(frames[1][1], (640, 360))
                    frame = cv2.hconcat([f1, f2])
            else:
                frame = None

        if frame is None:
            time.sleep(0.1)
            continue

        ok, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if not ok:
            continue

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n"
        )
        time.sleep(0.033)


@app.route("/")
def index():
    camera_blocks = "".join(
        f"<div><h3>{config['name']}</h3>"
        f"<p style='margin:0 0 8px;color:#aaa'>{config.get('location', 'Unknown')}</p>"
        f"<img src='/live-detection-camera-{i}' "
        f"style='width:100%;max-width:640px;display:block'/></div>"
        for i, config in enumerate(CAMERA_CONFIGS, start=1)
    )
    return (
        "<html><body style='margin:0;background:#111;color:#fff;font-family:sans-serif'>"
        f"<h2 style='padding:16px'>PPE Live Detection — NVR ({NVR_IP})</h2>"
        "<div style='display:flex;flex-wrap:wrap;gap:16px;justify-content:center;padding:16px'>"
        f"{camera_blocks}"
        "</div></body></html>"
    )


@app.route("/live-detection")
def live_detection():
    return Response(
        _generate_mjpeg(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@app.route("/live-detection-camera-<int:camera_id>")
def live_detection_camera(camera_id):
    if camera_id < 1 or camera_id > len(CAMERA_CONFIGS):
        return jsonify({"error": f"Camera {camera_id} not configured"}), 404
    return Response(
        _generate_mjpeg(camera_id=camera_id),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )

@app.route("/health")
def health():
    return {
        "status": "ok",
        "stream_running": _stream_running,
        "nvr_ip": NVR_IP,
        "active_channels": ACTIVE_CHANNELS,
        "cameras": {
            f"camera_{i}": {
                "name": config["name"],
                "location": config.get("location", "Unknown"),
                "has_frame": i in _latest_frames,
                "endpoint": f"/live-detection-camera-{i}",
            }
            for i, config in enumerate(CAMERA_CONFIGS, start=1)
        },
    }


@app.route("/detection-alerts", methods=["GET"])
def detection_alerts():
    status = request.args.get("status")
    alerts = get_all_alerts(status=status)
    return jsonify({"count": len(alerts), "data": alerts})


def main():
    log_file = setup_notification_logging()
    print(f"Notification logs: {log_file}")

    thread = threading.Thread(target=_camera_loop, daemon=True)
    thread.start()

    print(f"API running on http://0.0.0.0:{PORT}")
    print(f"NVR: {NVR_IP} — {len(CAMERA_CONFIGS)} channel(s) configured")
    for i, config in enumerate(CAMERA_CONFIGS, start=1):
        print(
            f"  Camera {i} ({config.get('location', 'Unknown')}): "
            f"http://localhost:{PORT}/live-detection-camera-{i}"
        )
    print(f"Combined: http://localhost:{PORT}/live-detection")
    print(f"Alerts: http://localhost:{PORT}/detection-alerts")
    app.run(host="0.0.0.0", port=PORT, threaded=True)


if __name__ == "__main__":
    main()
