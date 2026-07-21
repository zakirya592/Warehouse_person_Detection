import os
import threading
import time

import cv2
from flask import Flask, Response, jsonify, request
from flask_cors import CORS

# Import detection logic and models from camera_shoes
from camera_shoes import CAMERA_CONFIGS, PersonTracker, process_frame
from detection_alert_db import get_all_alerts

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
        print(f"Connecting to {config['name']} at {config['ip']}...")
        cap = None
        for rtsp_url in config["rtsp_urls"]:
            print(f"  Trying: {rtsp_url}")
            cap = cv2.VideoCapture(rtsp_url)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc("H", "2", "6", "4"))

            if cap.isOpened():
                ret, frame = cap.read()
                if ret and frame is not None:
                    caps.append((cap, config["name"], i))
                    print(f"Connected to {config['name']}")
                    break
                cap.release()
            else:
                cap.release()

        if cap is None or not any(c[0] == cap for c in caps):
            print(f"Failed to connect to {config['name']}")

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
    return (
        "<html><body style='margin:0;background:#111;color:#fff;font-family:sans-serif'>"
        "<h2 style='padding:16px'>PPE Live Detection</h2>"
        "<div style='display:flex;flex-wrap:wrap;gap:16px;justify-content:center;padding:16px'>"
        "<div><h3>Camera 1</h3>"
        "<img src='/live-detection-camera-1' style='width:100%;max-width:640px;display:block'/></div>"
        "<div><h3>Camera 2</h3>"
        "<img src='/live-detection-camera-2' style='width:100%;max-width:640px;display:block'/></div>"
        "</div></body></html>"
    )


@app.route("/live-detection")
def live_detection():
    return Response(
        _generate_mjpeg(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@app.route("/live-detection-camera-1")
def live_detection_camera_1():
    return Response(
        _generate_mjpeg(camera_id=1),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@app.route("/live-detection-camera-2")
def live_detection_camera_2():
    return Response(
        _generate_mjpeg(camera_id=2),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@app.route("/health")
def health():
    return {
        "status": "ok",
        "stream_running": _stream_running,
        "cameras": {
            f"camera_{i}": {
                "has_frame": i in _latest_frames,
                "endpoint": f"/live-detection-camera-{i}",
            }
            for i in range(1, len(CAMERA_CONFIGS) + 1)
        },
    }


@app.route("/detection-alerts", methods=["GET"])
def detection_alerts():
    status = request.args.get("status")
    alerts = get_all_alerts(status=status)
    return jsonify({"count": len(alerts), "data": alerts})


def main():
    thread = threading.Thread(target=_camera_loop, daemon=True)
    thread.start()

    print(f"API running on http://0.0.0.0:{PORT}")
    print(f"Camera 1: http://localhost:{PORT}/live-detection-camera-1")
    print(f"Camera 2: http://localhost:{PORT}/live-detection-camera-2")
    print(f"Combined: http://localhost:{PORT}/live-detection")
    print(f"Alerts: http://localhost:{PORT}/detection-alerts")
    app.run(host="0.0.0.0", port=PORT, threaded=True)


if __name__ == "__main__":
    main()
