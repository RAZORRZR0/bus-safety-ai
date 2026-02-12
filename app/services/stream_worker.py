import argparse
import time

import cv2
import requests


def run(rtsp: str, api_base: str, token: str, bus_id: str, route_id: str, timeslot: str):
    cap = cv2.VideoCapture(rtsp)
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open stream: {rtsp}")
    headers = {"Authorization": f"Bearer {token}"}

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                time.sleep(0.2)
                continue
            small = cv2.resize(frame, (640, 360))
            ok_enc, enc = cv2.imencode(".jpg", small)
            if not ok_enc:
                time.sleep(0.2)
                continue
            params = {
                "bus_id": bus_id,
                "route_id": route_id,
                "timeslot": timeslot,
                "reserved_seats_total": 4,
                "special_category_present": "false",
            }
            try:
                files = {"file": ("frame.jpg", enc.tobytes(), "image/jpeg")}
                requests.post(
                    f"{api_base}/ingest/frame",
                    headers=headers,
                    params=params,
                    files=files,
                    timeout=4,
                )
            except requests.RequestException:
                pass
            time.sleep(0.5)
    finally:
        cap.release()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RTSP stream worker for transport policing MVP")
    parser.add_argument("--rtsp", required=True, help="RTSP url or camera index, e.g. 0 or rtsp://...")
    parser.add_argument("--api-base", default="http://127.0.0.1:8000")
    parser.add_argument("--token", required=True)
    parser.add_argument("--bus-id", default="BUS-01")
    parser.add_argument("--route-id", default="R-12")
    parser.add_argument("--timeslot", default="08:00-10:00")
    args = parser.parse_args()

    rtsp_input = int(args.rtsp) if args.rtsp.isdigit() else args.rtsp
    run(rtsp_input, args.api_base, args.token, args.bus_id, args.route_id, args.timeslot)
