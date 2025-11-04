import cv2, os, sys, math
import time
from datetime import datetime, timedelta
import tkinter as tk
from tkinter import filedialog, simpledialog

#video_path = "video_penuh\JR226.mp4"
# Input source (file OR RTSP URL)
video_path = filedialog.askopenfilename(title="Select Video File or cancel to type URL",
                                        filetypes=[["Video files", "*.mp4 *.avi *.mov *.mkv"]])
if not video_path:
    video_path = simpledialog.askstring("Video URL", "Enter camera URL (e.g. rtsp://user:pass@host:554/stream):")
    if not video_path:
        print("[ERROR] No source provided. Exiting.")
        raise SystemExit
    
out_dir = "frames"; os.makedirs(out_dir, exist_ok=True)
every_sec = 2

cap = cv2.VideoCapture(video_path)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if cap.get(cv2.CAP_PROP_FRAME_COUNT) > 0 else 0
fps = cap.get(cv2.CAP_PROP_FPS)
step = int(math.ceil(fps * every_sec))
time_obj = datetime.strptime("08:00:00", "%H:%M:%S").time()
start_date = datetime.today().date()
start_time = datetime.combine(start_date, time_obj)
current_timestamp = start_time
i = 0
saved = 0
while True:
    ret = cap.grab()
    if not ret: break
    if i % step == 0:
        ret, frame = cap.retrieve()
        if not ret: break
        current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) if total_frames > 0 else 0

        # Time labels (file or live)
        if total_frames > 0:
            current_time_sec = current_frame / fps
            total_time_sec = total_frames / fps
        else:
            current_time_sec = time.monotonic() - t0_live
            total_time_sec = 0.0
        current_time_str = time.strftime("%H:%M:%S", time.gmtime(current_time_sec))
        total_time_str = time.strftime("%H:%M:%S", time.gmtime(total_time_sec)) if total_time_sec else "--:--:--"
        time_obj_local = datetime.strptime(current_time_str, "%H:%M:%S").time()
        delta = timedelta(hours=time_obj_local.hour, minutes=time_obj_local.minute, seconds=time_obj_local.second)
        result = start_time + delta
        if int(time_obj_local.second) == 0 and int(time_obj_local.minute) > 0:
            if result.time() > current_timestamp.time():
                current_timestamp = result
                print(f"[INFO] {current_timestamp}")
                cv2.imwrite(os.path.join(out_dir, f"frame_{saved:05d}.jpg"), frame)
                saved += 1
    i += 1
cap.release()
print(f"Saved {saved} frames to {out_dir}")
