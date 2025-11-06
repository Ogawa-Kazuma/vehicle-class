import os
# Disable NNPACK to avoid "Unsupported Hardware" error on unsupported CPUs
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
os.environ['USE_NNPACK'] = '0'
os.environ['OMP_NUM_THREADS'] = '1'

import csv
import cv2
import ssl
import json
import copy
import time
import numpy as np 
import tkinter as tk
import supervision as sv 
import paho.mqtt.client as mqtt 
import tkinter.simpledialog as simpledialog
from ultralytics import YOLO
from tkinter import filedialog
from tkinter import messagebox 
from datetime import datetime, timedelta
from supervision.draw.utils import draw_text 
from supervision.detection.core import Detections
from supervision.detection.line_zone import LineZone
from collections import defaultdict, OrderedDict, deque
from supervision.detection.tools.polygon_zone import PolygonZone
import torch 

# --- SPEED ESTIMATION CONSTANTS (Requires Calibration!) ---
# You MUST adjust the SOURCE array and the PIXELS_PER_METER_SCALE for accuracy.

# 1. DEFAULT_SOURCE: Placeholder for the four points in the video frame. 
# This array will be initialized from perspective_points in process_video or this default will be used.
DEFAULT_SOURCE = np.array([[1252, 787], [2298, 803], [5039, 2159], [-550, 2159]], dtype=np.float32) 

# 2. TARGET: Defines the dimensions of the Bird's Eye View (BEV) space.
TARGET_WIDTH = 25
TARGET_HEIGHT = 250
TARGET = np.array(
    [
        [0, 0],
        [TARGET_WIDTH - 1, 0],
        [TARGET_WIDTH - 1, TARGET_HEIGHT - 1],
        [0, TARGET_HEIGHT - 1],
    ], dtype=np.float32 # Ensure TARGET is float32
)

# 3. PIXELS_PER_METER_SCALE: The primary calibration factor.
# Adjust this value until a vehicle traveling at a known speed (e.g., 50 km/h) is displayed correctly.
PIXELS_PER_METER_SCALE = 0.2 # <<--- ADJUST THIS VALUE FOR ACCURACY (< 1)

# 4. TIME_SCALE_FACTOR: Correction for videos where 1 video second != 1 real-world second.
# Use 1.5 since your video runs slower (1 video second = 1.5 real seconds).
TIME_SCALE_FACTOR = 5 # <<--- YOUR CORRECTION FACTOR HERE

# --- END SPEED ESTIMATION CONSTANTS ---

# --- CONFIG ---
MODEL_PATH = "rahh.pt"
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "rnd")
# Stream URL configuration (RTSP, HTTP, etc.)
# Examples: "rtsp://username:password@ip:port/stream", "http://ip:port/video"
STREAM_URL = ""  # Set to empty string to prompt user, or set default stream URL here
STREAM_NAME = "live_stream"  # Identifier for config/log files
USE_LIVE_STREAM = True  # Set to False to use file dialog instead
# Use Malay class names
CLASS_MAP = {0: 'Kelas 1', 1: 'Kelas 2', 2: 'Kelas 3', 3: 'Kelas 4', 4: 'Kelas 5', 5: 'Kelas 6'}
SUMMARY_CLASSES = ['Kelas 1', 'Kelas 2', 'Kelas 3', 'Kelas 4', 'Kelas 5', 'Kelas 6']
CLASS_COLOR = {'Kelas 1': (186,179,255), 'Kelas 2': (186,223,255), 'Kelas 3': (186,255,255),
               'Kelas 4': (201,255,186), 'Kelas 5': (255,255,186), 'Kelas 6': (255,203,241)}

BUTTONS = {'poly': (770, 20, 850, 60), 
           'start': (870, 20, 950, 60), 
           'line': (970, 20, 1050, 60), 
           'exit': (1070, 20, 1150, 60)} 
MIN_CONF = 0.25
FAST_MODE_FPS = 120
DISPLAY_SCALE = 0.75

BUTTON_COLOR_INACTIVE = (50, 50, 50)
BUTTON_COLOR_ACTIVE = (0, 165, 255) 

SHORTCUTS = OrderedDict([
    ("ESC / EXIT", "Exit App"),
    ("ENTER", "Finalize Poly"),
    ("R", "Reset All"),
    ("S", "Start/Restart"),
    ("L", "Edit Line"),
    ("C", "Calibrate Perspective"), 
    ("A", "Auto Line"),
    ("D", "Flow Mode"),
    ("F", "Fast Mode"),
])

# --- MQTT CONFIG ---
MQTT_BROKER = "broker.react.net.my"
MQTT_PORT = 8883 # Secure port
MQTT_TOPIC = "kkr/stl/ai/data/live" # New Topic from user request
MQTT_USER = "test_ai_stl"
MQTT_PASSWORD = "test_ai_stl_2025" 
MQTT_CLIENT_ID = "ppk_video"
PUBLISH_INTERVAL_MINUTES = 1 
# --- END MQTT CONFIG ---

# --- STREAM CONFIG ---
STREAM_RECONNECT_DELAY = 2  # seconds to wait before reconnecting
STREAM_MAX_RECONNECT_ATTEMPTS = 10  # max reconnection attempts before giving up
STREAM_BUFFER_SIZE = 1  # reduce buffer for low latency
# --- END STREAM CONFIG ---

# --- GLOBAL STATE VARIABLES ---
polygon_points = []
poly_editing = False
polygon_defined = False
start_processing = False
line_edit_mode = False
count_line_y = 0

# NEW PERSPECTIVE GLOBALS
perspective_points = [] # User-selected points for the SOURCE array
perspective_editing = False

frame_width = 0
frame_height = 0

zone_polygon = None
polygon_annotator = None
line_zone = None 
trace_annotator = None 
view_transformer = None 
coordinates = defaultdict(lambda: deque(maxlen=60)) 

run_date = None
run_start_time = None
force_exit = False 
root = None 
start_dt = None

global_offset_dt = None 

global_in_offset = 0
global_out_offset = 0

last_in_count = 0
last_out_count = 0
tracker_direction_map = {} 

individual_log_file = None
individual_log_writer = None
summary_log_file = None
summary_log_writer = None

logged_tracker_ids = set() 
total_class_counts = defaultdict(int) 
flow_direction_mode = "BOTH"

mqtt_client = None 
last_publish_time = None 

# Stream connection state
stream_source = None  # Will hold stream URL or file path
is_live_stream = False  # Flag to indicate if using live stream
stream_start_time = None  # Real-time start for live streams

# --- NEW METRIC AGGREGATION (For 15-minute average calculation) ---
# NOTE: These lists will ONLY contain '0's unless you implement the
# perspective transformation and speed/headway calculation logic.
agg_speeds = [] 
agg_headways = []
agg_gaps = []
agg_occupancy_frames = []
# --- END NEW METRIC AGGREGATION ---

# --- ViewTransformer Class for Speed Estimation ---

class ViewTransformer:
    def __init__(self, source: np.ndarray, target: np.ndarray) -> None:
        # Ensure input arrays are float32 as required by cv2.getPerspectiveTransform
        source = source.astype(np.float32)
        target = target.astype(np.float32)
        self.m = cv2.getPerspectiveTransform(source, target)

    def transform_points(self, points: np.ndarray) -> np.ndarray:
        if points.size == 0:
            return points

        reshaped_points = points.reshape(-1, 1, 2).astype(np.float32)
        transformed_points = cv2.perspectiveTransform(reshaped_points, self.m)
        return transformed_points.reshape(-1, 2)

# --- CONFIG PERSISTENCE ---

def get_config_path(stream_source):
    """Generates the path for the config JSON file based on the video path or stream identifier."""
    global is_live_stream
    # Save config files in the same directory as the script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if is_live_stream:
        # For live streams, use STREAM_NAME
        config_name = f"{STREAM_NAME}_config.json"
    else:
        # For files, use filename
        video_name = os.path.basename(stream_source)
        config_name = f"{os.path.splitext(video_name)[0]}_config.json"
    return os.path.join(script_dir, config_name)

def load_config(stream_source):
    """Loads configuration data from a JSON file."""
    config_path = get_config_path(stream_source)
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
                print(f"[INFO] Loaded configuration from {config_path}")
                
                # Load perspective points with validation
                global perspective_points
                loaded_perspective = config.get('perspective_points')
                if loaded_perspective:
                    # Validate format: should be a list of 4 points, each with 2 coordinates
                    if isinstance(loaded_perspective, list) and len(loaded_perspective) == 4:
                        # Validate each point has 2 coordinates
                        if all(isinstance(p, (list, tuple)) and len(p) == 2 for p in loaded_perspective):
                            perspective_points = [[float(p[0]), float(p[1])] for p in loaded_perspective]
                            print("[INFO] Loaded custom perspective points.")
                        else:
                            print("[WARN] Invalid perspective_points format in config. Using defaults.")
                            perspective_points = []
                    else:
                        print(f"[WARN] Invalid perspective_points count in config (expected 4, got {len(loaded_perspective) if loaded_perspective else 0}). Using defaults.")
                        perspective_points = []
                else:
                    perspective_points = []

                return config
        except json.JSONDecodeError as e:
            print(f"[WARN] Config file is corrupted or incomplete: {e}")
            print(f"[WARN] You may need to recalibrate. Config file: {config_path}")
            return {}
        except Exception as e:
            print(f"[WARN] Failed to load config from {config_path}: {e}")
            return {}
    return {}

def save_config(stream_source, config_data):
    """Saves configuration data to a JSON file."""
    config_path = get_config_path(stream_source)
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    
    # NEW: Save the perspective points
    global perspective_points
    
    # Convert perspective_points to list of lists (ensure JSON-serializable)
    # Handle both tuples and lists, and ensure all values are native Python types
    if perspective_points:
        config_data['perspective_points'] = [
            [float(p[0]), float(p[1])] for p in perspective_points
        ]
    else:
        config_data['perspective_points'] = []
    
    try:
        # Write to temporary file first, then rename (atomic write)
        temp_path = config_path + '.tmp'
        with open(temp_path, 'w') as f:
            json.dump(config_data, f, indent=4)
            f.flush()
            os.fsync(f.fileno())  # Force write to disk
        
        # Atomic rename (only if write succeeded)
        os.replace(temp_path, config_path)
        print(f"[INFO] Saved configuration to {config_path}")
        return True
    except Exception as e:
        print(f"[ERROR] Failed to save config to {config_path}: {e}")
        # Clean up temp file if it exists
        temp_path = config_path + '.tmp'
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except:
                pass
        return False


# --- LOGGING FUNCTIONS (Unchanged) ---

def open_log_files(stream_source):
    """Opens and initializes the two CSV log files, including metadata headers."""
    global individual_log_file, individual_log_writer, summary_log_file, summary_log_writer, run_date, is_live_stream
    
    if is_live_stream:
        # Use STREAM_NAME for live streams
        log_identifier = STREAM_NAME
    else:
        log_identifier = os.path.basename(stream_source).split('.')[0]
    
    # 1. Individual Vehicle Log
    individual_path = os.path.join(OUT_DIR, f"{log_identifier}_log.csv")
    os.makedirs(os.path.dirname(individual_path), exist_ok=True)
    
    individual_log_file = open(individual_path, 'w', newline='')
    individual_log_writer = csv.writer(individual_log_file)
    individual_log_writer.writerow(['Timestamp', 'ID', 'Type', 'Confidence', 'VideoTime', 'Image'])
    print(f"[INFO] Individual log file created: {individual_path}")

    # 2. Summary Log
    summary_path = os.path.join(OUT_DIR, f"{log_identifier}_summary_log.csv")
    summary_log_file = open(summary_path, 'w', newline='')
    summary_log_writer = csv.writer(summary_log_file)
    
    summary_log_writer.writerow([log_identifier]) 
    
    date_str = run_date if run_date else "DD/MM/YYYY" 
    if date_str and len(date_str) == 10 and date_str[4] == '-':
        try:
            date_obj = datetime.strptime(date_str, "%Y-%m-%d")
            date_str = date_obj.strftime("%d/%m/%Y")
        except ValueError:
            pass 
            
    summary_log_writer.writerow([date_str]) 
    summary_log_writer.writerow(['Masa'] + SUMMARY_CLASSES)
    print(f"[INFO] Summary log file created: {summary_path}")


def close_log_files():
    """Closes all open log files."""
    global individual_log_file, summary_log_file
    if individual_log_file:
        individual_log_file.close()
        print("[INFO] Individual log file closed.")
    if summary_log_file:
        summary_log_file.close()
        print("[INFO] Summary log file closed.")

def write_individual_log(writer, current_dt, tracker_id, class_id, distance, confidence, video_time, image_name):
    """Writes a single vehicle crossing event to the individual log file."""
    if writer is None:
        print("[WARN] Individual log writer not initialized.")
        return
        
    class_name = CLASS_MAP.get(class_id, "UNKNOWN")
    timestamp_str = current_dt.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3] 
    
    writer.writerow([timestamp_str, int(tracker_id), class_name, confidence, video_time, image_name])
    individual_log_file.flush() 

def log_cropped_image(frame, bbox, tracker_id, current_dt, filename, stream_id_folder):
    """Crops the bounding box area from the frame and saves it to disk."""
    global frame_width, frame_height
    
    crops_dir = os.path.join(OUT_DIR, "vehicle_captures", stream_id_folder)
    os.makedirs(crops_dir, exist_ok=True)
    
    x1, y1, x2, y2 = [int(i) for i in bbox]
    
    padding = 10
    x1 = max(0, x1 - padding)
    y1 = max(0, y1 - padding)
    x2 = min(frame_width, x2 + padding)
    y2 = min(frame_height, y2 + padding)

    cropped_image = frame[y1:y2, x1:x2]
    
    if cropped_image.size > 0:
        file_path = os.path.join(crops_dir, filename)
        cv2.imwrite(file_path, cropped_image)
    else:
        pass 

# --- MQTT FUNCTIONS (MODIFIED) ---

def on_connect(client, userdata, flags, rc):
    """Callback function for when the client connects to the MQTT broker."""
    if rc == 0:
        print("[INFO] MQTT Connected successfully with result code " + str(rc))
    else:
        print("[ERROR] MQTT Failed to connect, return code " + str(rc))

def connect_mqtt():
    """Initializes and connects the global MQTT client with TLS/SSL."""
    global mqtt_client
    
    mqtt_client = mqtt.Client(
        client_id=MQTT_CLIENT_ID, 
        protocol=mqtt.MQTTv311,
    ) 
    
    mqtt_client.on_connect = on_connect

    try:
        mqtt_client.tls_set(
            tls_version=ssl.PROTOCOL_TLSv1_2, 
            cert_reqs=ssl.CERT_REQUIRED
        ) 
    except Exception as e:
        print(f"[ERROR] TLS setup failed: {e}. Connection will likely fail.")
    
    mqtt_client.username_pw_set(MQTT_USER, MQTT_PASSWORD)
    
    try:
        mqtt_client.connect(MQTT_BROKER, MQTT_PORT, keepalive=60)
        mqtt_client.loop_start() 
        return mqtt_client
    except Exception as e:
        print(f"[ERROR] Failed to connect to MQTT broker: {e}")
        mqtt_client = None
        return None

def publish_15min_report(stream_source, current_dt, class_counts):
    """Formats the summary data into the new 15-minute JSON structure, calculates averages, and publishes it."""
    global mqtt_client
    global agg_speeds, agg_headways, agg_gaps, agg_occupancy_frames

    if mqtt_client is None or not mqtt_client.is_connected():
        print("[WARN] MQTT client not connected. Attempting to re-connect...")
        connect_mqtt()
        if mqtt_client is None or not mqtt_client.is_connected():
            print("[ERROR] MQTT client still not connected. Skipping publish.")
            return

    # --- 1. Calculate Averages from Aggregation Lists ---
    # NOTE: These averages will be 0.00 until the necessary computer vision logic
    # for speed, headway, gap, and occupancy is implemented in the main loop.
    
    # Use numpy for easy averaging. Use 0.0 if the list is empty (division by zero prevention).
    avg_speed = np.mean(agg_speeds) if agg_speeds else 0
    avg_headway = np.mean(agg_headways) if agg_headways else 0
    avg_gap = np.mean(agg_gaps) if agg_gaps else 0
    avg_occupancy = np.mean(agg_occupancy_frames) if agg_occupancy_frames else 0
    
    # 2. Static device info from the user's provided JSON example
    device_id = "TEST-AI-0001"
    device_name = "TEST-AI-0001"
    lane_id = "SB_FL"
    lane_name = "Fast Lane"
    direction_id = 0 

    total_volume = sum(class_counts.values())
    
    # 3. Create the data payload matching the new format
    data = {
        "types": "camera",
        "device_id": device_id,
        "device_name": device_name,
        "lane_id": lane_id,
        "lane_name": lane_name,
        # Timestamp is the *end* time of the 15-minute period
        "timestamp": current_dt.strftime("%Y-%m-%d %H:%M:%S"),
        
        # Class counts (Total over the 15-minute period)
        "class_1": class_counts.get('Kelas 1', 0),
        "class_2": class_counts.get('Kelas 2', 0),
        "class_3": class_counts.get('Kelas 3', 0),
        "class_4": class_counts.get('Kelas 4', 0),
        "class_5": class_counts.get('Kelas 5', 0),
        "class_6": class_counts.get('Kelas 6', 0),
        
        # Aggregated statistics (Now using calculated averages)
        "volume": total_volume,
        "speed": int(avg_speed),      
        "headway": int(avg_headway),    
        "gap": int(avg_gap),        
        "occupancy": int(avg_occupancy),  
        "sampling_number": 1, 
        "direction_id": direction_id
    }
    
    payload = json.dumps(data, indent=4)
    
    try:
        # Publish with QoS 0
        mqtt_client.publish(MQTT_TOPIC, payload, qos=0)
        print(f"[INFO] Published 15-minute MQTT data to {MQTT_TOPIC} ending at {data['timestamp']}")
    except Exception as e:
        print(f"[ERROR] Failed to publish MQTT message: {e}")

    # --- 4. Reset Aggregation Lists AFTER publishing ---
    agg_speeds.clear()
    agg_headways.clear()
    agg_gaps.clear()
    agg_occupancy_frames.clear()


# --- HELPER FUNCTIONS (Unchanged) ---

def is_point_in_button(x, y, button_coords):
    x_min, y_min, x_max, y_max = button_coords
    return x_min <= x <= x_max and y_min <= y <= y_max

def update_line_zone(y, width):
    global line_zone
    line_start = sv.Point(x=0, y=y)
    line_end = sv.Point(x=width, y=y)
    line_zone = sv.LineZone(
        start=line_start, 
        end=line_end,
        triggering_anchors=(sv.Position.BOTTOM_CENTER,),
        minimum_crossing_threshold=1 
    ) 

def reset_all(): 
    global polygon_points, poly_editing, polygon_defined, start_processing 
    global line_edit_mode, zone_polygon, polygon_annotator, line_zone, trace_annotator 
    global logged_tracker_ids, total_class_counts, flow_direction_mode
    global global_in_offset, global_out_offset, coordinates  
    global agg_speeds, agg_headways, agg_gaps, agg_occupancy_frames # NEW: Reset metric lists
    global perspective_editing

    polygon_points.clear()
    poly_editing = False
    polygon_defined = False
    start_processing = False
    line_edit_mode = False
    zone_polygon = None
    polygon_annotator = None
    trace_annotator = None 
    logged_tracker_ids.clear()
    total_class_counts = defaultdict(int)
    flow_direction_mode = "BOTH" 
    coordinates.clear() # Reset historical speed data
    perspective_editing = False
    
    # Reset metric lists
    agg_speeds.clear()
    agg_headways.clear()
    agg_gaps.clear()
    agg_occupancy_frames.clear()

    print("[INFO] All zones and processing state reset.")

def draw_text_simple(frame, text, position, color):
    """Simple wrapper for drawing text on the frame."""
    x, y = position
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
    
def draw_shortcuts(frame, height):
    start_y = height - 70 
    x_offset = 20
    spacing = 200 
    
    cv2.putText(frame, "SHORTCUTS:", (x_offset, start_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
    x_offset += 110
    
    current_x = x_offset
    for i, (key, action) in enumerate(SHORTCUTS.items()):
        text = f"{key}: {action}"
        if i % 4 == 0 and i != 0: 
            start_y += 20
            current_x = x_offset
            
        cv2.putText(frame, text, (current_x, start_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1, cv2.LINE_AA)
        current_x += spacing


def draw_ui(frame, width, height, current_line_zone, current_polygon_points):
    """Draws all UI elements (buttons, line, polygon points, shortcuts, and counts)."""
    global total_class_counts, flow_direction_mode, line_edit_mode
    
    # 1. Draw Buttons 
    for name, coords in BUTTONS.items():
        x_min, y_min, x_max, y_max = coords
        
        color_rgb = BUTTON_COLOR_INACTIVE
        if (name == 'poly' and poly_editing) or            (name == 'line' and line_edit_mode) or            (name == 'start' and start_processing): 
            color_rgb = BUTTON_COLOR_ACTIVE
        
        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color_rgb, -1)
        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (255, 255, 255), 2)
        
        text_x = x_min + 5
        text_y = y_min + 35
        cv2.putText(frame, name.capitalize(), (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)


 # 2. Draw Current Line Zone 
    if current_line_zone is not None:
        y_coord = current_line_zone.vector.start.y 
        line_color = (255, 255, 255)
        if line_edit_mode:
            line_color = (0, 255, 255) 
        
        cv2.line(
            frame,
            (0, y_coord),  
            (width, y_coord), 
            line_color, 2
        )

         # Draw the centered IN/OUT counter box manually 
        raw_in_count = (current_line_zone.in_count + global_in_offset) if current_line_zone else 0
        raw_out_count = (current_line_zone.out_count + global_out_offset) if current_line_zone else 0

        in_count = raw_in_count if flow_direction_mode in ["IN", "BOTH"] else 0
        out_count = raw_out_count if flow_direction_mode in ["OUT", "BOTH"] else 0
        
        box_x = int(width / 2) - 35
        box_y = y_coord
        
        box_width = 80
        box_height = 40
        cv2.rectangle(frame, (box_x, box_y - box_height), (box_x + box_width, box_y), (255, 255, 255), -1) 
        
        cv2.putText(frame, f"in: {in_count}", (box_x + 5, box_y - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)
        cv2.putText(frame, f"out: {out_count}", (box_x + 5, box_y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)

    # 3. Draw Polygon Points 
    if poly_editing or polygon_defined:
        for i, (px, py) in enumerate(current_polygon_points):
            cv2.circle(frame, (px, py), 5, (0, 255, 255), -1)
            draw_text_simple(frame, str(i+1), (px + 10, py + 10), (0, 255, 255))
        
        if len(current_polygon_points) > 1:
            pts = np.array(current_polygon_points, np.int32)
            pts = pts.reshape((-1, 1, 2))
            is_closed = polygon_defined
            line_color = (0, 255, 255) if poly_editing else (255, 255, 255)
            cv2.polylines(frame, [pts], is_closed, line_color, 2)

    # 4. Draw Perspective Points (NEW)
    global perspective_points
    # FIX: Only draw the perspective points when actively editing to reduce clutter
    if perspective_editing:
        # Draw perspective points and bounding box
        for i, (px_float, py_float) in enumerate(perspective_points):
            # FIX: Convert coordinates to integers for all drawing functions
            px_int = int(px_float)
            py_int = int(py_float)
            
            color = (255, 0, 255) # Magenta for perspective points
            cv2.circle(frame, (px_int, py_int), 5, color, -1)
            
            # Label point order (TL, TR, BR, BL)
            label = ['TL', 'TR', 'BR', 'BL'][i] if i < 4 else 'Extra'
            draw_text_simple(frame, label, (px_int + 10, py_int + 10), color)
            
        if len(perspective_points) >= 2:
            # Draw the connecting lines
            pts = np.array(perspective_points, np.int32)
            pts = pts.reshape((-1, 1, 2))
            cv2.polylines(frame, [pts], len(perspective_points) == 4, (255, 0, 255), 2)

    # 5. Draw Status Message  
    status = ""
    if perspective_editing:
        order = ['TL', 'TR', 'BR', 'BL']
        next_point = order[len(perspective_points)] if len(perspective_points) < 4 else 'Done'
        status = f"STATUS: Perspective Calibration (Click {len(perspective_points)+1}/4: Next={next_point}) - Processing Still Running"
    elif poly_editing:
        status = "STATUS: Polygon Edit Mode (Click to Add, ENTER to Finish) - Processing Still Running"
    elif line_edit_mode:
        status = "STATUS: Line Edit Mode (Click to Set Y) - Processing Still Running"
    elif start_processing:
        status = "STATUS: Processing (Press S to Restart)"
    else:
        status = "STATUS: Initial State (Define Zones/Perspective and Press S)"
        
    draw_text_simple(frame, status, (20, height - 90), (112, 25, 25))
    
    # 5. Draw Shortcuts 
    draw_shortcuts(frame, height)
    
    # 6. Draw Class Counts 
    
    flow_color = (0, 255, 0) if flow_direction_mode == "IN" else (175, 196, 104) if flow_direction_mode == "OUT" else (255, 255, 0)
    draw_text_simple(frame, f"FLOW MODE: {flow_direction_mode} (Press D to Toggle)", (20, 100), flow_color)
    
    y_offset = 130 
    draw_text_simple(frame, "TOTAL COUNTS:", (20, y_offset), (0, 255, 255))
    y_offset += 30
    
    for class_name in SUMMARY_CLASSES:
        count = total_class_counts[class_name]
        color = CLASS_COLOR.get(class_name, (255, 255, 255))
        draw_text_simple(frame, f"{class_name}: {count}", (30, y_offset), color)
        y_offset += 30
        
    if current_line_zone:
        raw_total_in = current_line_zone.in_count + global_in_offset
        raw_total_out = current_line_zone.out_count + global_out_offset
        
        # Apply flow direction filter for display
        total_in = raw_total_in if flow_direction_mode in ["IN", "BOTH"] else 0
        total_out = raw_total_out if flow_direction_mode in ["OUT", "BOTH"] else 0
        
        draw_text_simple(frame, f"IN: {total_in}", (30, y_offset), (97, 212, 101))
        y_offset += 30
        draw_text_simple(frame, f"OUT: {total_out}", (30, y_offset), (70, 173, 242))
    
    return frame

# --- NEW GUI ELICITATION (Unchanged) ---

def elicit_time_gui():
    """Opens a modal tkinter window to elicit run date and start time."""
    
    results = {'date': None, 'time': None}
    
    dialog = tk.Toplevel(root)
    dialog.title("Enter Run Time")
    dialog.attributes('-topmost', True) 
    
    def validate_and_submit():
        date_str = date_entry.get()
        time_str = time_entry.get()
        
        try:
            datetime.strptime(date_str, "%Y-%m-%d")
            datetime.strptime(time_str, "%H:%M:%S")
            
            results['date'] = date_str
            results['time'] = time_str
            dialog.destroy()
            
        except ValueError:
            messagebox.showerror("Invalid Format", "Date must be YYYY-MM-DD and Time must be HH:MM:SS (24-hour).", parent=dialog)
            
    tk.Label(dialog, text="Run Date (YYYY-MM-DD):").grid(row=0, column=0, padx=5, pady=5, sticky="w")
    date_entry = tk.Entry(dialog)
    date_entry.grid(row=0, column=1, padx=5, pady=5)
    
    tk.Label(dialog, text="Start Time (HH:MM:SS):").grid(row=1, column=0, padx=5, pady=5, sticky="w")
    time_entry = tk.Entry(dialog)
    time_entry.grid(row=1, column=1, padx=5, pady=5)
    
    submit_button = tk.Button(dialog, text="Set Time & Continue", command=validate_and_submit)
    submit_button.grid(row=2, column=0, columnspan=2, pady=10)

    root.deiconify()
    
    dialog.transient(root) 
    dialog.grab_set() 
    root.wait_window(dialog) 
    root.withdraw()

    return results

def elicit_start_time(stream_source: str) -> bool:
    """Elicits the run date and start time from the user.
       Returns True if time is successfully set, False otherwise."""
    
    global run_date, run_start_time, global_offset_dt, root, start_dt
    
    # Initialize root window if it doesn't exist
    if root is None:
        root = tk.Tk()
        root.withdraw()  # Hide the root window initially
    
    root.deiconify() 
    time_data = elicit_time_gui()
    root.withdraw()

    if time_data['date'] and time_data['time']:
        try:
            run_date = datetime.strptime(time_data['date'], "%Y-%m-%d").strftime("%Y-%m-%d")
            run_start_time = datetime.strptime(time_data['time'], "%H:%M:%S").strftime("%H:%M:%S")

            start_dt = datetime.strptime(f"{run_date} {run_start_time}", "%Y-%m-%d %H:%M:%S")
            global_offset_dt = start_dt

            config_data = load_config(stream_source)
            config_data['run_date'] = run_date
            config_data['run_start_time'] = run_start_time
            save_config(stream_source, config_data)
            
            print(f"[INFO] Run time set: {run_date} {run_start_time}")
            return True
        
        except ValueError as e:
            messagebox.showerror("Time Format Error", f"Invalid date or time format. Please use YYYY-MM-DD and HH:MM:SS. Error: {e}")
            return False
    else:
        return False


def initial_start_and_reset(cap, stream_source, frame_width, frame_height):
    """Resets counters and starts processing. Used for initial start or full reset."""
    global start_processing, line_zone, logged_tracker_ids, total_class_counts, run_date, run_start_time, start_dt, last_publish_time
    global agg_speeds, agg_headways, agg_gaps, agg_occupancy_frames, stream_start_time, is_live_stream, global_offset_dt # NEW: Reset metric lists

    # 1. Check/Elicit Time
    if is_live_stream:
        # For live streams, use current time automatically
        start_dt = datetime.now()
        run_date = start_dt.strftime("%Y-%m-%d")
        run_start_time = start_dt.strftime("%H:%M:%S")
        global_offset_dt = start_dt
        last_publish_time = start_dt
        stream_start_time = datetime.now()  # Record real-time start for live streams
        
        # Save to config
        config_data = load_config(stream_source)
        config_data['run_date'] = run_date
        config_data['run_start_time'] = run_start_time
        save_config(stream_source, config_data)
        
        print(f"[INFO] Live stream started at: {run_date} {run_start_time}")
    else:
        # For video files, check/elicit time
        if run_date is None or run_start_time is None:
            print("[PROMPT] Configuration time missing. Please enter run time in GUI pop-up.")
            if not elicit_start_time(stream_source):
                print("[WARN] Start aborted: Time not set.")
                return 
        
        if run_date and run_start_time:
            start_dt_str = f"{run_date} {run_start_time}"
            try:
                start_dt = datetime.strptime(start_dt_str, "%Y-%m-%d %H:%M:%S")
                last_publish_time = start_dt
            except ValueError as e:
                print(f"[ERROR] Invalid saved time format: {e}")
                start_dt = None
                return 
    
    # 2. Rewind Video (only for file sources)
    if not is_live_stream:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    # 3. Reset Counters and State
    if line_zone:
        line_y = line_zone.vector.start.y
        update_line_zone(line_y, frame_width)
        
    logged_tracker_ids.clear() 
    total_class_counts = defaultdict(int) 
    
    # Reset metric lists for a fresh 15-minute start
    agg_speeds.clear()
    agg_headways.clear()
    agg_gaps.clear()
    agg_occupancy_frames.clear()

    # 4. Set Processing State
    start_processing = True
    print("[INFO] Processing started and initiated.")


def mouse_callback(event, x, y, flags, param, stream_source, cap, frame_height, frame_width):
    global poly_editing, polygon_points, polygon_defined, line_edit_mode
    global line_zone, start_processing, DISPLAY_SCALE, zone_polygon
    global polygon_annotator, run_date, run_start_time, force_exit, flow_direction_mode
    global agg_speeds, agg_headways, agg_gaps, agg_occupancy_frames
    global perspective_editing, perspective_points, view_transformer

    if DISPLAY_SCALE != 1.0:
        x = int(x / DISPLAY_SCALE)
        y = int(y / DISPLAY_SCALE)

    if perspective_editing:
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(perspective_points) < 4:
                perspective_points.append((x, y))
                print(f"[INFO] Added perspective point ({len(perspective_points)}/4): ({x}, {y})")
            
            if len(perspective_points) == 4:
                # 4 points collected, finalize calibration
                source_array = np.array(perspective_points, dtype=np.float32)
                
                # Re-initialize ViewTransformer with new points
                global TARGET 
                view_transformer = ViewTransformer(source=source_array, target=TARGET)
                perspective_editing = False

                # Save the new config
                config_data = load_config(stream_source) # Load existing data, this also updates perspective_points global
                # Re-save to ensure the new points are persisted
                save_config(stream_source, config_data)
                print("[INFO] Perspective calibration finalized and saved. Order: TL, TR, BR, BL.")

        elif event == cv2.EVENT_RBUTTONDOWN and len(perspective_points) > 0:
            perspective_points.pop()
            print("[INFO] Removed last perspective point.")

    # Handle Polygon Editing
    elif poly_editing:
        if event == cv2.EVENT_LBUTTONDOWN:
            polygon_points.append((x, y))
            print(f"[INFO] Added point: ({x}, {y})")
        elif event == cv2.EVENT_RBUTTONDOWN and len(polygon_points) > 0:
            polygon_points.pop()
            print("[INFO] Removed last point.")
    
    # Handle Line Editing
    elif line_edit_mode:
        if event == cv2.EVENT_LBUTTONDOWN:
            new_line_y = y

            global global_in_offset, global_out_offset
            
            if line_zone:
                global_in_offset += line_zone.in_count
                global_out_offset += line_zone.out_count

            update_line_zone(new_line_y, frame_width)
            line_edit_mode = False 
            
            config_data = load_config(stream_source)
            config_data['polygon_points'] = polygon_points
            config_data['line_y'] = new_line_y
            config_data['run_date'] = run_date
            config_data['run_start_time'] = run_start_time
            save_config(stream_source, config_data)

            print(f"[INFO] Line set at y={new_line_y}. Counts preserved via offset. Ready to start processing.")

    # Handle Button Clicks
    elif event == cv2.EVENT_LBUTTONDOWN:
        if is_point_in_button(x, y, BUTTONS['poly']):
            poly_editing = not poly_editing
            line_edit_mode = False
            perspective_editing = False

            if poly_editing:
                polygon_points.clear()
                polygon_defined = False
                zone_polygon = None
                polygon_annotator = None
                print("[INFO] Polygon edit mode: click to add points (right-click to remove). Press 'Enter' to finish.")
            else:
                print("[INFO] Polygon edit mode OFF.")

        elif is_point_in_button(x, y, BUTTONS['start']):
            if polygon_defined and line_zone is not None:
                initial_start_and_reset(cap, stream_source, frame_width, frame_height)
                line_edit_mode = False
                poly_editing = False
                perspective_editing = False
            else:
                print("[WARN] Define polygon and line zone first.")

        elif is_point_in_button(x, y, BUTTONS['line']):
            line_edit_mode = True
            poly_editing = False
            perspective_editing = False
            print("[INFO] Line edit mode: click to set line.")
            
        elif is_point_in_button(x, y, BUTTONS['exit']):
            force_exit = True
            print("[INFO] Exit button pressed. Terminating.")


# --- MAIN PROCESSING FUNCTION ---

def process_stream(stream_source):
    global line_edit_mode, poly_editing, polygon_points, polygon_defined
    global force_exit, last_minute, flow_direction_mode
    global logged_tracker_ids, start_processing
    global run_date, run_start_time, start_dt, last_publish_time
    global individual_log_file, individual_log_writer, summary_log_file, summary_log_writer
    global frame_width, frame_height, mqtt_client
    global agg_speeds, agg_headways, agg_gaps, agg_occupancy_frames # Include aggregation lists
    global view_transformer, coordinates, trace_annotator
    global perspective_points, perspective_editing, TARGET, DEFAULT_SOURCE
    global TIME_SCALE_FACTOR, is_live_stream, stream_start_time

    period_start_minute = None

    # --- DETERMINE IF LIVE STREAM OR FILE ---
    is_live_stream = (stream_source.startswith('rtsp://') or 
                     stream_source.startswith('http://') or 
                     stream_source.startswith('https://') or
                     stream_source.startswith('rtmp://') or
                     stream_source.startswith('tcp://'))

    # --- INITIALIZATION ---
    # Detect GPU availability and set device accordingly
    if torch.cuda.is_available():
        device = 0  # Use first GPU
        device_name = torch.cuda.get_device_name(0)
        print(f"[INFO] GPU detected: {device_name}. Using GPU for inference.")
    else:
        device = 'cpu'
        print("[INFO] No GPU detected. Falling back to CPU for inference.")
    
    model = YOLO(MODEL_PATH)
    
    # Configure capture options for RTSP streams
    if is_live_stream and stream_source.startswith('rtsp://'):
        os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = "rtsp_transport;tcp|max_delay;0|reorder_queue_size;0|stimeout;3000000|fflags;nobuffer|flags;low_delay"
    
    cap = cv2.VideoCapture(stream_source)
    
    # Set buffer size for low latency (live streams)
    if is_live_stream:
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, STREAM_BUFFER_SIZE)
            # Additional properties for low latency
            cap.set(cv2.CAP_PROP_FPS, 30)  # Set expected FPS
        except Exception:
            pass
    
    if not cap.isOpened():
        print(f"Error: Could not open stream source {stream_source}")
        return
    
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) 
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # For live streams, FPS might be 0, use default
    if fps <= 0 or fps > 120:
        fps = 30.0  # Default FPS for live streams
        print(f"[INFO] FPS not available or invalid, using default: {fps}")
    
    tracker_fps = min(fps, 30)
    
    # Get identifier for logs/config (set before calling load_config)
    if is_live_stream:
        stream_id = STREAM_NAME
        stream_id_folder = stream_id
    else:
        stream_id = os.path.splitext(os.path.basename(stream_source))[0]
        stream_id_folder = stream_id
    
    # Load config after is_live_stream is set
    config = load_config(stream_source)
    run_date = config.get('run_date')
    run_start_time = config.get('run_start_time')
    
    # --- PERSPECTIVE AND VIEWTRAFORMER SETUP (NEW LOGIC) ---
    if len(perspective_points) != 4:
        # If not loaded from config, use default and inform user
        perspective_points = [tuple(p) for p in DEFAULT_SOURCE]
        print("[WARN] Using default perspective points. Press 'C' to calibrate for this video.")

    # Initialize view_transformer with current perspective_points
    source_array = np.array(perspective_points, dtype=np.float32)
    view_transformer = ViewTransformer(source=source_array, target=TARGET)
    coordinates.clear() 
    # --- END PERSPECTIVE SETUP ---
    
    open_log_files(stream_source) 
    connect_mqtt()

    byte_tracker = sv.ByteTrack(frame_rate=tracker_fps)

    thickness = sv.calculate_optimal_line_thickness(
        resolution_wh=(frame_width, frame_height)
    )
    trace_annotator = sv.TraceAnnotator(
        thickness=thickness,
        trace_length=int(fps * 4), # 2 seconds of trace
        position=sv.Position.BOTTOM_CENTER,
    )

    # --- ZONE DEFINITION ---
    line_y = config.get('line_y')
    if line_y is None:
        line_y_raw = 1500 
        if frame_height < 1080:
            line_y = int(frame_height * 0.75)
        elif frame_height < 2160:
            line_y = min(line_y_raw, int(frame_height * 0.75))
        else:
            line_y = line_y_raw

    update_line_zone(line_y, frame_width) 
    
    poly_points = config.get('polygon_points')
    if poly_points is None or len(poly_points) < 3:
        default_poly_points = [
            (int(frame_width * 0.20), int(frame_height * 0.50)), 
            (int(frame_width * 0.80), int(frame_height * 0.50)),
            (frame_width, frame_height), 
            (0, frame_height) 
        ]
        poly_points = default_poly_points
    
    polygon_points = poly_points 
    
    if len(polygon_points) >= 3:
        polygon_defined = True
        polygon_array = np.array(polygon_points)
        zone_polygon = PolygonZone(polygon=polygon_array)
        polygon_annotator = sv.PolygonZoneAnnotator(zone=zone_polygon, color=sv.Color.WHITE, thickness=2, opacity=0.1) 
    else:
        polygon_defined = False
    
    # Time tracking variables
    last_minute = None
    minute_class_counts_1min = defaultdict(int)
    minute_class_counts_15min = defaultdict(int)
    period_start_minute = None
    
    # Setup OpenCV window and mouse callback
    cv2.namedWindow('Processed Video')
    def local_mouse_callback(event, x, y, flags, param):
        mouse_callback(event, x, y, flags, param, stream_source, cap, frame_width, frame_height)
    cv2.setMouseCallback('Processed Video', local_mouse_callback)
    
    current_dt = datetime.now()
    read_failures = 0
    reconnect_attempts = 0
    
    while True: 
        # Check for exit flag at the start of each iteration
        if force_exit:
            print("[INFO] Exit requested. Terminating.")
            break
        
        if start_processing:
            # For live streams, drop old buffered frames to get the latest one
            if is_live_stream:
                # Grab the latest frame, dropping any buffered ones
                ret = False
                frame = None
                for _ in range(10):  # Try to get the latest frame (drop up to 9 old ones)
                    temp_ret, temp_frame = cap.read()
                    if temp_ret:
                        ret = temp_ret
                        frame = temp_frame
                    else:
                        break
            else:
                ret, frame = cap.read()
            
            # Handle stream reconnection for live sources
            if not ret and is_live_stream:
                if force_exit:
                    break
                    
                read_failures += 1
                if read_failures % 30 == 0:  # Log every 30 failures
                    print(f"[WARN] Failed to read frame from stream (failures: {read_failures})")
                
                if reconnect_attempts < STREAM_MAX_RECONNECT_ATTEMPTS:
                    reconnect_attempts += 1
                    print(f"[INFO] Attempting to reconnect to stream (attempt {reconnect_attempts}/{STREAM_MAX_RECONNECT_ATTEMPTS})...")
                    cap.release()
                    time.sleep(STREAM_RECONNECT_DELAY)
                    
                    # Reopen stream
                    if stream_source.startswith('rtsp://'):
                        os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = "rtsp_transport;tcp|max_delay;0|reorder_queue_size;0|stimeout;3000000|fflags;nobuffer|flags;low_delay"
                    cap = cv2.VideoCapture(stream_source)
                    if is_live_stream:
                        try:
                            cap.set(cv2.CAP_PROP_BUFFERSIZE, STREAM_BUFFER_SIZE)
                            cap.set(cv2.CAP_PROP_FPS, 30)  # Set expected FPS
                        except Exception:
                            pass
                    
                    if cap.isOpened():
                        print("[INFO] Stream reconnected successfully.")
                        read_failures = 0
                    else:
                        print("[ERROR] Failed to reconnect to stream.")
                else:
                    print("[ERROR] Max reconnection attempts reached. Exiting.")
                    force_exit = True
                    break
                
                continue  # Skip processing this iteration
            elif not ret:
                # For files, end of video
                if force_exit:
                    break
                print("[INFO] End of video or stream closed.")
                break
            else:
                # Successfully read frame
                read_failures = 0
                reconnect_attempts = 0
        else:
            # For live streams, drop old buffered frames to get the latest one
            if is_live_stream:
                # Grab the latest frame, dropping any buffered ones
                ret = False
                frame = None
                for _ in range(10):  # Try to get the latest frame (drop up to 9 old ones)
                    temp_ret, temp_frame = cap.read()
                    if temp_ret:
                        ret = temp_ret
                        frame = temp_frame
                    else:
                        break
            else:
                ret, frame = cap.read()
            
            if not ret:
                if is_live_stream:
                    continue  # Keep trying for live streams
                else:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Rewind for files
                    ret, frame = cap.read()
                    if not ret or force_exit:
                        break
            elif not is_live_stream:
                # For files, show first frame when not processing
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                
        if period_start_minute is None and start_processing:
            period_start_minute = current_dt.strftime("%H:%M")
            
        # --- INITIALIZATION BLOCK ---
        ids_to_process = []
        newly_counted_in_ids = set()
        newly_counted_out_ids = set()

        in_ids = set()  

        # --- TIME ALIGNMENT ---
        if is_live_stream:
            # For live streams, use real-time
            if start_dt and stream_start_time:
                # Calculate elapsed real-time since stream start
                real_elapsed = (datetime.now() - stream_start_time).total_seconds()
                current_dt = start_dt + timedelta(seconds=real_elapsed)
            else:
                current_dt = datetime.now()
            current_minute = current_dt.strftime("%H:%M")
            frame_number = 0  # Not applicable for live streams
        else:
            # For files, use frame-based time
            frame_number = cap.get(cv2.CAP_PROP_POS_FRAMES)
            if start_dt and fps > 0:
                time_elapsed_seconds = (frame_number - 1) / fps 
                current_dt = start_dt + timedelta(seconds=time_elapsed_seconds)
                current_minute = current_dt.strftime("%H:%M")
            else:
                current_dt = datetime.now() 
                current_minute = current_dt.strftime("%H:%M") 

        # --- DRAW UI ---
        # Safety check: ensure frame is valid before processing
        if frame is None:
            continue
            
        display = draw_ui(frame.copy(), frame_width, frame_height, line_zone, polygon_points)
        
        if start_processing: 
            
            # --- MODEL PREDICTION & TRACKING ---
            results = model(frame, device=device, verbose=False)[0]
            detections = Detections.from_ultralytics(results)
            detections = detections[detections.confidence >= MIN_CONF]
            detections = detections[np.isin(detections.class_id, list(CLASS_MAP.keys()))]
            detections = byte_tracker.update_with_detections(detections)
            mask = detections.tracker_id != -1
            mask = np.array(mask, dtype=bool)
            detections = detections[mask]

            all_tracked_detections = copy.deepcopy(detections)
            if all_tracked_detections.tracker_id is not None:
                all_tracked_detections.class_id = all_tracked_detections.class_id.astype(int)
                all_tracked_detections.tracker_id = all_tracked_detections.tracker_id.astype(int)
            
            # 5. SPEED CALCULATION
            
            if view_transformer is not None and len(perspective_points) == 4:
                # Get bottom-center anchor points for perspective transform
                points = all_tracked_detections.get_anchors_coordinates(
                    anchor=sv.Position.BOTTOM_CENTER
                )
                # Transform the points to Bird's Eye View
                transformed_points = view_transformer.transform_points(points=points).astype(int)

                # Store the transformed Y coordinate for speed calculation
                for tracker_id, [_, y] in zip(all_tracked_detections.tracker_id, transformed_points):
                    coordinates[tracker_id].append(y)

            # 6. APPLY POLYGON FILTERING
            if polygon_defined and zone_polygon is not None:
                is_in_polygon = zone_polygon.trigger(detections=detections)
                is_in_polygon = np.array(is_in_polygon, dtype=bool)
                detections = detections[is_in_polygon]

            # 7. Annotation (Bounding Boxes)
            if len(detections) > 0:

                detections.class_id = detections.class_id.astype(int)
    
                if detections.tracker_id is not None:
                    detections.tracker_id = detections.tracker_id.astype(int)
    
                    for xyxy, confidence, class_id, tracker_id in zip(
                        detections.xyxy, detections.confidence, detections.class_id, detections.tracker_id
                    ):
                        x1, y1, x2, y2 = [int(i) for i in xyxy]
            
                        class_name = CLASS_MAP.get(class_id, 'UNKNOWN')
                        color = CLASS_COLOR.get(class_name, (255, 255, 255))
                        
                        # --- Speed Calculation and Label Generation (Modified for PIXELS_PER_METER_SCALE and TIME_SCALE_FACTOR) ---
                        speed_label = ""
                        # Only attempt speed calc if view_transformer is initialized and we have data
                        if view_transformer is not None and len(perspective_points) == 4 and len(coordinates[tracker_id]) > fps / 2:
                            coordinate_start = coordinates[tracker_id][-1] # latest
                            coordinate_end = coordinates[tracker_id][0] # oldest
                            
                            # Distance in BEV pixels
                            distance_pixels = abs(coordinate_start - coordinate_end) 
                            
                            # Convert to real-world distance (meters) using the scale factor
                            distance_meters = distance_pixels * PIXELS_PER_METER_SCALE 
                            
                            # CORRECTED: Apply TIME_SCALE_FACTOR to account for video-to-real-time difference
                            time_secs = (len(coordinates[tracker_id]) / fps) * TIME_SCALE_FACTOR
                            
                            if time_secs > 0:
                                # Speed in m/s, converted to km/h (m/s * 3.6 = km/h)
                                speed = (distance_meters / time_secs) * 3.6 
                            else:
                                speed = 0
                            
                            speed_label = f" {int(speed)} km/h"

                        label = f"#{class_name}{speed_label}"

                        # Draw Bounding Box
                        cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)
            
                        # Draw Label Background 
                        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                        cv2.rectangle(display, (x1, y1 - h - 10), (x1 + w, y1), color, -1)
            
                        # Draw Label Text 
                        cv2.putText(
                            display, 
                            label, 
                            (x1, y1 - 5), 
                            cv2.FONT_HERSHEY_SIMPLEX, 
                            0.6, 
                            (0, 0, 0), # Text color is black for contrast
                            2
                        )
            
            # 8. Update Line Counter and Log Events
            if all_tracked_detections.tracker_id is not None and len(all_tracked_detections) > 0:
                
                tracked_detections_for_line = all_tracked_detections
                tracked_detections_for_line.class_id = tracked_detections_for_line.class_id.astype(int)
                tracked_detections_for_line.tracker_id = tracked_detections_for_line.tracker_id.astype(int)
                
                in_mask, out_mask = line_zone.trigger(detections=tracked_detections_for_line)
                
                current_in_ids_full = set(tracked_detections_for_line.tracker_id[in_mask])
                current_out_ids_full = set(tracked_detections_for_line.tracker_id[out_mask])

                current_in_ids = current_in_ids_full
                current_out_ids = current_out_ids_full

            else:
                current_in_ids = set()
                current_out_ids = set()

            if flow_direction_mode == 'IN':
                # If only 'IN' is allowed, zero out any 'OUT' crossings
                current_out_ids = set() 
            elif flow_direction_mode == 'OUT':
                # If only 'OUT' is allowed, zero out any 'IN' crossings
                current_in_ids = set()

            newly_counted_in_ids = current_in_ids - logged_tracker_ids
            newly_counted_out_ids = current_out_ids - logged_tracker_ids

            ids_to_process = []
            if flow_direction_mode == "BOTH" or flow_direction_mode == "IN":
                ids_to_process.extend(list(newly_counted_in_ids)) 
            if flow_direction_mode == "BOTH" or flow_direction_mode == "OUT":
                ids_to_process.extend(list(newly_counted_out_ids))
            
            in_ids = newly_counted_in_ids 
            current_tracker_ids = set(detections.tracker_id) if detections.tracker_id is not None else set()

            for tracker_id in ids_to_process:
                idx = np.where(all_tracked_detections.tracker_id == tracker_id)[0]

                if len(idx) > 0:
                    idx = idx[0]
                    bbox = all_tracked_detections.xyxy[idx] 
                    class_id = all_tracked_detections.class_id[idx]
                    class_name = CLASS_MAP.get(int(class_id), "UNKNOWN")
                    direction = "IN" if tracker_id in in_ids else "OUT"
                    confidence = all_tracked_detections.confidence[idx]

                    # Calculate video time for logging
                    if is_live_stream:
                        # For live streams, use real elapsed time
                        if stream_start_time:
                            elapsed = (datetime.now() - stream_start_time).total_seconds()
                        else:
                            elapsed = 0
                        hours, remainder = divmod(elapsed, 3600)
                        minutes, seconds = divmod(remainder, 60)
                        video_time_str = f"{int(hours):02d}:{int(minutes):02d}:{seconds:06.3f}"
                    else:
                        frame_num = cap.get(cv2.CAP_PROP_POS_FRAMES)
                        time_elapsed_seconds = (frame_num - 1) / fps if fps > 0 else 0
                        hours, remainder = divmod(time_elapsed_seconds, 3600)
                        minutes, seconds = divmod(remainder, 60)
                        video_time_str = f"{int(hours):02d}:{int(minutes):02d}:{seconds:06.3f}"
                    
                    timestamp_for_crop = current_dt.strftime("%Y%m%d_%H%M%S%f")[:-3]
                    image_file_name = f"{class_name}_{int(tracker_id)}.jpg"

                    if current_dt:
                        write_individual_log(individual_log_writer, current_dt, tracker_id, class_id, direction, confidence, video_time_str, image_file_name)
                    
                    if current_dt:
                        log_cropped_image(frame, bbox, tracker_id, current_dt, image_file_name, stream_id_folder)
                    
                    logged_tracker_ids.add(tracker_id)
                    total_class_counts[class_name] += 1

                    if class_name in SUMMARY_CLASSES:
                        minute_class_counts_1min[class_name] += 1
                        minute_class_counts_15min[class_name] += 1 
                        
                    # --- METRIC AGGREGATION ---
                    # Calculate speed when vehicle crosses the line
                    vehicle_speed = 0.0
                    if view_transformer is not None and len(perspective_points) == 4 and len(coordinates[tracker_id]) > fps / 2:
                        coordinate_start = coordinates[tracker_id][-1]  # latest
                        coordinate_end = coordinates[tracker_id][0]     # oldest
                        distance_pixels = abs(coordinate_start - coordinate_end)
                        distance_meters = distance_pixels * PIXELS_PER_METER_SCALE
                        time_secs = (len(coordinates[tracker_id]) / fps) * TIME_SCALE_FACTOR
                        if time_secs > 0:
                            vehicle_speed = (distance_meters / time_secs) * 3.6  # km/h
                    
                    agg_speeds.append(vehicle_speed)
                    agg_headways.append(0.0) 
                    agg_gaps.append(0.0) 
                    # --- END METRIC AGGREGATION ---


            logged_tracker_ids = logged_tracker_ids.intersection(current_tracker_ids) 
            
            # --- OCCUPANCY PLACEHOLDER (Per Frame) ---
            # Occupancy requires calculating the area of bounding boxes relative to the zone area.
            # For simplicity, we append 0.0 for now.
            agg_occupancy_frames.append(0.0)
            # --- END OCCUPANCY PLACEHOLDER ---


            # --- SUMMARY LOGGING ROLLOVER (1 MINUTE) ---
            if start_dt and current_minute != last_minute:
                if last_minute is not None and summary_log_writer:
                    summary_period = f"{period_start_minute} - {last_minute}"
                    summary_log_writer.writerow([summary_period] + [minute_class_counts_1min[c] for c in SUMMARY_CLASSES])
                    summary_log_file.flush()
                    
                    minute_class_counts_1min = defaultdict(int)
                    
                last_minute = current_minute
                if period_start_minute is None:
                    period_start_minute = current_minute

            # --- MQTT PUBLISHING ROLLOVER (15 MINUTES) ---
            if start_dt and last_publish_time and current_dt >= (last_publish_time + timedelta(minutes=PUBLISH_INTERVAL_MINUTES)):
                
                # Calls publish_15min_report, which calculates averages and resets agg_ lists
                publish_15min_report(stream_source, current_dt, minute_class_counts_15min)

                # Reset the 15-minute counter and update the last publish time
                minute_class_counts_15min = defaultdict(int)
                last_publish_time = current_dt

        
        # --- ANNOTATE LINE AND POLYGON ---
        if polygon_defined and polygon_annotator is not None:
            display = polygon_annotator.annotate(scene=display)
        
        # --- ANNOTATE TRACES ---
        if start_processing and trace_annotator is not None:
            display = trace_annotator.annotate(scene=display, detections=detections)

        # --- DISPLAY AND KEY HANDLING ---
        
        if DISPLAY_SCALE != 1.0:
            display = cv2.resize(display, (int(frame_width * DISPLAY_SCALE), int(frame_height * DISPLAY_SCALE)))

        # For live streams, use minimal wait time to reduce latency (both when processing and not)
        if is_live_stream:
            wait_time_ms = 1  # Minimal wait for live streams
        elif fps > 0 and start_processing: 
            wait_time_ms = max(1, int(1000 / fps)) 
        else: 
            wait_time_ms = 100 

        cv2.imshow('Processed Video', display)

        k = cv2.waitKey(wait_time_ms) & 0xFF

        if k == 27: # ESC key
            force_exit = True
            break
            
        elif k == ord('l') or k == ord('L'): # L key (Edit Line)
            line_edit_mode = not line_edit_mode
            poly_editing = False
            print(f"[INFO] Line edit mode {'ON' if line_edit_mode else 'OFF'}.")

        elif k == ord('c') or k == ord('C'): # C key (Calibrate Perspective) (NEW)
            perspective_editing = not perspective_editing
            poly_editing = False
            line_edit_mode = False
            
            if perspective_editing:
                perspective_points.clear()
                print("[INFO] Perspective Calibration mode ON. Click 4 points in order: Top-Left, Top-Right, Bottom-Right, Bottom-Left.")
            else:
                print("[INFO] Perspective Calibration mode OFF (Cancelled).")
        
        elif k == ord('s') or k == ord('S'): # S key (Start/Restart)
            if polygon_defined and line_zone is not None:
                initial_start_and_reset(cap, stream_source, frame_width, frame_height) 
            else:
                print("[WARN] Define polygon and line zone first.")

        elif k == ord('r') or k == ord('R'): # R key (Reset All)
            reset_all()

        elif k == ord('d') or k == ord('D'): # D key (Toggle Flow Direction)
            if flow_direction_mode == "BOTH":
                flow_direction_mode = "IN"
            elif flow_direction_mode == "IN":
                flow_direction_mode = "OUT"
            elif flow_direction_mode == "OUT":
                flow_direction_mode = "BOTH"
            print(f"[INFO] Flow direction mode set to: {flow_direction_mode}")

        elif k == ord('a') or k == ord('A'): # A key (Auto Line)
            if start_processing and 'detections' in locals() and detections and hasattr(detections, "xyxy") and detections.xyxy.size > 0:
                ys = detections.xyxy[:, 3] 
                new_line_y = int(np.mean(ys))
                
                global global_in_offset, global_out_offset
                if line_zone:
                    global_in_offset += line_zone.in_count
                    global_out_offset += line_zone.out_count

                update_line_zone(new_line_y, frame_width)
                
                # Save the new line config immediately
                config_data = load_config(stream_source)
                config_data['polygon_points'] = polygon_points
                config_data['line_y'] = line_zone.vector.start.y
                config_data['run_date'] = run_date
                config_data['run_start_time'] = run_start_time
                save_config(stream_source, config_data)
                
                print(f"[INFO] Auto line set at y={new_line_y}")
            else:
                print("[WARN] Start processing first or no detections found for Auto Line.")
                
        elif k == 13: # Enter key (Finalize Poly)
            if poly_editing and len(polygon_points) >= 3:
                poly_editing = False
                polygon_defined = True
                print("[INFO] Polygon definition finalized.")
                
                polygon_array = np.array(polygon_points)
                zone_polygon = PolygonZone(polygon=polygon_array)
                polygon_annotator = sv.PolygonZoneAnnotator(zone=zone_polygon, color=sv.Color.WHITE, thickness=2, opacity=0.1) 
                
                config_data = load_config(stream_source)
                config_data['polygon_points'] = polygon_points
                if line_zone:
                    config_data['line_y'] = line_zone.vector.start.y
                config_data['run_date'] = run_date
                config_data['run_start_time'] = run_start_time
                save_config(stream_source, config_data)

            elif poly_editing and len(polygon_points) < 3:
                print("[WARN] Need at least 3 points to define a polygon.")

        # Check for exit after key handling
        if force_exit:
            break

    # Final summary log entry
    if summary_log_writer is not None and last_minute is not None:
        minute_end = current_dt.strftime("%H:%M") 
        summary_period = f"{period_start_minute} - {minute_end}"
        summary_log_writer.writerow([summary_period] + [minute_class_counts_1min[c] for c in SUMMARY_CLASSES])
    
    cap.release()
    cv2.destroyAllWindows()
    close_log_files()
    
    if mqtt_client:
        print("[INFO] Disconnecting MQTT client.")
        mqtt_client.loop_stop()
        mqtt_client.disconnect()
    
    # Clean up tkinter root window
    global root
    if root is not None:
        root.destroy()
        root = None

    print("[INFO] Processing finished. Logs saved.")


def get_stream_url():
    """Get stream URL from user input or configuration."""
    global STREAM_URL, USE_LIVE_STREAM, root
    
    # Initialize root window if it doesn't exist
    if root is None:
        root = tk.Tk()
        root.withdraw()
    
    if USE_LIVE_STREAM:
        if STREAM_URL:
            # Use configured URL
            stream_source = STREAM_URL
            print(f"[INFO] Using configured stream URL: {stream_source}")
        else:
            # Prompt user for stream URL
            stream_source = simpledialog.askstring(
                "Stream URL",
                "Enter stream URL (RTSP, HTTP, etc.):\nExample: rtsp://user:pass@ip:port/stream",
                initialvalue="rtsp://"
            )
        
        if stream_source:
            return stream_source
        else:
            print("[INFO] No stream URL provided. Exiting.")
            return None
    else:
        # Use file dialog
        video_file_path = filedialog.askopenfilename(
            title="Select Video File to Process",
            filetypes=[("Video files", "*.mp4 *.avi *.mov")]
        )
        return video_file_path


if __name__ == "__main__":
    stream_source = get_stream_url()
    
    if stream_source:
        print(f"[INFO] Processing source: {stream_source}")
        process_stream(stream_source)
    else:
        print("[INFO] No source selected. Exiting.")