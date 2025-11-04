import cv2
import numpy as np
import os
import csv
import tkinter as tk
import json
import copy
import tkinter.simpledialog as simpledialog
from tkinter import filedialog
from datetime import datetime, timedelta
from collections import defaultdict, OrderedDict, deque
from ultralytics import YOLO
import supervision as sv 
from supervision.detection.tools.polygon_zone import PolygonZone 
from supervision.draw.utils import draw_text 
from supervision.detection.core import Detections
from supervision.detection.line_zone import LineZone
from tkinter import messagebox 

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
PIXELS_PER_METER_SCALE = 0.5 # <<--- ADJUST THIS VALUE FOR ACCURACY

# 4. TIME_SCALE_FACTOR: Correction for videos where 1 video second != 1 real-world second.
# Use 1.5 since your video runs slower (1 video second = 1.5 real seconds).
TIME_SCALE_FACTOR = 1.5 # <<--- YOUR CORRECTION FACTOR HERE

# --- END SPEED ESTIMATION CONSTANTS ---

# --- CONFIG ---
MODEL_PATH = "roboflow.pt"
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "rnd")
# UPDATED: Use Malay class names as per user's log examples
CLASS_MAP = {0: 'Kelas 1', 1: 'Kelas 2', 2: 'Kelas 3', 3: 'Kelas 4', 4: 'Kelas 5', 5: 'Kelas 6'}
SUMMARY_CLASSES = ['Kelas 1', 'Kelas 2', 'Kelas 3', 'Kelas 4', 'Kelas 5', 'Kelas 6']
CLASS_COLOR = {'Kelas 1': (186,179,255), 'Kelas 2': (186,223,255), 'Kelas 3': (186,255,255),
               'Kelas 4': (201,255,186), 'Kelas 5': (255,255,186), 'Kelas 6': (255,203,241)}
# Updated BUTTONS for horizontal centering
BUTTONS = {'poly': (770, 20, 850, 60), 
           'start': (870, 20, 950, 60), 
           'line': (970, 20, 1050, 60), 
           'exit': (1070, 20, 1150, 60)} 
MIN_CONF = 0.15
FAST_MODE_FPS = 15
DISPLAY_SCALE = 0.75

BUTTON_COLOR_INACTIVE = (50, 50, 50)
BUTTON_COLOR_ACTIVE = (0, 165, 255) 

# Dictionary for on-screen shortcut display
SHORTCUTS = OrderedDict([
    ("ESC / EXIT", "Exit App"),
    ("ENTER", "Finalize Poly"),
    ("R", "Reset All"),
    ("S", "Start/Restart"),
    ("L", "Edit Line"),
    ("C", "Calibrate Perspective"), # NEW Shortcut
    ("A", "Auto Line"),
    ("D", "Flow Mode")
])

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

# New globals for persistent time
run_date = None
run_start_time = None
force_exit = False 
root = None 
start_dt = None

global_in_offset = 0
global_out_offset = 0

last_in_count = 0
last_out_count = 0
tracker_direction_map = {} 

# Log file handles and counters
individual_log_file = None
individual_log_writer = None
summary_log_file = None
summary_log_writer = None

logged_tracker_ids = set() 
total_class_counts = defaultdict(int) 
flow_direction_mode = "BOTH"

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

def get_config_path(video_path):
    """Generates the path for the config JSON file based on the video path."""
    video_name = os.path.basename(video_path)
    config_name = f"{os.path.splitext(video_name)[0]}_config.json"
    return os.path.join(OUT_DIR, config_name)

def load_config(video_path):
    """Loads configuration data from a JSON file."""
    config_path = get_config_path(video_path)
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
                print(f"[INFO] Loaded configuration from {config_path}")
                
                # Load perspective points
                global perspective_points
                loaded_perspective = config.get('perspective_points')
                if loaded_perspective and len(loaded_perspective) == 4:
                    perspective_points = loaded_perspective # Load list of points
                    print("[INFO] Loaded custom perspective points.")

                return config
        except Exception as e:
            print(f"[WARN] Failed to load config from {config_path}: {e}")
            return {}
    return {}

def save_config(video_path, config_data):
    """Saves configuration data to a JSON file."""
    config_path = get_config_path(video_path)
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    
    # NEW: Save the perspective points
    global perspective_points
    config_data['perspective_points'] = perspective_points
    
    try:
        with open(config_path, 'w') as f:
            json.dump(config_data, f, indent=4)
        print(f"[INFO] Saved configuration to {config_path}")
        return True
    except Exception as e:
        print(f"[ERROR] Failed to save config to {config_path}: {e}")
        return False


# --- LOGGING FUNCTIONS (Unchanged) ---

def open_log_files(video_path):
    """Opens and initializes the two CSV log files, including metadata headers."""
    global individual_log_file, individual_log_writer, summary_log_file, summary_log_writer, run_date
    
    video_name = os.path.basename(video_path).split('.')[0]
    
    # 1. Individual Vehicle Log
    individual_path = os.path.join(OUT_DIR, f"{video_name}_log.csv")
    os.makedirs(os.path.dirname(individual_path), exist_ok=True)
    
    individual_log_file = open(individual_path, 'w', newline='')
    individual_log_writer = csv.writer(individual_log_file)
    # Header: Timestamp,ID,Type,Confidence,VideoTime,Image
    individual_log_writer.writerow(['Timestamp', 'ID', 'Type', 'Confidence', 'VideoTime', 'Image'])
    print(f"[INFO] Individual log file created: {individual_path}")

    # 2. Summary Log
    summary_path = os.path.join(OUT_DIR, f"{video_name}_summary_log.csv")
    summary_log_file = open(summary_path, 'w', newline='')
    summary_log_writer = csv.writer(summary_log_file)
    
    # Write metadata rows to match example
    summary_log_writer.writerow([video_name]) # Row 1: Project ID / Video Name
    
    # Row 2: Run Date (convert YYYY-MM-DD to DD/MM/YYYY for the header)
    date_str = run_date if run_date else "DD/MM/YYYY" # Placeholder if not yet set
    if date_str and len(date_str) == 10 and date_str[4] == '-':
        try:
            date_obj = datetime.strptime(date_str, "%Y-%m-%d")
            date_str = date_obj.strftime("%d/%m/%Y")
        except ValueError:
            pass 
            
    summary_log_writer.writerow([date_str]) 
    
    # Row 3: Header - 'Masa' (Time Period) + SUMMARY_CLASSES
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

def write_individual_log(writer, current_dt, tracker_id, class_id, direction, confidence, video_time, image_name):
    """Writes a single vehicle crossing event to the individual log file."""
    if writer is None:
        print("[WARN] Individual log writer not initialized.")
        return
        
    class_name = CLASS_MAP.get(class_id, "UNKNOWN")
    # Format timestamp with milliseconds (up to 3 decimal places)
    timestamp_str = current_dt.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3] 
    
    # Data to write: Timestamp, ID, Type, Confidence, VideoTime, Image
    writer.writerow([timestamp_str, int(tracker_id), class_name, confidence, video_time, image_name])
    individual_log_file.flush() 

def log_cropped_image(frame, bbox, tracker_id, current_dt, filename, video_name_folder):
    """Crops the bounding box area from the frame and saves it to disk."""
    global frame_width, frame_height
    
    # Ensure OUT_DIR/crops exists
    crops_dir = os.path.join(OUT_DIR, "vehicle_captures",video_name_folder)
    os.makedirs(crops_dir, exist_ok=True)
    
    # Get coordinates and ensure they are integers
    x1, y1, x2, y2 = [int(i) for i in bbox]
    
    # Add a small buffer/padding (e.g., 10 pixels)
    padding = 10
    x1 = max(0, x1 - padding)
    y1 = max(0, y1 - padding)
    x2 = min(frame_width, x2 + padding)
    y2 = min(frame_height, y2 + padding)

    # Crop the image
    cropped_image = frame[y1:y2, x1:x2]
    
    if cropped_image.size > 0:
        # Filename is passed in, use it directly.
        file_path = os.path.join(crops_dir, filename)
        cv2.imwrite(file_path, cropped_image)
    else:
        pass 

# --- HELPER FUNCTIONS ---

def is_point_in_button(x, y, button_coords):
    x_min, y_min, x_max, y_max = button_coords
    return x_min <= x <= x_max and y_min <= y <= y_max

def update_line_zone(y, width):
    global line_zone
    # Line starts at (0, y) and ends at (width, y)
    line_start = sv.Point(x=0, y=y)
    line_end = sv.Point(x=width, y=y)
    # Re-initialize LineZone to reset internal state if the line moves
    line_zone = sv.LineZone(
        start=line_start, 
        end=line_end,
        triggering_anchors=(sv.Position.BOTTOM_CENTER,),
    ) 

def reset_all(): 
    global polygon_points, poly_editing, polygon_defined, start_processing 
    global line_edit_mode, zone_polygon, polygon_annotator, line_zone, trace_annotator 
    global logged_tracker_ids, total_class_counts, flow_direction_mode
    global global_in_offset, global_out_offset, coordinates
    global perspective_editing # Do NOT reset perspective_points to maintain calibration

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
    flow_direction_mode = "BOTH" # Reset flow mode
    coordinates.clear() # Reset historical speed data
    perspective_editing = False # Turn off calibration mode

    print("[INFO] All zones and processing state reset.")

def draw_text_simple(frame, text, position, color):
    """Simple wrapper for drawing text on the frame."""
    x, y = position
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
    
def draw_shortcuts(frame, height):
    start_y = height - 70 
    x_offset = 20
    spacing = 200 
    
    # Draw header
    cv2.putText(frame, "SHORTCUTS:", (x_offset, start_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
    x_offset += 110
    
    current_x = x_offset
    for i, (key, action) in enumerate(SHORTCUTS.items()):
        text = f"{key}: {action}"
        # Simple line wrap - wrap every 4 items
        if i % 4 == 0 and i != 0: 
            start_y += 20
            current_x = x_offset
            
        cv2.putText(frame, text, (current_x, start_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        current_x += spacing


def draw_ui(frame, width, height, current_line_zone, current_polygon_points):
    """Draws all UI elements (buttons, line, polygon points, shortcuts, and counts)."""
    global total_class_counts, flow_direction_mode, line_edit_mode, perspective_editing
    
    # 1. Draw Buttons (Centered)
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


 # 2. Draw Current Line Zone (if defined)
    if current_line_zone is not None:
        y_coord = current_line_zone.vector.start.y 

        line_color = (255, 255, 255)
        if line_edit_mode:
            line_color = (0, 255, 255) 
        
        cv2.line(
            frame,
            (0, y_coord),  # Start at X=0
            (width, y_coord), # End at X=width
            line_color, 2
        )

        # Draw the centered IN/OUT counter box manually 
        in_count = (current_line_zone.in_count + global_in_offset) if current_line_zone else 0
        out_count = (current_line_zone.out_count + global_out_offset) if current_line_zone else 0
        
        # Position the box (e.g., center of the screen)
        box_x = int(width / 2) - 35
        box_y = y_coord
        
        # Background box (White color, slightly above the line)
        box_width = 80
        box_height = 40
        cv2.rectangle(frame, (box_x, box_y - box_height), (box_x + box_width, box_y), (255, 255, 255), -1) 
        
        # In text 
        cv2.putText(frame, f"in: {in_count}", (box_x + 5, box_y - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)
        # Out text 
        cv2.putText(frame, f"out: {out_count}", (box_x + 5, box_y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)

    # 3. Draw Polygon Points (if editing)
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

    # 5. Draw Status
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
        
    draw_text_simple(frame, status, (20, height - 90), (255, 255, 255))
    
    # 6. Draw Shortcuts (bottom-left)
    draw_shortcuts(frame, height)
    
    # 7. Draw Class Counts (Top-Left)
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
        
    # Draw In/Out line counts (Top-Left - kept for display)
    if current_line_zone:
        total_in = current_line_zone.in_count + global_in_offset
        total_out = current_line_zone.out_count + global_out_offset
        
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

def elicit_start_time(video_path: str) -> bool:
    """Elicits the run date and start time from the user.
       Returns True if time is successfully set, False otherwise."""
    
    global run_date, run_start_time, start_dt, root
    
    root.deiconify() 
    
    time_data = elicit_time_gui()
    
    root.withdraw()

    if time_data['date'] and time_data['time']:
        try:
            run_date = datetime.strptime(time_data['date'], "%Y-%m-%d").strftime("%Y-%m-%d")
            run_start_time = datetime.strptime(time_data['time'], "%H:%M:%S").strftime("%H:%M:%S")

            start_dt = datetime.strptime(f"{run_date} {run_start_time}", "%Y-%m-%d %H:%M:%S")

            # Save the new config with time data
            config_data = load_config(video_path) # Need to reload to ensure all data is present before saving
            config_data['run_date'] = run_date
            config_data['run_start_time'] = run_start_time
            save_config(video_path, config_data)
            
            print(f"[INFO] Run time set: {run_date} {run_start_time}")
            return True
        
        except ValueError as e:
            messagebox.showerror("Time Format Error", f"Invalid date or time format. Please use YYYY-MM-DD and HH:MM:SS. Error: {e}")
            return False
    else:
        return False


def initial_start_and_reset(cap, video_path, frame_width, frame_height):
    """Rewinds the video, resets counters, and starts processing. Used for initial start or full reset."""
    global start_processing, line_zone, logged_tracker_ids, total_class_counts, run_date, run_start_time, start_dt
    
    # 1. Check/Elicit Time
    if run_date is None or run_start_time is None:
        print("[PROMPT] Configuration time missing. Please enter run time in GUI pop-up.")
        if not elicit_start_time(video_path):
            print("[WARN] Start aborted: Time not set.")
            return # Abort start if time elicitation failed
        
    # CRITICAL FIX: Calculate start_dt if run_date and run_start_time are now available
    if run_date and run_start_time:
        start_dt_str = f"{run_date} {run_start_time}"
        try:
            start_dt = datetime.strptime(start_dt_str, "%Y-%m-%d %H:%M:%S")
        except ValueError as e:
            print(f"[ERROR] Invalid saved time format: {e}")
            start_dt = None
            return # Abort start if time is invalid
    
    # 2. Rewind Video
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    # 3. Reset Counters and State
    if line_zone:
        line_y = line_zone.vector.start.y
        # frame_width is a global set in process_video
        update_line_zone(line_y, frame_width)
        
    logged_tracker_ids.clear() # Reset individual log tracker
    total_class_counts = defaultdict(int) # Reset total display counts
    
    # 4. Set Processing State
    start_processing = True
    print("[INFO] Video started and processing initiated.")


def mouse_callback(event, x, y, flags, param, video_path, cap, frame_height, frame_width):
    global poly_editing, polygon_points, polygon_defined, line_edit_mode
    global line_zone, start_processing, DISPLAY_SCALE, zone_polygon
    global polygon_annotator, run_date, run_start_time, force_exit, flow_direction_mode
    global perspective_editing, perspective_points, view_transformer # NEW Globals

    # Scale the click coordinates UP to match the original frame's resolution
    if DISPLAY_SCALE != 1.0:
        x = int(x / DISPLAY_SCALE)
        y = int(y / DISPLAY_SCALE)

    # Handle Perspective Editing (NEW)
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
                config_data = load_config(video_path) # Load existing data, this also updates perspective_points global
                # Re-save to ensure the new points are persisted
                save_config(video_path, config_data)
                print("[INFO] Perspective calibration finalized and saved. Order: TL, TR, BR, BL.")

        elif event == cv2.EVENT_RBUTTONDOWN and len(perspective_points) > 0:
            perspective_points.pop()
            print("[INFO] Removed last perspective point.")

    # Handle Polygon Editing (Existing)
    elif poly_editing:
        if event == cv2.EVENT_LBUTTONDOWN:
            polygon_points.append((x, y))
            print(f"[INFO] Added point: ({x}, {y})")
        elif event == cv2.EVENT_RBUTTONDOWN and len(polygon_points) > 0:
            polygon_points.pop()
            print("[INFO] Removed last point.")
    
    # Handle Line Editing (Existing)
    elif line_edit_mode:
        if event == cv2.EVENT_LBUTTONDOWN:
            new_line_y = y

            global global_in_offset, global_out_offset
            
            if line_zone:
                global_in_offset += line_zone.in_count
                global_out_offset += line_zone.out_count

            update_line_zone(new_line_y, frame_width)
            line_edit_mode = False 
            
            # Save the new line config
            config_data = load_config(video_path)
            config_data['polygon_points'] = polygon_points
            config_data['line_y'] = new_line_y
            config_data['run_date'] = run_date
            config_data['run_start_time'] = run_start_time
            save_config(video_path, config_data)

            print(f"[INFO] Line set at y={new_line_y}. Counts preserved via offset. Ready to start processing.")

    # Handle Button Clicks (Existing)
    elif event == cv2.EVENT_LBUTTONDOWN:
        if is_point_in_button(x, y, BUTTONS['poly']):
            poly_editing = not poly_editing
            line_edit_mode = False
            perspective_editing = False # Turn off other modes
            
            if poly_editing:
                polygon_points.clear()
                polygon_defined = False
                zone_polygon = None
                polygon_annotator = None
                print("[INFO] Polygon edit mode: click to add points (right-click to remove). Press 'Enter' to finish.")
            else:
                print("[INFO] Polygon edit mode OFF.")

        elif is_point_in_button(x, y, BUTTONS['start']):
            if polygon_defined and line_zone is not None and len(perspective_points) == 4:
                initial_start_and_reset(cap, video_path, frame_width, frame_height)
                
                line_edit_mode = False
                poly_editing = False
                perspective_editing = False
            else:
                print("[WARN] Define polygon, line zone, AND perspective (Press C) first.")

        elif is_point_in_button(x, y, BUTTONS['line']):
            line_edit_mode = True
            poly_editing = False
            perspective_editing = False # Turn off other modes
            print("[INFO] Line edit mode: click to set line.")
            
        elif is_point_in_button(x, y, BUTTONS['exit']):
            force_exit = True
            print("[INFO] Exit button pressed. Terminating.")


# --- MAIN PROCESSING FUNCTION ---

def process_video(video_path):
    global line_edit_mode, poly_editing, polygon_points, polygon_defined
    global force_exit, last_minute, minute_class_counts, flow_direction_mode
    global logged_tracker_ids, start_processing
    global run_date, run_start_time, start_dt
    global individual_log_file, individual_log_writer, summary_log_file, summary_log_writer
    global frame_width, frame_height
    global view_transformer, coordinates, trace_annotator
    global perspective_points, perspective_editing, TARGET, DEFAULT_SOURCE
    global TIME_SCALE_FACTOR # Added global for the new constant

    period_start_minute = None 

    # --- INITIALIZATION ---
    model = YOLO(MODEL_PATH)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return
    
    if logged_tracker_ids is None:
        logged_tracker_ids = set()

    # Initialize video dimensions
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) 
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    tracker_fps = min(fps, 30)

    video_name_folder = os.path.splitext(os.path.basename(video_path))[0]

    # Load persistent configuration (This also loads perspective_points)
    config = load_config(video_path)
    
    # Global state update from config
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

    # --- LOGGING SETUP ---
    open_log_files(video_path) 
    
    # Tracking and Annotation Setup
    byte_tracker = sv.ByteTrack(frame_rate=tracker_fps)
    
    thickness = sv.calculate_optimal_line_thickness(
        resolution_wh=(frame_width, frame_height)
    )
    trace_annotator = sv.TraceAnnotator(
        thickness=thickness,
        trace_length=int(fps * 2), # 2 seconds of trace
        position=sv.Position.BOTTOM_CENTER,
    )
    
    # --- ZONE DEFINITION (Default or loaded) ---
    
    # 1. Line Zone 
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
    
    # 2. Polygon Zone 
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

    # --- END ZONE DEFINITION ---
    
    # Time tracking variables
    last_minute = None
    minute_class_counts = defaultdict(int)
    period_start_minute = None
    
    # Setup OpenCV window and mouse callback
    cv2.namedWindow('Processed Video')
    def local_mouse_callback(event, x, y, flags, param):
        mouse_callback(event, x, y, flags, param, video_path, cap, frame_width, frame_height)
    cv2.setMouseCallback('Processed Video', local_mouse_callback)
    
    current_dt = datetime.now() 
    
    while True: 
        
        if start_processing:
            ret, frame = cap.read()
            if not ret or force_exit:
                break
        else:
            ret, frame = cap.read() 
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            
            if not ret or force_exit:
                break
                
        if period_start_minute is None and start_processing:
            period_start_minute = current_dt.strftime("%H:%M")
            
        # --- INITIALIZATION BLOCK ---
        ids_to_process = []
        newly_counted_in_ids = set()
        newly_counted_out_ids = set()

        in_ids = set()  

        frame_number = cap.get(cv2.CAP_PROP_POS_FRAMES)
        
        # --- TIME ALIGNMENT ---
        if start_dt and fps > 0:
            time_elapsed_seconds = (frame_number - 1) / fps 
            current_dt = start_dt + timedelta(seconds=time_elapsed_seconds)
            current_minute = current_dt.strftime("%H:%M")
        else:
            current_dt = datetime.now() 
            current_minute = current_dt.strftime("%H:%M") 

        # --- DRAW UI ---
        display = draw_ui(frame.copy(), frame_width, frame_height, line_zone, polygon_points)
        
        if start_processing: 
            
            # --- MODEL PREDICTION & TRACKING ---
            results = model(frame, device = 0, verbose=False)[0]
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

            # 5. APPLY POLYGON FILTERING
            if polygon_defined and zone_polygon is not None:
                is_in_polygon = zone_polygon.trigger(detections=detections)
                is_in_polygon = np.array(is_in_polygon, dtype=bool)
                detections = detections[is_in_polygon]

            # 6. Annotation (Bounding Boxes)
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

                        label = f"#{tracker_id}{class_name}{speed_label}"

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
            
            # 7. Update Line Counter and Log Events
            if all_tracked_detections.tracker_id is not None and len(all_tracked_detections) > 0:
                
                tracked_detections_for_line = all_tracked_detections
                tracked_detections_for_line.class_id = tracked_detections_for_line.class_id.astype(int)
                tracked_detections_for_line.tracker_id = tracked_detections_for_line.tracker_id.astype(int)
                
                in_mask, out_mask = line_zone.trigger(detections=tracked_detections_for_line)
                
                current_in_ids_full = set(tracked_detections_for_line.tracker_id[in_mask])
                current_out_ids_full = set(tracked_detections_for_line.tracker_id[out_mask])

                ids_in_polygon = set(detections.tracker_id) if detections.tracker_id is not None and len(detections) > 0 else set()
                
                current_in_ids = current_in_ids_full.intersection(ids_in_polygon)
                current_out_ids = current_out_ids_full.intersection(ids_in_polygon)

            else:
                current_in_ids = set()
                current_out_ids = set()

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
                    frame_number_log = cap.get(cv2.CAP_PROP_POS_FRAMES)
                    confidence = all_tracked_detections.confidence[idx]

                    time_elapsed_seconds = (frame_number_log - 1) / fps 
                    hours, remainder = divmod(time_elapsed_seconds, 3600)
                    minutes, seconds = divmod(remainder, 60)
                    video_time_str = f"{int(hours):02d}:{int(minutes):02d}:{seconds:06.3f}"

                    timestamp_for_crop = current_dt.strftime("%Y%m%d_%H%M%S%f")[:-3]
                    image_file_name = f"ID{int(tracker_id)}_{timestamp_for_crop}.jpg"

                    if current_dt:
                        write_individual_log(individual_log_writer, current_dt, tracker_id, class_id, direction, confidence, video_time_str, image_file_name)
                    
                    if current_dt:
                        log_cropped_image(frame, bbox, tracker_id, current_dt, image_file_name, video_name_folder)
                    
                    logged_tracker_ids.add(tracker_id)
                    total_class_counts[class_name] += 1

                    if class_name in SUMMARY_CLASSES:
                        minute_class_counts[class_name] += 1

            logged_tracker_ids = logged_tracker_ids.intersection(current_tracker_ids) 

            # --- SUMMARY LOGGING ROLLOVER ---
            if start_dt and current_minute != last_minute:
                if last_minute is not None and summary_log_writer:
                    summary_period = f"{period_start_minute} - {last_minute}"
                    summary_log_writer.writerow([summary_period] + [minute_class_counts[c] for c in SUMMARY_CLASSES])
                    summary_log_file.flush()

                    period_start_minute = current_minute 
                    
                    minute_class_counts = defaultdict(int)
                    
                last_minute = current_minute

        
        # --- ANNOTATE LINE AND POLYGON ---
        if polygon_defined and polygon_annotator is not None:
            display = polygon_annotator.annotate(scene=display)
        
        # --- ANNOTATE TRACES ---
        if start_processing and trace_annotator is not None:
            display = trace_annotator.annotate(scene=display, detections=detections)
            
        # --- DISPLAY AND KEY HANDLING ---
        
        if DISPLAY_SCALE != 1.0:
            display = cv2.resize(display, (int(frame_width * DISPLAY_SCALE), int(frame_height * DISPLAY_SCALE)))

        if fps > 0 and start_processing:
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
            perspective_editing = False
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
            if polygon_defined and line_zone is not None and len(perspective_points) == 4:
                initial_start_and_reset(cap, video_path, frame_width, frame_height) 
            else:
                print("[WARN] Define polygon, line zone, AND perspective (Press C) first.")

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
                config_data = load_config(video_path)
                config_data['polygon_points'] = polygon_points
                config_data['line_y'] = line_zone.vector.start.y
                config_data['run_date'] = run_date
                config_data['run_start_time'] = run_start_time
                save_config(video_path, config_data)
                
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
                
                # Save the new polygon config
                config_data = load_config(video_path)
                config_data['polygon_points'] = polygon_points
                if line_zone:
                    config_data['line_y'] = line_zone.vector.start.y
                config_data['run_date'] = run_date
                config_data['run_start_time'] = run_start_time
                save_config(video_path, config_data)

            elif poly_editing and len(polygon_points) < 3:
                print("[WARN] Need at least 3 points to define a polygon.")


    # Final summary log entry
    if summary_log_writer is not None and last_minute is not None:
        minute_end = current_dt.strftime("%H:%M") 
        summary_period = f"{period_start_minute} - {minute_end}"
        summary_log_writer.writerow([summary_period] + [minute_class_counts[c] for c in SUMMARY_CLASSES])
    
    cap.release()
    cv2.destroyAllWindows()
    close_log_files()
    print("[INFO] Processing finished. Logs saved.")


if __name__ == "__main__":
    root = tk.Tk()
    root.withdraw() 
    video_file_path = filedialog.askopenfilename(
        title="Select Video File to Process",
        filetypes=[("Video files", "*.mp4 *.avi *.mov")]
    )

    if video_file_path:
        print(f"[INFO] Selected video: {video_file_path}")
        process_video(video_file_path)
    else:
        print("[INFO] No video file selected. Exiting.")