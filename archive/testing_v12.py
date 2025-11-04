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
from collections import defaultdict, OrderedDict
from ultralytics import YOLO
import supervision as sv 
from supervision.detection.tools.polygon_zone import PolygonZone 
from supervision.draw.utils import draw_text 
from supervision.detection.core import Detections
from supervision.detection.line_zone import LineZone
from tkinter import messagebox 
import paho.mqtt.client as mqtt # NEW: Import MQTT library
import ssl # NEW: Import SSL module for secure connection

# --- CONFIG ---
MODEL_PATH = "pretrained.pt"
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
    ("A", "Auto Line"),
    ("D", "Flow Mode")
])

# --- MQTT CONFIG (NEW) ---
# Extracted from user's configuration images
MQTT_BROKER = "broker.react.net.my"
MQTT_PORT = 8883 # Secure port
MQTT_TOPIC = "kkr/stl/ai/data" # Default topic for publishing summary counts
MQTT_USER = "test_ai_stl" # User's provided username
MQTT_PASSWORD = "test_ai_stl_2025" # User's provided password
MQTT_CLIENT_ID = "ppk"
PUBLISH_INTERVAL_MINUTES = 15 # Publish every 15 minutes
# --- END MQTT CONFIG ---

# --- GLOBAL STATE VARIABLES ---
polygon_points = []
poly_editing = False
polygon_defined = False
start_processing = False
line_edit_mode = False
count_line_y = 0

frame_width = 0
frame_height = 0

zone_polygon = None
polygon_annotator = None
line_zone = None 
trace_annotator = None 

# New globals for persistent time
run_date = None
run_start_time = None
force_exit = False 
root = None # Tkinter root instance (Defined at module level)
start_dt = None

global_in_offset = 0
global_out_offset = 0

last_in_count = 0
last_out_count = 0
# This map tracks the final crossing direction for a logged ID
tracker_direction_map = {} 

# Log file handles and counters
individual_log_file = None
individual_log_writer = None
summary_log_file = None
summary_log_writer = None

logged_tracker_ids = set() 
total_class_counts = defaultdict(int) 
flow_direction_mode = "BOTH"

# NEW: Global MQTT Client
mqtt_client = None 

# --- CONFIG PERSISTENCE ---

def get_config_path(video_path):
# ... (function content unchanged)
    """Generates the path for the config JSON file based on the video path."""
    video_name = os.path.basename(video_path)
    config_name = f"{os.path.splitext(video_name)[0]}_config.json"
    return os.path.join(OUT_DIR, config_name)

def load_config(video_path):
# ... (function content unchanged)
    """Loads configuration data from a JSON file."""
    config_path = get_config_path(video_path)
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
                print(f"[INFO] Loaded configuration from {config_path}")
                return config
        except Exception as e:
            print(f"[WARN] Failed to load config from {config_path}: {e}")
            return {}
    return {}

def save_config(video_path, config_data):
# ... (function content unchanged)
    """Saves configuration data to a JSON file."""
    config_path = get_config_path(video_path)
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    try:
        with open(config_path, 'w') as f:
            json.dump(config_data, f, indent=4)
        print(f"[INFO] Saved configuration to {config_path}")
        return True
    except Exception as e:
        print(f"[ERROR] Failed to save config to {config_path}: {e}")
        return False


# --- LOGGING FUNCTIONS ---

def open_log_files(video_path):
# ... (function content unchanged)
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
# ... (function content unchanged)
    """Closes all open log files."""
    global individual_log_file, summary_log_file
    if individual_log_file:
        individual_log_file.close()
        print("[INFO] Individual log file closed.")
    if summary_log_file:
        summary_log_file.close()
        print("[INFO] Summary log file closed.")

# UPDATED: write_individual_log to accept new fields
def write_individual_log(writer, current_dt, tracker_id, class_id, confidence, video_time, image_name):
# ... (function content unchanged)
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

# UPDATED: log_cropped_image to accept the pre-generated filename
def log_cropped_image(frame, bbox,filename, video_name_folder):
# ... (function content unchanged)
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

def prompt_for_start_time():
# ... (function content unchanged)
    """Prompts the user for the video run time using a dedicated Tkinter instance."""
    # This function is now deprecated in favor of elicit_time_gui in the main flow
    # but kept for potential legacy/fallback reasons.
    try:
        # Create a new, temporary Tkinter root
        temp_root = tk.Tk()
        temp_root.withdraw() # Hide the main window
        
        time_str = simpledialog.askstring(
            "Configuration Time Missing", 
            "Please enter the video's start time (HH:MM):",
            parent=temp_root
        )
        
        temp_root.destroy() # Immediately clean up the temporary root
        return time_str
    except Exception as e:
        print(f"[ERROR] Could not display time prompt: {e}")
        return None

# --- NEW MQTT FUNCTIONS (NEW) ---

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

    # Configure secure connection using TLSv1.2 and relying on system CA certificates
    try:
        mqtt_client.tls_set(
            tls_version=ssl.PROTOCOL_TLSv1_2, # Matches user's config
            cert_reqs=ssl.CERT_REQUIRED # Requires CA signed certificate
        ) 
    except Exception as e:
        print(f"[ERROR] TLS setup failed: {e}. Connection will likely fail.")
    
    # Set credentials
    mqtt_client.username_pw_set(MQTT_USER, MQTT_PASSWORD)
    
    # Attempt connection
    try:
        mqtt_client.connect(MQTT_BROKER, MQTT_PORT, keepalive=60)
        mqtt_client.loop_start() # Start the background thread for network traffic
        return mqtt_client
    except Exception as e:
        print(f"[ERROR] Failed to connect to MQTT broker: {e}")
        mqtt_client = None
        return None

def publish_summary_data(video_path, period_start, period_end, class_counts):
    """Formats the summary data and publishes it to the configured topic."""
    global mqtt_client
    
    if mqtt_client is None or not mqtt_client.is_connected():
        print("[WARN] MQTT client not connected. Attempting to re-connect...")
        connect_mqtt()
        if mqtt_client is None or not mqtt_client.is_connected():
            print("[ERROR] MQTT client still not connected. Skipping publish.")
            return

    # Create the data payload
    data = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "period": f"{period_start} - {period_end}",
        "class_counts": dict(class_counts),
        "source_video": os.path.basename(video_path)
    }
    
    payload = json.dumps(data)
    
    try:
        # Publish with QoS 0 (default in MQTT.fx)
        mqtt_client.publish(MQTT_TOPIC, payload, qos=0)
        print(f"[INFO] Published MQTT data for {data['period']}")
    except Exception as e:
        print(f"[ERROR] Failed to publish MQTT message: {e}")

# --- HELPER FUNCTIONS ---

def is_point_in_button(x, y, button_coords):
# ... (function content unchanged)
    x_min, y_min, x_max, y_max = button_coords
    return x_min <= x <= x_max and y_min <= y <= y_max

def update_line_zone(y, width):
# ... (function content unchanged)
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
# ... (function content unchanged)
    global polygon_points, poly_editing, polygon_defined, start_processing 
    global line_edit_mode, zone_polygon, polygon_annotator, line_zone, trace_annotator 
    global logged_tracker_ids, total_class_counts, flow_direction_mode
    global global_in_offset, global_out_offset  

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
    

    print("[INFO] All zones and processing state reset.")

def draw_text_simple(frame, text, position, color):
# ... (function content unchanged)
    """Simple wrapper for drawing text on the frame."""
    x, y = position
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
    
def draw_shortcuts(frame, height):
# ... (function content unchanged)
    start_y = height - 70 # Give more space at the bottom
    x_offset = 20
    spacing = 200 # Increased spacing to prevent cluster
    
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
# ... (function content unchanged)
    """Draws all UI elements (buttons, line, polygon points, shortcuts, and counts)."""
    global total_class_counts, flow_direction_mode, line_edit_mode
    
    # 1. Draw Buttons (Centered)
    for name, coords in BUTTONS.items():
        x_min, y_min, x_max, y_max = coords
        
        color_rgb = BUTTON_COLOR_INACTIVE
        if (name == 'poly' and poly_editing) or            (name == 'line' and line_edit_mode) or            (name == 'start' and start_processing): # 'start' highlights when processing
            color_rgb = BUTTON_COLOR_ACTIVE
        
        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color_rgb, -1)
        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (255, 255, 255), 2)
        
        text_x = x_min + 5
        text_y = y_min + 35
        cv2.putText(frame, name.capitalize(), (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)


 # 2. Draw Current Line Zone (if defined)
    if current_line_zone is not None:
        # Get the Y coordinate from the line_zone vector
        y_coord = current_line_zone.vector.start.y 

        # Draw the full-width white line
        line_color = (255, 255, 255)
        # Use yellow if in line edit mode, as seen in your image
        if line_edit_mode:
            line_color = (0, 255, 255) # Cyan/Yellowish for active edit
        
        cv2.line(
            frame,
            (0, y_coord),  # Start at X=0
            (width, y_coord), # End at X=width
            line_color, 2
        )

        # Draw the centered IN/OUT counter box manually 
        in_count = (current_line_zone.in_count + global_in_offset) if current_line_zone else 0
        out_count = (current_line_zone.out_count) if current_line_zone else 0
        out_count = (current_line_zone.out_count + global_out_offset) if current_line_zone else 0
        
        # Position the box (e.g., center of the screen)
        box_x = int(width / 2) - 35
        box_y = y_coord
        
        # Background box (Yellow color, slightly above the line)
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
            
    # 4. Draw Status
    status = ""
    if poly_editing:
        status = "STATUS: Polygon Edit Mode (Click to Add, ENTER to Finish) - Processing Still Running"
    elif line_edit_mode:
        status = "STATUS: Line Edit Mode (Click to Set Y) - Processing Still Running"
    elif start_processing:
        status = "STATUS: Processing (Press S to Restart)"
    else:
        status = "STATUS: Initial State (Define Zones and Press S)"
        
    draw_text_simple(frame, status, (20, height - 90), (255, 255, 255))
    
    # 5. Draw Shortcuts (bottom-left)
    draw_shortcuts(frame, height)
    
    # 6. Draw Class Counts (Top-Left - kept for display)
    
    # Draw Flow Direction Mode (New)
    flow_color = (0, 255, 0) if flow_direction_mode == "IN" else (175, 196, 104) if flow_direction_mode == "OUT" else (255, 255, 0)
    draw_text_simple(frame, f"FLOW MODE: {flow_direction_mode} (Press D to Toggle)", (20, 100), flow_color)
    
    y_offset = 130 # Moved down for better spacing.
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
        
        # Display IN count
        draw_text_simple(frame, f"IN: {total_in}", (30, y_offset), (97, 212, 101))
        
        y_offset += 30 # Move down for the OUT count
        
        # Display OUT count
        draw_text_simple(frame, f"OUT: {total_out}", (30, y_offset), (70, 173, 242))
    
    return frame

# --- NEW GUI ELICITATION ---

def elicit_time_gui():
# ... (function content unchanged)
    """Opens a modal tkinter window to elicit run date and start time."""
    
    # Store results here
    results = {'date': None, 'time': None}
    
    # Create the main window (dialog box)
    dialog = tk.Toplevel(root)
    dialog.title("Enter Run Time")
    dialog.attributes('-topmost', True) # Keep it on top
    
    def validate_and_submit():
        date_str = date_entry.get()
        time_str = time_entry.get()
        
        try:
            # Validate formats
            datetime.strptime(date_str, "%Y-%m-%d")
            datetime.strptime(time_str, "%H:%M:%S")
            
            # Save results and close
            results['date'] = date_str
            results['time'] = time_str
            dialog.destroy()
            
        except ValueError:
            messagebox.showerror("Invalid Format", "Date must be YYYY-MM-DD and Time must be HH:MM:SS (24-hour).", parent=dialog)
            
    # Labels and Entries
    tk.Label(dialog, text="Run Date (YYYY-MM-DD):").grid(row=0, column=0, padx=5, pady=5, sticky="w")
    date_entry = tk.Entry(dialog)
    date_entry.grid(row=0, column=1, padx=5, pady=5)
    
    tk.Label(dialog, text="Start Time (HH:MM:SS):").grid(row=1, column=0, padx=5, pady=5, sticky="w")
    time_entry = tk.Entry(dialog)
    time_entry.grid(row=1, column=1, padx=5, pady=5)
    
    # Submit Button
    submit_button = tk.Button(dialog, text="Set Time & Continue", command=validate_and_submit)
    submit_button.grid(row=2, column=0, columnspan=2, pady=10)

    root.deiconify()
    
    # Modal behavior
    dialog.transient(root) 
    dialog.grab_set() 
    root.wait_window(dialog) 
    root.withdraw()

    return results

def elicit_start_time(video_path: str) -> bool:
# ... (function content unchanged)
    """Elicits the run date and start time from the user.
       Returns True if time is successfully set, False otherwise."""
    
    global run_date, run_start_time, global_offset_dt, root
    
    # CRITICAL FIX: Explicitly de-withdraw the root before calling the dialog
    root.deiconify() 
    
    time_data = elicit_time_gui()
    
    # CRITICAL FIX: Re-hide the root after the dialog closes
    root.withdraw()

    if time_data['date'] and time_data['time']:
        try:
            # Validate and parse the date/time strings
            run_date = datetime.strptime(time_data['date'], "%Y-%m-%d").strftime("%Y-%m-%d")
            run_start_time = datetime.strptime(time_data['time'], "%H:%M:%S").strftime("%H:%M:%S")

            # Reset the global offset datetime object
            start_dt = datetime.strptime(f"{run_date} {run_start_time}", "%Y-%m-%d %H:%M:%S")
            global_offset_dt = start_dt

            # Save the new config with time data
            config_data = load_config(video_path)
            config_data['run_date'] = run_date
            config_data['run_start_time'] = run_start_time
            save_config(video_path, config_data)
            
            print(f"[INFO] Run time set: {run_date} {run_start_time}")
            return True
        
        except ValueError as e:
            messagebox.showerror("Time Format Error", f"Invalid date or time format. Please use YYYY-MM-DD and HH:MM:SS. Error: {e}")
            return False
    else:
        # User canceled the dialog
        return False


# RENAMED and MODIFIED: was restart_processing
def initial_start_and_reset(cap, video_path, frame_width, frame_height):
# ... (function content unchanged)
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
        
    # minute_class_counts is not global here, but it's defined and reset inside the main loop's rollover logic
    # We only clear the display and individual logs
    logged_tracker_ids.clear() # Reset individual log tracker
    total_class_counts = defaultdict(int) # Reset total display counts
    
    # 4. Set Processing State
    start_processing = True
    print("[INFO] Video started and processing initiated.")


def mouse_callback(event, x, y, flags, param, video_path, cap, frame_height, frame_width):
# ... (function content unchanged)
    global poly_editing, polygon_points, polygon_defined, line_edit_mode
    global line_zone, start_processing, DISPLAY_SCALE, zone_polygon
    global polygon_annotator, run_date, run_start_time, force_exit, flow_direction_mode

    # Scale the click coordinates UP to match the original frame's resolution
    if DISPLAY_SCALE != 1.0:
        x = int(x / DISPLAY_SCALE)
        y = int(y / DISPLAY_SCALE)

    # Handle Polygon Editing
    if poly_editing:
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

            # --- FIX: Preserve line counts by using global offset ---
            global global_in_offset, global_out_offset
            
            # 1. Add current line counts to the global offset before resetting the line_zone
            if line_zone:
                global_in_offset += line_zone.in_count
                global_out_offset += line_zone.out_count

            # 2. Re-initialize LineZone (this call resets internal line_zone.in_count/out_count to 0)
            update_line_zone(new_line_y, frame_width)
            line_edit_mode = False 
            
            # Save the new line config
            config_data = {
                'polygon_points': polygon_points,
                'line_y': new_line_y,
                'run_date': run_date,
                'run_start_time': run_start_time
            }
            save_config(video_path, config_data)

            print(f"[INFO] Line set at y={new_line_y}. Counts preserved via offset. Ready to start processing.")

    # Handle Button Clicks
    elif event == cv2.EVENT_LBUTTONDOWN:
        if is_point_in_button(x, y, BUTTONS['poly']):
            poly_editing = not poly_editing
            line_edit_mode = False
            # Processing continues to run - we only toggle the UI editing mode

            if poly_editing:
                polygon_points.clear()
                polygon_defined = False
                zone_polygon = None
                polygon_annotator = None
                print("[INFO] Polygon edit mode: click to add points (right-click to remove). Press 'Enter' to finish.")
            else:
                print("[INFO] Polygon edit mode OFF.")

        elif is_point_in_button(x, y, BUTTONS['start']):
            # Check if necessary zones are defined
            if polygon_defined and line_zone is not None:
                # User request: Pressing 'Start' ALWAYS restarts the video from the beginning
                initial_start_and_reset(cap, video_path, frame_width, frame_height)
                
                line_edit_mode = False
                poly_editing = False
            else:
                print("[WARN] Define polygon and line zone first.")

        elif is_point_in_button(x, y, BUTTONS['line']):
            line_edit_mode = True
            poly_editing = False
            # Processing continues to run - we only toggle the UI editing mode
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
    global frame_width, frame_height, mqtt_client

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

    # Load persistent configuration
    config = load_config(video_path)
    
    # Global state update from config
    run_date = config.get('run_date')
    run_start_time = config.get('run_start_time')
    
    # --- TIME CHECK AND PROMPT SECTION ---
    # This block is for initial config load when start_processing is pre-set
    if run_start_time is None and start_processing:
        # We now rely on initial_start_and_reset to handle the initial time prompt via GUI
        # This block is kept but simplified to avoid double prompting if user just hit 'S'
        if run_date is None or run_start_time is None:
            # For simplicity during initial load, we will just force a pause if time is unset.
             start_processing = False 
             print("[WARN] Start processing halted. Press 'S' to initiate time setup.")
    
    # --- LOGGING SETUP ---
    open_log_files(video_path) 
    
    # --- MQTT SETUP (NEW) ---
    connect_mqtt()

    # Tracking and Annotation Setup
    byte_tracker = sv.ByteTrack(frame_rate=tracker_fps)

    class_colors_list = [
        sv.Color(*CLASS_COLOR[CLASS_MAP[class_id]])
        for class_id in sorted(CLASS_MAP.keys())
    ]
    color_palette = sv.ColorPalette(colors=class_colors_list)
    
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
    
    frame_count = 0
    current_dt = datetime.now() 
    
    while True: 
        
        # --- Check if processing is ON to read next frame (Continuous Detection) ---
        if start_processing:
            ret, frame = cap.read()
            if not ret or force_exit:
                break
        else:
            
            # If not processing, just display the first frame repeatedly
            ret, frame = cap.read() 
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            
            if not ret or force_exit:
                break
                
        if period_start_minute is None and start_processing:
            # Initialize the start minute for the first period
            current_dt = datetime.now() # Ensure current_dt is updated even if start_dt is not set
            period_start_minute = current_dt.strftime("%H:%M")
            
        # --- INITIALIZATION BLOCK ---
        ids_to_process = []
        newly_counted_in_ids = set()
        newly_counted_out_ids = set()

        in_ids = set()  
        out_ids = set()

        frame_number = cap.get(cv2.CAP_PROP_POS_FRAMES)
        
        # --- TIME ALIGNMENT ---
        if start_dt and fps > 0:
            # CRITICAL FIX: Use (frame_number - 1) for accurate time delta calculation starting at 0
            time_elapsed_seconds = (frame_number - 1) / fps 
            current_dt = start_dt + timedelta(seconds=time_elapsed_seconds)
            current_minute = current_dt.strftime("%H:%M")
        else:
            current_dt = datetime.now() 
            current_minute = current_dt.strftime("%H:%M") 

        # --- DRAW UI ---
        display = draw_ui(frame.copy(), frame_width, frame_height, line_zone, polygon_points)
        
        if start_processing: # Only run heavy processing when explicitly started
            
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
            
            # 5. APPLY POLYGON FILTERING
            if polygon_defined and zone_polygon is not None:
                is_in_polygon = zone_polygon.trigger(detections=detections)
                is_in_polygon = np.array(is_in_polygon, dtype=bool)
                detections = detections[is_in_polygon]

            # 6. Annotation (Bounding Boxes)
            if len(detections) > 0:

                detections.class_id = detections.class_id.astype(int)
    
                # Ensure Class IDs and Tracker IDs are clean integers for drawing and labeling
                if detections.tracker_id is not None:
                    detections.tracker_id = detections.tracker_id.astype(int)
    
                    for xyxy, confidence, class_id, tracker_id in zip(
                        detections.xyxy, detections.confidence, detections.class_id, detections.tracker_id
                    ):
                        x1, y1, x2, y2 = [int(i) for i in xyxy]
            
                        class_name = CLASS_MAP.get(class_id, 'UNKNOWN')
                        color = CLASS_COLOR.get(class_name, (255, 255, 255))
                        label = f"#{tracker_id} {class_name} {confidence:.2f}"

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
                
                # Ensure Class IDs and Tracker IDs are clean integers for line counting
                
                tracked_detections_for_line = all_tracked_detections # all_tracked_detections is already a deep copy, use it directly.
                tracked_detections_for_line.class_id = tracked_detections_for_line.class_id.astype(int)
                tracked_detections_for_line.tracker_id = tracked_detections_for_line.tracker_id.astype(int)
                # line_zone.trigger updates internal counts and gives us the masks
                in_mask, out_mask = line_zone.trigger(detections=tracked_detections_for_line)
                
                # Get IDs that have passed the line in *this* frame (using the masks)
                current_in_ids_full = set(tracked_detections_for_line.tracker_id[in_mask])
                current_out_ids_full = set(tracked_detections_for_line.tracker_id[out_mask])

                # Get the IDs that are currently visible AND inside the polygon
                ids_in_polygon = set(detections.tracker_id) if detections.tracker_id is not None and len(detections) > 0 else set()
                
                # Filter the counted IDs to ONLY include those that are also in the polygon
                current_in_ids = current_in_ids_full.intersection(ids_in_polygon)
                current_out_ids = current_out_ids_full.intersection(ids_in_polygon)
            else:
                # If detections are empty or tracker_id is None, skip counting.
                current_in_ids = set()
                current_out_ids = set()

            # Find IDs that have crossed since the last check AND have not been logged
            newly_counted_in_ids = current_in_ids - logged_tracker_ids
            newly_counted_out_ids = current_out_ids - logged_tracker_ids

            # Compile list of IDs to log based on flow direction
            ids_to_process = []
            
            if flow_direction_mode == "BOTH" or flow_direction_mode == "IN":
                ids_to_process.extend(list(newly_counted_in_ids)) 
            
            if flow_direction_mode == "BOTH" or flow_direction_mode == "OUT":
                ids_to_process.extend(list(newly_counted_out_ids))
            
            # Define set of newly counted 'IN' IDs for direction logic in the loop
            in_ids = newly_counted_in_ids 

            # Get the IDs currently visible on screen
            current_tracker_ids = set(detections.tracker_id) if detections.tracker_id is not None else set()

            for tracker_id in ids_to_process:
                # Find the detection corresponding to this tracker_id in the *UNFILTERED* set
                idx = np.where(all_tracked_detections.tracker_id == tracker_id)[0]

                if len(idx) > 0:
                    idx = idx[0]
                    bbox = all_tracked_detections.xyxy[idx] # Use unfiltered set data
                    class_id = all_tracked_detections.class_id[idx]
                    class_name = CLASS_MAP.get(int(class_id), "UNKNOWN")
                    
                    direction = "IN" if tracker_id in in_ids else "OUT"
                    # Get frame-specific data for logging
                    frame_number = cap.get(cv2.CAP_PROP_POS_FRAMES)
                    #confidence = detections.confidence[idx]
                    confidence = all_tracked_detections.confidence[idx]

                    # Calculate video time (HH:MM:SS.ms)
                    time_elapsed_seconds = (frame_number - 1) / fps 
                    hours, remainder = divmod(time_elapsed_seconds, 3600)
                    minutes, seconds = divmod(remainder, 60)
                    video_time_str = f"{int(hours):02d}:{int(minutes):02d}:{seconds:06.3f}"

                    # Build image name (used for both log entry and saving the file)
                    # Name format: ID<TrackerID>_YYYYMMDD_HHMMSSms.jpg
                    timestamp_for_crop = current_dt.strftime("%Y%m%d_%H%M%S%f")[:-3]
                    image_file_name = f"ID{int(tracker_id)}_{timestamp_for_crop}.jpg"

                    # 1. Log to individual CSV (with all required fields)
                    if current_dt:
                        write_individual_log(individual_log_writer, current_dt, tracker_id, class_id, direction, confidence, video_time_str, image_file_name)
                    
                    # 2. Log cropped image (now uses the calculated image_file_name to save the file)
                    if current_dt:
                        log_cropped_image(frame, bbox, tracker_id, current_dt, image_file_name, video_name_folder)
                    
                    # 3. Mark as logged AND update total class count

                    logged_tracker_ids.add(tracker_id)
                    total_class_counts[class_name] += 1

                    # CRITICAL FIX: Update minute counts ONLY for logged vehicles
                    if class_name in SUMMARY_CLASSES:
                        minute_class_counts[class_name] += 1

            logged_tracker_ids = logged_tracker_ids.intersection(current_tracker_ids) 

            # --- SUMMARY LOGGING ROLLOVER ---
            if start_dt and current_minute != last_minute:
                if last_minute is not None and summary_log_writer:
                    # Log the completed minute. The period is [period_start_minute, last_minute]
                    summary_period = f"{period_start_minute} - {last_minute}"
                    summary_log_writer.writerow([summary_period] + [minute_class_counts[c] for c in SUMMARY_CLASSES])
                    summary_log_file.flush()
                    
                    # NEW: Publish MQTT Data
                    publish_summary_data(video_path, period_start_minute, last_minute, minute_class_counts)

                    period_start_minute = current_minute # Start the new logging period
                    
                    # Reset counters for the new minute
                    minute_class_counts = defaultdict(int)
                    
                # Update the tracking variables
                last_minute = current_minute

        
        # --- ANNOTATE LINE AND POLYGON ---
        if polygon_defined and polygon_annotator is not None:
            display = polygon_annotator.annotate(scene=display)
        
        # --- DISPLAY AND KEY HANDLING ---
        
        if DISPLAY_SCALE != 1.0:
            display = cv2.resize(display, (int(frame_width * DISPLAY_SCALE), int(frame_height * DISPLAY_SCALE)))

        if fps > 0 and start_processing: # Use FPS when processing
            wait_time_ms = max(1, int(1000 / fps)) 
        else: # Wait longer if not processing to allow UI interaction
            wait_time_ms = 100 

        cv2.imshow('Processed Video', display)

        k = cv2.waitKey(wait_time_ms) & 0xFF

        if k == 27: # ESC key
            force_exit = True
            break
            
        elif k == ord('l') or k == ord('L'): # L key (Edit Line)
            line_edit_mode = not line_edit_mode
            poly_editing = False
            # Processing continues to run
            print(f"[INFO] Line edit mode {'ON' if line_edit_mode else 'OFF'}.")
        
        elif k == ord('s') or k == ord('S'): # S key (Start/Restart)
            if polygon_defined and line_zone is not None:
                # User request: Always full restart
                initial_start_and_reset(cap, video_path, frame_width, frame_height) 
            else:
                print("[WARN] Define polygon and line zone first.")

        elif k == ord('r') or k == ord('R'): # R key (Reset All)
            reset_all()

        elif k == ord('d') or k == ord('D'): # D key (Toggle Flow Direction) (NEW FEATURE)
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
                update_line_zone(new_line_y, frame_width)
                
                # Save the new line config immediately
                config_data = {
                    'polygon_points': polygon_points,
                    'line_y': new_line_y,
                    'run_date': run_date,
                    'run_start_time': run_start_time
                }
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
                config_data = {
                    'polygon_points': polygon_points,
                    'line_y': line_zone.vector.start.y,
                    'run_date': run_date,
                    'run_start_time': run_start_time
                }
                save_config(video_path, config_data)

            elif poly_editing and len(polygon_points) < 3:
                print("[WARN] Need at least 3 points to define a polygon.")


    # Final summary log entry
    if summary_log_writer is not None and last_minute is not None:
        # Log the final period [period_start_minute, current_minute]
        minute_end = current_dt.strftime("%H:%M") 
        summary_period = f"{period_start_minute} - {minute_end}"
        summary_log_writer.writerow([summary_period] + [minute_class_counts[c] for c in SUMMARY_CLASSES])
        
        # NEW: Publish final MQTT Data
        publish_summary_data(video_path, period_start_minute, minute_end, minute_class_counts)
    
    cap.release()
    cv2.destroyAllWindows()
    close_log_files()
    
    # NEW: Disconnect MQTT client
    if mqtt_client:
        print("[INFO] Disconnecting MQTT client.")
        mqtt_client.loop_stop()
        mqtt_client.disconnect()

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