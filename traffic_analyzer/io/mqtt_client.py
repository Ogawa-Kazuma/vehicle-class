"""
MQTT Client Module

Handles MQTT publishing for traffic data.
"""

import json
import ssl
from typing import Dict, Optional, Any
from datetime import datetime
try:
    import paho.mqtt.client as mqtt
    MQTT_AVAILABLE = True
except ImportError:
    MQTT_AVAILABLE = False
    mqtt = None


class MQTTClient:
    """
    MQTT client for publishing traffic analysis data.
    
    Supports TLS-secured connections with authentication.
    """
    
    def __init__(self, broker: str, port: int = 8883,
                 username: Optional[str] = None,
                 password: Optional[str] = None,
                 client_id: str = "traffic_analyzer",
                 topic_prefix: str = "traffic/data"):
        """
        Initialize MQTT client.
        
        Args:
            broker: MQTT broker hostname
            port: MQTT broker port (default 8883 for TLS)
            username: MQTT username
            password: MQTT password
            client_id: Client ID
            topic_prefix: Topic prefix for messages
        """
        if not MQTT_AVAILABLE:
            raise ImportError("paho-mqtt required. Install with: pip install paho-mqtt")
        
        self.broker = broker
        self.port = port
        self.username = username
        self.password = password
        self.topic_prefix = topic_prefix
        
        self.client = mqtt.Client(client_id=client_id, protocol=mqtt.MQTTv311)
        self.connected = False
        
        # Setup TLS
        self.client.tls_set(tls_version=ssl.PROTOCOL_TLSv1_2, cert_reqs=ssl.CERT_REQUIRED)
        
        # Setup authentication
        if username and password:
            self.client.username_pw_set(username, password)
        
        # Callbacks
        self.client.on_connect = self._on_connect
        self.client.on_disconnect = self._on_disconnect
    
    def _on_connect(self, client, userdata, flags, rc):
        """Connection callback."""
        if rc == 0:
            self.connected = True
            print(f"[MQTT] Connected to {self.broker}:{self.port}")
        else:
            print(f"[MQTT] Connection failed with code {rc}")
            self.connected = False
    
    def _on_disconnect(self, client, userdata, rc):
        """Disconnection callback."""
        self.connected = False
        print(f"[MQTT] Disconnected")
    
    def connect(self, keepalive: int = 60):
        """Connect to broker."""
        try:
            self.client.connect(self.broker, self.port, keepalive=keepalive)
            self.client.loop_start()
        except Exception as e:
            print(f"[MQTT] Connection error: {e}")
    
    def disconnect(self):
        """Disconnect from broker."""
        self.client.loop_stop()
        self.client.disconnect()
    
    def publish_15min_report(self, timestamp: datetime, class_counts: Dict[str, int],
                           avg_speed: Optional[float] = None,
                           avg_headway: Optional[float] = None,
                           avg_gap: Optional[float] = None,
                           avg_occupancy: Optional[float] = None):
        """
        Publish 15-minute summary report.
        
        Args:
            timestamp: Report timestamp
            class_counts: Dictionary of class -> count
            avg_speed: Average speed (km/h)
            avg_headway: Average headway (seconds)
            avg_gap: Average gap (seconds)
            avg_occupancy: Average occupancy (%)
        """
        if not self.connected:
            print("[MQTT] Not connected, skipping publish")
            return
        
        payload = {
            "timestamp": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            "class_counts": class_counts,
            "total": sum(class_counts.values()),
            "metrics": {
                "avg_speed_kmh": avg_speed,
                "avg_headway_sec": avg_headway,
                "avg_gap_sec": avg_gap,
                "avg_occupancy_percent": avg_occupancy,
            }
        }
        
        topic = f"{self.topic_prefix}/15min"
        self.client.publish(topic, json.dumps(payload), qos=1)
        print(f"[MQTT] Published 15-min report to {topic}")
    
    def publish_event(self, event_type: str, data: Dict[str, Any]):
        """
        Publish custom event.
        
        Args:
            event_type: Event type (e.g., "vehicle_detected")
            data: Event data dictionary
        """
        if not self.connected:
            return
        
        payload = {
            "event_type": event_type,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "data": data
        }
        
        topic = f"{self.topic_prefix}/events/{event_type}"
        self.client.publish(topic, json.dumps(payload), qos=1)

