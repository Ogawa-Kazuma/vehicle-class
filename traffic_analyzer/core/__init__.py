"""Core functionality for traffic analyzer."""

from traffic_analyzer.core.state import AppState
from traffic_analyzer.core.vehicle_classifier import VehicleClassifier
from traffic_analyzer.core.time_sync import TimeSynchronizer

__all__ = ['AppState', 'VehicleClassifier', 'TimeSynchronizer']

