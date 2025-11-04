"""
Centroid Tracker Implementation

Simple but effective tracker based on centroid distance matching.
Used across multiple scripts in the codebase.
"""

import math
import numpy as np
from collections import OrderedDict
from typing import List, Tuple, Dict


class CentroidTracker:
    """
    Tracks objects by matching centroids between frames.
    
    Uses Euclidean distance to associate detections across frames.
    Objects that disappear for too many frames are removed.
    """
    
    def __init__(self, max_distance: float = 50, max_disappeared: int = 10):
        """
        Initialize tracker.
        
        Args:
            max_distance: Maximum distance for centroid matching (pixels)
            max_disappeared: Frames before object is deregistered
        """
        self.nextObjectID = 0
        self.objects = OrderedDict()  # objectID -> centroid (x, y)
        self.disappeared = OrderedDict()  # objectID -> frames_disappeared
        self.max_distance = max_distance
        self.max_disappeared = max_disappeared
    
    def register(self, centroid: Tuple[int, int]) -> int:
        """
        Register a new object.
        
        Args:
            centroid: (x, y) centroid coordinates
            
        Returns:
            Assigned object ID
        """
        objectID = self.nextObjectID
        self.objects[objectID] = centroid
        self.disappeared[objectID] = 0
        self.nextObjectID += 1
        return objectID
    
    def deregister(self, objectID: int):
        """
        Remove an object from tracking.
        
        Args:
            objectID: Object ID to remove
        """
        if objectID in self.objects:
            del self.objects[objectID]
        if objectID in self.disappeared:
            del self.disappeared[objectID]
    
    def update(self, input_centroids: List[Tuple[int, int]]) -> Dict[int, Tuple[int, int]]:
        """
        Update tracker with new detections.
        
        Args:
            input_centroids: List of (x, y) centroids from current frame
            
        Returns:
            Dictionary mapping objectID -> centroid
        """
        # If no existing objects, register all new centroids
        if len(self.objects) == 0:
            for centroid in input_centroids:
                self.register(centroid)
            return self.objects
        
        # If no new centroids, mark all as disappeared
        if len(input_centroids) == 0:
            for objectID in list(self.disappeared.keys()):
                self.disappeared[objectID] += 1
                if self.disappeared[objectID] > self.max_disappeared:
                    self.deregister(objectID)
            return self.objects
        
        # Match existing objects to new centroids
        objectIDs = list(self.objects.keys())
        objectCentroids = list(self.objects.values())
        
        # Calculate distance matrix
        D = np.linalg.norm(
            np.array(objectCentroids)[:, np.newaxis] - 
            np.array(input_centroids)[np.newaxis, :],
            axis=2
        )
        
        # Find best matches (greedy algorithm)
        rows = D.min(axis=1).argsort()
        cols = D.argmin(axis=1)[rows]
        
        usedRows = set()
        usedCols = set()
        
        # Update matched objects
        for (row, col) in zip(rows, cols):
            if row in usedRows or col in usedCols:
                continue
            
            # Check if distance is acceptable
            if D[row, col] > self.max_distance:
                continue
            
            objectID = objectIDs[row]
            self.objects[objectID] = input_centroids[col]
            self.disappeared[objectID] = 0
            usedRows.add(row)
            usedCols.add(col)
        
        # Handle unmatched existing objects
        unusedRows = set(range(len(objectCentroids))) - usedRows
        for row in unusedRows:
            objectID = objectIDs[row]
            self.disappeared[objectID] += 1
            if self.disappeared[objectID] > self.max_disappeared:
                self.deregister(objectID)
        
        # Register new objects
        unusedCols = set(range(len(input_centroids))) - usedCols
        for col in unusedCols:
            self.register(input_centroids[col])
        
        return self.objects

