from typing import Dict, List, ParamSpecKwargs

import cv2
import numpy as np

from supervision.draw.color import Color
from supervision.geometry.dataclasses import Point, Rect, Vector
from supervision.tools.detections import Detections


def check_bottle_hit(detections):
    """
    Checks if there are any bottles with width larger than height.

    Parameters:
    detections (numpy.ndarray): An array of detections with bounding boxes in the format [x1, y1, x2, y2]

    Returns:
    bool: True if there is at least one bottle with width larger than height, False otherwise.
    """
    for detection in detections:
        x1, y1, x2, y2 = detection
        width = x2 - x1
        height = y2 - y1

        if width > height:
            return True

    return False

def draw_quadrilateral(image, vertices, color=(255, 0, 0), thickness=4):
    """
    Draws a quadrilateral on the given image.

    Parameters:
    image (numpy.ndarray): The image on which to draw the quadrilateral.
    vertices (list): List of four tuples representing the vertices of the quadrilateral.
    color (tuple): The color of the lines in BGR format (default is green).
    thickness (int): The thickness of the lines (default is 2).
    """
    # Convert vertices to NumPy array
    pts = np.array(vertices, np.int32)

    # Reshape the array into shape accepted by polylines
    # pts = pts.reshape((-1, 1, 2))

    # Draw the quadrilateral
    cv2.polylines(image, [pts], isClosed=True, color=color, thickness=thickness)

def draw_straight_line(image, pt1, pt2, color=(0, 255, 0), thickness=3):
  """
  Draws a straight line passing through two points on the given image and returns the start and end points of the line.

  Parameters:
  image (numpy.ndarray): The image on which to draw the line.
  pt1 (tuple): The coordinates of the first point (x1, y1).
  pt2 (tuple): The coordinates of the second point (x2, y2).
  color (tuple): The color of the line in BGR format (default is green).
  thickness (int): The thickness of the line (default is 2).

  Returns:
  tuple: The start and end points of the line.
  """
  # Get image dimensions
  height , width = image.shape[:2]

  # Calculate the slope of the line
  if pt2[0] - pt1[0] != 0:
      slope = (pt2[1] - pt1[1]) / (pt2[0] - pt1[0])
  else:
      slope = np.inf  # Infinite slope for vertical line

  # Calculate the intercept of the line
  if slope != np.inf:
      intercept = pt1[1] - slope * pt1[0]
  else:
      intercept = pt1[1]

  # Initialize points list
  points = []

  if slope != np.inf:
      # Intersect with left boundary (x=0)
      y0 = int(intercept)
      if 0 <= y0 < height:
          points.append((0, y0))

      # Intersect with right boundary (x=width-1)
      y1 = int(slope * (width - 1) + intercept)
      if 0 <= y1 < height:
          points.append((width - 1, y1))
  else:
      # If slope is infinite, line is vertical, intersect at top and bottom
      points.append((pt1[0], 0))
      points.append((pt1[0], height - 1))

  # Intersect with top boundary (y=0)
  if slope != 0 and slope != np.inf:
      x0 = int(-intercept / slope)
      if 0 <= x0 < width:
          points.append((x0, 0))

      # Intersect with bottom boundary (y=height-1)
      x1 = int((height - 1 - intercept) / slope)
      if 0 <= x1 < width:
          points.append((x1, height - 1))

  # Select the two points within the image boundaries
  if len(points) > 2:
      points = sorted(points, key=lambda p: (p[0], p[1]))[:2]

  # Draw the line
  if len(points) == 2:
      cv2.line(image, points[0], points[1], color, thickness)

  return points[0], points[1]

def check_ball_touch_feet(player_keypoints, ball_bbox, threshold=100):
    """
    Check if the ball touched the player's feet.

    Args:
        player_keypoints (list): List of player's keypoints. Each keypoint is a tuple (x, y).
        ball_bbox (np.ndarray): Bounding box of the ball in the format [x1, y1, x2, y2].
        threshold (int): Distance threshold to consider that the ball touched the player's feet.

    Returns:
        bool: True if the ball touched the player's feet, False otherwise.
    """
    ball_center = ((ball_bbox[0] + ball_bbox[2]) // 2, (ball_bbox[1] + ball_bbox[3]) // 2)
    
    # Assuming keypoints 15 and 16 are the left and right foot respectively (as per COCO dataset keypoints)
    foot_keypoints = [player_keypoints[15], player_keypoints[16]]
    
    for keypoint in foot_keypoints:
        if keypoint:  # Ensure the keypoint is not None
            distance = np.sqrt((ball_center[0] - keypoint[0]) ** 2 + (ball_center[1] - keypoint[1]) ** 2)
            if distance <= threshold:
                return True

    return False

