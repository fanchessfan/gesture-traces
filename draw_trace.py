import cv2
import json
import re
import argparse
import sys
# import mediapipe as mp
import numpy as np

def map_landmark(name):
    if name == "nose":        return "face", 0,    None, False
    if name == "left_hand":   return "left_hand", 8, None, False
    if name == "right_hand":  return "right_hand",8, None, False
    if name == "left_wrist":  return "pose", 15,   None, False
    if name == "right_wrist": return "pose", 16,   None, False
    if name == "handed":      return "pose", 15,   16,   True
    print("Unknown landmark:", name); sys.exit(1)
    
def load_landmarks(path):
    """Load the landmarks JSON (a list of time‐stamped records)."""
    with open(path, 'r') as f:
        return json.load(f)


def extract_landmark_trace(landmarks_data, start_time, end_time, category, lid):
    """
    Extract normalized (x,y) coords for landmark index lid between start_time and end_time.
    """
    trace = []
    for rec in landmarks_data:
        t = rec.get("time", 0.0)
        if start_time <= t <= end_time and category in rec:
            lst = rec[category]
            if lid < len(lst):
                lm = lst[lid]
                if category != "pose" or lm.get("visibility", 0) > 0.5:
                    trace.append((lm["x"], lm["y"]))
    return trace


def draw_trace(img, trace, width, height):
    """Draw a polyline + green circles for the normalized trace on img."""
    pts = [(int(x*width), int(y*height)) for x,y in trace]
    if len(pts) >= 2:
        cv2.polylines(img, [np.array(pts, np.int32)], False, (255,0,0), 2)
    for p in pts:
        cv2.circle(img, p, 3, (0,255,0), -1)
    return img

def show_image(img):
    import matplotlib.pyplot as plt
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()

def main2():
    landmarks = load_landmarks("empty_landmarks.json")
    category, id_l, id_r, handed = map_landmark("left_wrist")

    trace = extract_landmark_trace(landmarks, 5.0, 6.0, category, id_l) # Test trace extraction
    img_trace = draw_trace(np.zeros((480, 640, 3), dtype=np.uint8), trace, 640, 480)

    show_image(img_trace)

if __name__ == "__main__":
    main2()
