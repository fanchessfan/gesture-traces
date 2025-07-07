import cv2
import json
import re
import argparse
import sys
import mediapipe as mp
import numpy as np
# can reduce packages used

def serialize_landmarks(landmarks, landmark_type=None):
    """
    Convert MediaPipe landmarks to a JSON-serializable list.
    For 'face' type, only the nose landmark (index 1) is returned.
    """
    if landmark_type == "face":
        # Use index 1 for the nose landmark.
        lm = landmarks.landmark[1]
        lm_data = {'x': lm.x, 'y': lm.y, 'z': lm.z}
        if hasattr(lm, 'visibility'):
            lm_data['visibility'] = lm.visibility
        return [lm_data]
    else:
        serialized = []
        for lm in landmarks.landmark:
            lm_data = {'x': lm.x, 'y': lm.y, 'z': lm.z}
            if hasattr(lm, 'visibility'):
                lm_data['visibility'] = lm.visibility
            serialized.append(lm_data)
        return serialized



def main():
    # Parse command line arguments
    """
    4/24/2025: Adding option, --no-video, to output markers and skeletons on a black screen.

    """
    parser = argparse.ArgumentParser(
        description="Detection gesture skeletons, and overlay markers and gesture skeletons on a video, and output gesture landmarks in a JSON file."
    )
    parser.add_argument("--json", type=str, default="markers.json", 
                        help="Path to the JSON file with markers (default: markers.json)")
    parser.add_argument("--input", type=str, default="video.mp4", 
                        help="Path to the input MP4 video file (default: video.mp4)")
    parser.add_argument("--output", type=str, default="video_marker.mp4", 
                        help="Path to the output annotated MP4 video file (default: video_marker.mp4)")
    parser.add_argument("--landmarks-output", type=str, default="landmarks.json", 
                        help="Path to the output JSON file for gesture landmarks (default: landmarks.json)")
    parser.add_argument("--no-display", action="store_true",
                        help="Process video without displaying the video window (faster processing)")
    parser.add_argument("--no-video", action="store_true",
                        help="Output only skeleton and markers on a black frame")
    args = parser.parse_args()

    # Load the markers JSON file
    try:
        with open(args.json, 'r') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error loading JSON file: {e}")
        sys.exit(1)

    markers = []
    # Extract all markers from the JSON structure.
    for collection in data.get("contains", []):
        annotation_page = collection.get("first", {})
        for annotation in annotation_page.get("items", []):
            marker_text = annotation.get("body", {}).get("value", "")
            video_target = annotation.get("target", {}).get("id", "")
            # Extract start and end times using regex
            match = re.search(r'#t=([\d\.]+),([\d\.]+)', video_target)
            if match:
                start_time = float(match.group(1))
                end_time = float(match.group(2))
                markers.append({
                    "marker_text": marker_text,
                    "start_time": start_time,
                    "end_time": end_time
                })

    if not markers:
        print("No marker segments found in the JSON file.")
        sys.exit(1)
    else:
        print(f"Loaded {len(markers)} markers.")

    # Open the input video file using OpenCV
    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        print(f"Error opening video file: {args.input}")
        sys.exit(1)
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Video FPS: {fps}, Resolution: {width}x{height}")

    # Set up the VideoWriter to output the annotated video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(args.output, fourcc, fps, (width, height))
    
    # Initialize MediaPipe Holistic and Drawing utilities.
    mp_holistic = mp.solutions.holistic
    mp_drawing = mp.solutions.drawing_utils

    # Prepare a list to store landmarks data for each frame.
    landmarks_data = []

    frame_count = 0
    with mp_holistic.Holistic(min_detection_confidence=0.5,
                              min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("No more frames to read. Exiting loop.")
                break

            frame_count += 1
            # Calculate current video time in seconds.
            current_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            # Debug print for frame processing
            # print(f"Processing frame {frame_count} at time {current_time:.2f} sec", end='\r')

            active_markers = [m for m in markers if m["start_time"] <= current_time <= m["end_time"]]
            show_flag = (active_markers != [])

            # Process frame with MediaPipe (convert BGR to RGB)
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(image_rgb)

            if args.no_video:
                frame = np.zeros_like(frame)
            if show_flag:
                # Draw pose landmarks.
                if results.pose_landmarks:
                    mp_drawing.draw_landmarks(frame, results.pose_landmarks, 
                                            mp_holistic.POSE_CONNECTIONS)
                # Draw left hand landmarks.
                if results.left_hand_landmarks:
                    mp_drawing.draw_landmarks(frame, results.left_hand_landmarks, 
                                            mp_holistic.HAND_CONNECTIONS)
                # Draw right hand landmarks.
                if results.right_hand_landmarks:
                    mp_drawing.draw_landmarks(frame, results.right_hand_landmarks, 
                                            mp_holistic.HAND_CONNECTIONS)
                # For face landmarks, only draw the nose.
                if results.face_landmarks:
                    try:
                        # Use landmark index 1 as the nose.
                        nose = results.face_landmarks.landmark[1]
                        nose_x = int(nose.x * width)
                        nose_y = int(nose.y * height)
                        cv2.circle(frame, (nose_x, nose_y), 5, (0, 0, 255), -1)
                    except Exception as e:
                        print(f"Error drawing nose landmark: {e}")

            # Overlay markers (from JSON) on the frame if within active time.
            # active_markers = [m for m in markers if m["start_time"] <= current_time <= m["end_time"]]
            # print("frame, active_markers: ", ret, frame_count, active_markers == [])

            for i, m in enumerate(active_markers):
                cv2.putText(frame, m["marker_text"], (50, 50 + i * 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Write the annotated frame to the output video.
            out.write(frame)

            # Collect landmark data for the current frame.
            frame_landmarks = {"time": current_time}
            if results.pose_landmarks:
                frame_landmarks["pose"] = serialize_landmarks(results.pose_landmarks)
            if results.left_hand_landmarks:
                frame_landmarks["left_hand"] = serialize_landmarks(results.left_hand_landmarks)
            if results.right_hand_landmarks:
                frame_landmarks["right_hand"] = serialize_landmarks(results.right_hand_landmarks)
            if results.face_landmarks:
                frame_landmarks["face"] = serialize_landmarks(results.face_landmarks, landmark_type="face")
            landmarks_data.append(frame_landmarks)
            
            # Display the frame if not in no-display mode.
            if not args.no_display:
                cv2.imshow("Video", frame)
                # WaitKey delay based on fps; adjust if needed for smoother display.
                if cv2.waitKey(int(1000 / fps)) & 0xFF == ord('q'):
                    print("Exiting due to key press.")
                    break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    # Save the landmarks data to a JSON file.
    try:
        with open(args.landmarks_output, 'w') as f:
            json.dump(landmarks_data, f, indent=2)
        print(f"\nLandmark data saved to {args.landmarks_output}")
    except Exception as e:
        print(f"Error writing landmarks JSON file: {e}")

    print(f"Annotated video saved as {args.output}")

if __name__ == "__main__":
    main()
