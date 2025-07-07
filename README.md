# gesture-traces
This is a python project to automatically generate stroke traces from a video clip. 
# setup
- Install python
- Setup conda/venv virtual environment
- requirements.txt
# use
- Detect Landmarks: Uses mediapipe to detect the landmarks from a video, and stores them in an intermediate json file.
-  and command line arguments
- Generate trace: Takes the json file, and generates the trace from it, either with or without the original video.
