# gesture-traces

This is a Python project to automatically generate stroke traces from a video clip.

## Setup

1. **Install Python**

   * If Python is not installed, download and install it from [python.org](https://www.python.org/).
2. **Clone the repository**

   ```bash
   git clone https://github.com/fanchessfan/gesture-traces.git
   ```
3. **Create a virtual environment and install dependencies**

   ```powershell
   python -m venv .venv
   .venv\Scripts\activate
   pip install -r requirements.txt
   ```

## Usage

The following scripts use an argument parser to take in inputs and outputs. See the below instructions. 

* **Detect Landmarks**

  Uses MediaPipe to detect landmarks from a video and stores them in an intermediate JSON file. Configure input/output paths and options via command-line arguments:

  ```text
  options:
    -h, --help            show this help message and exit
    --json JSON           Path to the JSON file with markers (default: markers.json)
    --input INPUT         Path to the input MP4 video file (default: video.mp4)
    --output OUTPUT       Path to the output annotated MP4 video file (default: video_marker.mp4)       
    --landmarks-output LANDMARKS_OUTPUT
                          Path to the output JSON file for gesture landmarks (default: landmarks.json)  
    --no-display          Process video without displaying the video window (faster processing)
    --no-video            Output only skeleton and markers on a black frame
  ```

* **Generate Trace**

  Takes the JSON file and generates the stroke trace, either overlayed on the original video or rendered on a blank canvas.
