# Smart-Traffic-Light-System

## Traffic Light Control System

A smart traffic light control system that uses YOLO object detection and machine learning to optimize traffic flow. The system detects vehicles in each lane and predicts optimal green/red light durations based on traffic density.

## Features

- Real-time vehicle detection using YOLOv8
- ML-based prediction of optimal traffic light timing
- Support for 2-4 lanes
- Interactive GUI built with tkinter
- Visual traffic light simulation
- Real-time vehicle count display
- Dynamic traffic light timing adjustments

## Prerequisites

- Python 3.x
- OpenCV (cv2)
- ultralytics (YOLO)
- tkinter
- pickle

## Installation

1. Clone this repository
```bash
git clone https://github.com/CreativeCoder77/Smart-Traffic-Light-System.git
```
2. Install required dependencies:
```bash
pip install opencv-python ultralytics
```

3. Ensure you have the following model files in the `models` directory:
   - `model_green.pkl` - ML model for green light duration prediction
   - `model_red.pkl` - ML model for red light duration prediction
   - `yolov8s-seg.pt` - YOLOv8 segmentation model

## Usage

1. Run the application:
```bash
python traffic_light_system.py
```

2. Use the GUI to:
   - Set the number of lanes (2-4)
   - Select images for each lane
   - Detect vehicles using the "Detect Vehicles" button
   - Start/Stop the traffic light simulation

## How It Works

1. **Vehicle Detection**
   - Uses YOLOv8 to detect vehicles in uploaded images
   - Recognizes cars, motorcycles, buses, and trucks
   - Provides visual detection results with bounding boxes

2. **Traffic Light Timing**
   - ML models predict optimal green and red light durations
   - Predictions based on vehicle count in each lane
   - Prioritizes lanes with higher traffic density

3. **Simulation**
   - Visual representation of traffic lights for each lane
   - Real-time countdown timers
   - Dynamic status updates
   - Automatic cycling between lanes based on predicted timings

## GUI Components

- Lane configuration controls
- Image selection buttons for each lane
- Vehicle count displays
- Predicted timing displays
- Traffic light status indicators
- Detection results text area
- Simulation control buttons

## Error Handling

- Input validation for lane numbers
- Image selection verification
- Exception handling during vehicle detection and simulation
- User-friendly error messages

## Notes

- The system uses pre-trained ML models for timing predictions
- Vehicle detection is optimized for common vehicle types
- The simulation can be stopped at any time
- Detected vehicle images are displayed in separate windows

## License
- MIT License
