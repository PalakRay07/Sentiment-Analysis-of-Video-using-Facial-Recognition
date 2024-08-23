import cv2
import numpy as np
from flask import Flask, render_template, request, jsonify
from fer import FER

app = Flask(__name__)
app.template_folder = 'templates'

# Render the index.html page
@app.route('/')
def index():
    return render_template('index.html')

# Process the video and return the emotion detection results
@app.route('/process_video', methods=['POST'])
def process_video():
    # Get the uploaded video file
    video_file = request.files['video']
    video_filename = video_file.filename

    # Save the video file to disk
    video_file.save(video_filename)

    # Open the video file
    video_capture = cv2.VideoCapture(video_filename)

    # Get the total number of frames
    total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))

    # Initialize the FER detector
    detector = FER()

    # Initialize the emotion counts
    happy_count = 0
    sad_count = 0
    neutral_count = 0

    # Iterate through each frame in the video
    while True:
        ret, frame = video_capture.read()
        if not ret:
            break

        # Detect emotions in the frame
        result = detector.detect_emotions(frame)

        # Process the result to determine the emotion
        for face in result:
            emotions = face['emotions']
            if emotions['happy'] > emotions['sad']:
                happy_count += 1
            elif emotions['sad'] > emotions['neutral']:
                sad_count += 1
            else:
                neutral_count += 1

    # Release the video capture object
    video_capture.release()

    # Determine the maximum number of frames among happy, sad, and neutral
    max_frames = max(happy_count, sad_count, neutral_count)

    # Return the emotion detection results
    emotion_results = {
        'total_frames': total_frames,
        'maximum_frames': max_frames,
        'happy_frames': happy_count,
        'sad_frames': sad_count,
        'neutral_frames': neutral_count
    }

    return jsonify(emotion_results)

if __name__ == '__main__':
    app.run(debug=True)
