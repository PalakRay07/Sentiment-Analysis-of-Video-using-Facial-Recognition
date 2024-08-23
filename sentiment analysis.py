# Install the required libraries
# pip install fer

# Import the necessary libraries
import cv2
from fer import FER
import matplotlib.pyplot as plt

# Load the video
video_path = 'D:\yy.mp4'
video = cv2.VideoCapture(video_path)

# Create a FER object
detector = FER()

# Initialize an empty list to store emotion labels
emotion_labels = []

# Iterate through the frames
while True:
    ret, frame = video.read()
    if not ret:
        break
    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Detect emotions in the frame
    emotions = detector.detect_emotions(frame)
    if emotions is not None and len(emotions) > 0:
        # Get the dominant emotion
        dominant_emotion = max(emotions[0]['emotions'], key=emotions[0]['emotions'].get)
        emotion_labels.append(dominant_emotion)
    else:
        emotion_labels.append('Unknown')

# Release the video capture object
video.release()

# Perform sentiment analysis based on the collected emotions
positive_emotions = emotion_labels.count('happy') + emotion_labels.count('surprise')
negative_emotions = emotion_labels.count('sad') + emotion_labels.count('angry')
neutral_emotions = emotion_labels.count('neutral')

overall_sentiment = 'Positive' if positive_emotions > negative_emotions + neutral_emotions else 'Negative' if negative_emotions > positive_emotions + neutral_emotions else 'Neutral'

# Print the overall sentiment
print(f'Overall sentiment: {overall_sentiment}')

# Visualize the results
plt.bar(['Happy', 'sad', 'Neutral', 'Angry'], [positive_emotions, negative_emotions, neutral_emotions, emotion_labels.count('angry')])
plt.title('Emotion Distribution')
plt.xlabel('Emotions')
plt.ylabel('Count')
plt.show()
