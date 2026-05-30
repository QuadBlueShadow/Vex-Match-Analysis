from collection import DataCollector
from pathlib import Path
# Queue safely passes frames between the image-gathering thread and DataCollector thread.
from queue import Queue
# Thread lets image collection and data processing run at the same time.
from threading import Thread
import cv2
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Unique marker placed in the queue to tell the DataCollector thread to stop.
STOP = object()
# Only every 20th video frame is kept, matching the script's original sampling logic.
FRAME_SAMPLE_INTERVAL = 2
# Stop once this many sampled frames have been collected.
MAX_IMAGES = 60
# Number of sampled frames DataCollector sends to the model at a time.
BATCH_SIZE = 20

def collect_images(cap, frame_queue, images):
    # Runs on its own thread and only handles reading/sampling frames from the video.
    index = 0

    try:
        while True:
            ret, frame = cap.read()

            if not ret:
                break

            if index % FRAME_SAMPLE_INTERVAL == 0:
                # Keep this frame for the .npy training image output.
                images.append(frame)
                # Hand this same frame to the DataCollector thread for analysis.
                frame_queue.put(frame)
                print(index)

            index += 1

            if len(images) >= MAX_IMAGES:
                break
    finally:
        # Always tell the DataCollector thread there are no more frames to process.
        frame_queue.put(STOP)

def run_data_collector(frame_queue, collector):
    # Runs on its own thread and processes frames as soon as collect_images queues them.
    while True:
        # Wait here until a frame, or the STOP marker, is available.
        frame = frame_queue.get()

        try:
            if frame is STOP:
                # Analyze any leftover frames that did not fill a complete batch.
                collector.flush()
                break

            # Add this frame to DataCollector's batch and process if the batch is full.
            collector.analyze(frame)
        finally:
            # Mark this queue item as handled.
            frame_queue.task_done()

video_path = "Videos/Practice18.mkv"
video_path = "https://play.boxcast.com/p/xg5f6cth5ddcmjsvzpsl/v/1080p-byteranges.m3u8?Expires=1780185600&Signature=fovhaj8AP0l0FEbf~QW6OSyGfjl8D56ep7ofkHthT22a--pYubD0TVC3PSTXCGQp8mIYOHqYCyTZymwEDGIo0umq4Tn-4xqdc6UQP2vXj0QOzpTulorMQQag5jcVmfssv--iqPDNhQ3o-e~hvjDHGuK43j2425vWownPXUls~5f25ehBs3BY5wqecViOQUqx-7ALrr2RO8YmsbfSwsAksICbF6N3UWv9D1-EQEsi1DUxv188LhR2T2T9Hi2Adn6uJUG3Zv34k03dY9NMjuvUNqMiqgitnswXU-LtFzOPxJ4m1XjGlWZ5CRPLCj3K8uI5GqQl2tYZLAKVKa0Y9Mg2cw__&Key-Pair-Id=APKAJ7GUCBQUK6NTWZCA"

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    raise RuntimeError("Could not open video file")

fps = cap.get(cv2.CAP_PROP_FPS)
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print(f"FPS: {fps}")
print(f"Frames: {frame_count}")
print(f"Resolution: {width}x{height}")

fps = 60
delay = int(1000 / fps) if fps > 0 else 33  # milliseconds

cap.set(cv2.CAP_PROP_POS_FRAMES, 10 * 8000)

images = []
# Thread-safe bridge that lets one thread produce frames and the other consume them.
frame_queue = Queue()
# Owns the batching logic and collected x/y coordinate data.
collector = DataCollector(batch_size=BATCH_SIZE)

# Thread that waits for queued frames and sends them through DataCollector.
collector_thread = Thread(target=run_data_collector, args=(frame_queue, collector), name="data-collector")
# Thread that reads frames from the video and queues sampled frames for analysis.
image_thread = Thread(target=collect_images, args=(cap, frame_queue, images), name="image-collector")

# Start the consumer first so it is ready when frames begin arriving.
collector_thread.start()
# Start the producer that reads and queues frames from the video.
image_thread.start()

# Wait until frame collection has stopped.
image_thread.join()
# Wait until every queued frame has been processed by DataCollector.
collector_thread.join()

cap.release()

data = collector.data

if data:
    # Show the first collected coordinate row as a quick sanity check.
    print(data[0])

df = pd.DataFrame(data)
df.columns = ["x", "y"]

print(df.head())

df.to_csv("TrainingData/Qual32", index=False)
np.save("TrainingData/Qual32I.npy", images)

df.plot(x='x', y='y')

plt.show()