from collection import DataCollector
# Queue safely passes frames between the image-gathering thread and DataCollector thread.
from queue import Queue
# Thread lets image collection and data processing run at the same time.
from threading import Event, Thread
import cv2
import pytesseract
import time

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Unique marker placed in the queue to tell the DataCollector thread to stop.
STOP = object()
# Only every 20th video frame is kept, matching the script's original sampling logic.
FRAME_SAMPLE_INTERVAL = 5
# Stop once this many sampled frames have been collected.
MAX_IMAGES = 60
# Number of sampled frames DataCollector sends to the model at a time.
BATCH_SIZE = 5
# Seconds to wait for each worker during Ctrl+C cleanup before saving what is already collected.
SHUTDOWN_JOIN_TIMEOUT = 2

def collect_images(cap, frame_queue, stop_event):
    # Runs on its own thread and only handles reading/sampling frames from the video.
    index = 0
    images_saved = 0

    try:
        while not stop_event.is_set():
            ret, frame = cap.read()

            if not ret:
                break

            if index % FRAME_SAMPLE_INTERVAL == 0:
                match = frame[0:50, 0:450]
                match_state = frame[1039:1280, 1670:1920]

                match = pytesseract.image_to_string(match).strip()
                match_state = pytesseract.image_to_string(match_state).strip()

                # Hand this same frame to the DataCollector thread for analysis.
                frame_queue.put([frame, match, match_state, int(cap.get(cv2.CAP_PROP_POS_FRAMES))])
                images_saved += 1

            index += 1
    finally:
        stop_event.set()
        # Always tell the DataCollector thread there are no more frames to process.
        frame_queue.put(STOP)

def run_data_collector(frame_queue, collector):
    # Runs on its own thread and processes frames as soon as collect_images queues them.
    while True:
        # Wait here until a frame, or the STOP marker, is available.
        info = frame_queue.get()

        try:
            if info is STOP:
                # Analyze any leftover frames that did not fill a complete batch.
                collector.flush()
                break

            # Add this frame to DataCollector's batch and process if the batch is full.
            collector.analyze(info)
        finally:
            # Mark this queue item as handled.
            frame_queue.task_done()

video_path = "Videos/Practice18.mkv"
video_path = "https://play.boxcast.com/p/xg5f6cth5ddcmjsvzpsl/v/1080p-byteranges.m3u8?Expires=1780358400&Signature=dn8wSUbvT5qeTPQdN1xW-QEWvlB0hN8b1pYfEFyYWdNdGqQsVD8N65FI-fKKem9ltR0xuc-LRqWLf0z2QI~pFpIc5hh7C6LKJtOAS8Bg~IBm8DuVZi7DbO5kuQVYRf9tOmz2bWqoyrBe~oG1soJKq4CSznL9QKiw~sGCa6KfY3PUChgonSi3l8qlGfbbIg2E2SZkEOz5bbwcuElGBJjNZftQU27xsOH36pNSONwDojZCUuLAYclHHE3HCcrcXRZnzPCZy-~4izW6PJvBjynzo1MP01U-RIlStx4u9LIjNPtnBy6zvkO33zobBklXT3bkRqK3GiQF1Nw1AOpKQDRcZQ__&Key-Pair-Id=APKAJ7GUCBQUK6NTWZCA"

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

current_frame = 36_000
try:
    with open("TrainingLog.txt", 'r') as file:
        current_frame = int(file.readlines()[0])
        print(f"Found checkpoint at {current_frame}")
except Exception:
    print("No checkpoint found")

cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)

# Thread-safe bridge that lets one thread produce frames and the other consume them.
frame_queue = Queue()
# Lets Ctrl+C ask the image collection thread to stop before saving partial results.
stop_event = Event()
# Owns the batching logic and collected x/y coordinate data.
collector = DataCollector(batch_size=BATCH_SIZE)

# Thread that waits for queued frames and sends them through DataCollector.
collector_thread = Thread(target=run_data_collector, args=(frame_queue, collector), name="data-collector")
# Thread that reads frames from the video and queues sampled frames for analysis.
image_thread = Thread(target=collect_images, args=(cap, frame_queue, stop_event), name="image-collector")

try:
    # Start the consumer first so it is ready when frames begin arriving.
    collector_thread.start()
    # Start the producer that reads and queues frames from the video.
    image_thread.start()

    # Wait until frame collection has stopped.
    image_thread.join()
    # Wait until every queued frame has been processed by DataCollector.
    collector_thread.join()
except KeyboardInterrupt:
    print("\nCtrl+C received. Stopping frame collection and saving collected data...")
    stop_event.set()
    cap.release()
finally:
    stop_event.set()

    if image_thread.is_alive():
        image_thread.join(timeout=SHUTDOWN_JOIN_TIMEOUT)
    if collector_thread.is_alive():
        collector_thread.join(timeout=SHUTDOWN_JOIN_TIMEOUT)

    if image_thread.is_alive():
        print("Image collection is still stopping; saving the data collected so far.")
    if collector_thread.is_alive():
        print("Data collector is still processing; saving the data collected so far.")

    if cap.isOpened():
        cap.release()

    collector.flush()
    collector.save_data()

    with open("TrainingLog.txt", 'w') as file:
        file.writelines(str(collector.frame))