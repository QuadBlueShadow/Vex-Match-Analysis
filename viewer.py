from image_proccessing import get_r
from pathlib import Path
import cv2
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import supervision as sv

video_path = "Videos/MechaR-16-1.mkv"

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

saved_images = np.load("TrainingData/Qual32I.npy", allow_pickle=True)

fps = 60
delay = int(1000 / fps) if fps > 0 else 33  # milliseconds
delay = 200

index = 0
bounding_box_annotator = sv.BoxAnnotator() 
label_annotator = sv.LabelAnnotator() 

annotated_images = []

if False:
    while True:
        ret, frame = cap.read()

        if not ret:
            break

        #cv2.imshow("Video", frame)
        # load the results into the supervision Detections api detections = sv.Detections.from_inference(results) 
        # create supervision annotators 

        if index % 5 == 0:
            results = get_r([frame])[0]

            detections = sv.Detections.from_inference(results)

            # annotate the image with our inference results 
            annotated_image = bounding_box_annotator.annotate( scene=frame, detections=detections) 
            annotated_image = label_annotator.annotate( scene=annotated_image, detections=detections) 
            # display the image 
            annotated_images.append(annotated_image)
            print(index)

        index += 1

        # Press q to quit
        # if cv2.waitKey(delay) & 0xFF == ord("q"):
        #     break
else:
    for frame in saved_images:
        if index % 1 == 0:
            results = get_r([frame])[0]

            detections = sv.Detections.from_inference(results)

            # annotate the image with our inference results 
            annotated_image = bounding_box_annotator.annotate( scene=frame, detections=detections) 
            annotated_image = label_annotator.annotate( scene=annotated_image, detections=detections) 
            # display the image 
            annotated_images.append(annotated_image)

        index += 1

        # Press q to quit
        # if cv2.waitKey(delay) & 0xFF == ord("q"):
        #     break

for frame in annotated_images:
    cv2.imshow("Video", frame)

    # Press q to quit
    if cv2.waitKey(delay) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()