from components import MatchObject
# Used by DataCollector.flush() to run inference on the frames it has batched.
from image_proccessing import get_labels
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import statistics
import os

class DataCollector:
    def __init__(self, batch_size=20):
        self.match : str = "PRACTICE 1"
        self.current_match_state : str = "Waiting" # States: Waiting, AUTONOMOUS, DRIVER CONTROL, Done
        self.robots : list[MatchObject] = []
        self.mobile_goals : list[MatchObject] = []
        # Number of sampled frames to collect before calling the inference model.
        self.batch_size = batch_size
        # Frames that have arrived from the queue but have not been analyzed yet.
        self.pending_frames = []
        # Final [x, y] coordinate rows that training_data_proccesing.py writes to CSV.
        self.data = []
        self.images = 0

    def analyze(self, info):
        frame = info[0]
        self.match = info[1]
        match_state = info[2]

        if self.current_match_state == "AUTONOMOUS" and match_state == "":
            self.current_match_state = "Waiting"
        elif self.current_match_state == "DRIVER CONTROL" and match_state == "":
            self.current_match_state = "Done"
        else:
            self.current_match_state = match_state
        
        if self.current_match_state == "AUTONOMOUS" or self.current_match_state == "DRIVER CONTROL":
            # This is called once for each sampled frame received from the image thread.
            self.pending_frames.append(frame)
            self.images += 1

            # Once a full batch is ready, process it while image collection keeps running.
            if len(self.pending_frames) >= self.batch_size:
                self.flush()
        elif self.current_match_state == "Done":
            self.save_data()

            self.pending_frames = []

        self.clear_console()
        print(f"Images proccessed: {self.images}")
        print(f"Current Match: {self.match}")
        print(f"Match State: {self.current_match_state}")
        # print(f"Robot X: {self.data[-1][0]}")
        # print(f"Robot Y: {self.data[-1][1]}")

    def fix_blobs(self, objects):
        if len(objects) <= 4:
            return objects
        
        # blob = []
        # for i in range(len(objects)):
        #     o = objects[i]
        #     if (o[2] + o[3]) / 2 <= 100:
        #         blob.append(o)

        return objects

    def flush(self):
        # If no frames are waiting, there is nothing for the model to analyze.
        if not self.pending_frames:
            return

        # Run inference on every waiting frame, using the configured batch size.
        information = get_labels(self.pending_frames, batch_size=self.batch_size)

        self.pending_frames = []

        for predictions in information:
            try:
                robot_data = [[p.x, p.y, p.height, p.width] for p in predictions if p.class_name == "Robot"]

                robot_data = self.fix_blobs(robot_data)[0]
                # info = prediction
                self.data.append([robot_data[0], robot_data[1]])

            except Exception:
                # Some frames may have no predictions, so skip those rows.
                print("Failed")

    def save_data(self):
        if self.data:
            # Show the first collected coordinate row as a quick sanity check.
            print(self.data[0])

        df = pd.DataFrame(self.data)
        print(self.data)
        print(df)
        df.columns = ["x", "y"]

        print(df.head())

        df.to_csv(f"MatchData/{self.match}", index=False)
        #np.save(f"MatchData/{self.match}I.npy", self.images)
    
    def clear_console(self):
        os.system("cls" if os.name == "nt" else "clear")