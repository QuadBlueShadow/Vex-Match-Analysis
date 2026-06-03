from components import MatchObject
from image_proccessing import get_labels
import os
import time
from database import DataBase

class DataCollector:
    def __init__(self, batch_size=20):
        self.match : str = "PRACTICE 1"
        self.current_match_state : str = "Waiting" # States: Waiting, AUTONOMOUS, DRIVER CONTROL, Done
        self.robots : list[MatchObject] = []
        self.mobile_goals : list[MatchObject] = []
        self.reset_robots = True
        # Number of sampled frames to collect before calling the inference model.
        self.batch_size = batch_size
        # Frames that have arrived from the queue but have not been analyzed yet.
        self.pending_frames = []
        # Final [x, y] coordinate rows that training_data_proccesing.py writes to CSV.
        self.data = []
        self.images = 0
        self.frame = 0
        self.log = None
        self.check_loaded = True
        self.previous = None
        self.table_parameters = ["x1", "y1", "x2", "y2", "x3", "y3", "x4", "y4"]
        self.bot_count = 0

        for i in range(4):
            self.robots.append(MatchObject())

        self.db = DataBase("InnovateDivision", [self.match.replace(" ", "_").lower(), "x1 FLOAT", "y1 FLOAT", "x2 FLOAT", "y2 FLOAT", "x3 FLOAT", "y3 FLOAT", "x4 FLOAT", "y4 FLOAT"])

    def analyze(self, info):
        frame = info[0]
        self.match = info[1]
        match_state = info[2]
        self.frame = info[3]

        if self.current_match_state != "AUTONOMOUS" and match_state == "AUTONOMOUS":
            self.db.change_table([self.match.replace(" ", "_").lower(), "x1 FLOAT", "y1 FLOAT", "x2 FLOAT", "y2 FLOAT", "x3 FLOAT", "y3 FLOAT", "x4 FLOAT", "y4 FLOAT"])

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

            self.data = []
            self.pending_frames = []
            self.reset_robots = True

        self.clear_console()
        print(f"Images proccessed: {self.images}")
        print(f"Current Match: {self.match}")
        print(f"Match State: {self.current_match_state}")
        print(f"Frame: {self.frame}")
        if len(self.data) > 0:
            for d in self.data[-1]:
                print(d)

    def fix_blobs(self, objects):
        if len(objects) <= 1 or len(objects) == self.bot_count:
            return objects
        
        cluster_distance = 110
        cluster_distance_sq = cluster_distance ** 2
        clusters = []

        for o in objects:
            matching_clusters = []

            # Treat every detection as part of a possible blob. Full robot detections
            # will usually stay as one-item clusters, while fragments close together
            # will be grouped into one cluster.
            for cluster in clusters:
                if any((o[0] - p[0]) ** 2 + (o[1] - p[1]) ** 2 <= cluster_distance_sq for p in cluster):
                    matching_clusters.append(cluster)

            if not matching_clusters:
                clusters.append([o])
                continue

            # If this detection connects multiple clusters, merge those clusters so
            # one scattered blob still becomes a single output point.
            target_cluster = matching_clusters[0]
            target_cluster.append(o)

            for cluster in matching_clusters[1:]:
                target_cluster.extend(cluster)
                clusters.remove(cluster)

        output = []

        for cluster in clusters:
            if len(cluster) == 1:
                output.append(cluster[0])
                continue

            # Collapse each multi-detection blob into one averaged robot position.
            avg_x = sum(p[0] for p in cluster) / len(cluster)
            avg_y = sum(p[1] for p in cluster) / len(cluster)

            output.append([avg_x, avg_y])

        return output
    
    def assign_robots(self, data:list):
        if self.check_loaded and len(self.db.get_data(f"SELECT * FROM {self.match.replace(" ", "_").lower()}")) > 0:
            self.check_loaded = False
            self.reset_robots = False

            old_data = self.db.get_data(f"SELECT * FROM {self.match.replace(" ", "_").lower()}")[-1]
            for i in range(0, len(old_data), 2):
                if old_data[i] != 0:
                    self.bot_count += 1

            self.robots[0].set_position([old_data[0], old_data[1]])
            if self.bot_count > 1: self.robots[1].set_position([old_data[2], old_data[3]])
            if self.bot_count > 2: self.robots[2].set_position([old_data[4], old_data[5]])
            if self.bot_count > 3: self.robots[3].set_position([old_data[6], old_data[7]])

            for i in range(len(data)):
                smallest:MatchObject = min(self.robots, key=lambda r: r.get_dist(data[i]))
                smallest.set_position(data[i])

            print(f"Checkpoint found -> Robots: {self.bot_count}, Match: {self.match}")
            time.sleep(3)
        elif self.reset_robots:
            avg_x = sum(d[0] for d in data) / len(data)
            left = [d for d in data if d[0] < avg_x]
            right = [d for d in data if d[0] > avg_x]

            if len(left) > 0: self.robots[0].set_position(min(left, key=lambda l: l[1]))
            if len(left) > 1: self.robots[1].set_position(max(left, key=lambda l: l[1]))

            if len(left) > 0: self.robots[2].set_position(min(right, key=lambda r: r[1]))
            if len(right) > 1: self.robots[3].set_position(max(right, key=lambda r: r[1]))

            self.reset_robots = False
            self.check_loaded = False

            self.bot_count = len(data)

            print(f"Initializing robots -> L: {len(left)}, R: {len(right)}")
            time.sleep(3)
        else:
            for i in range(len(data)):
                smallest:MatchObject = min(self.robots, key=lambda r: r.get_dist(data[i]))
                smallest.set_position(data[i])

        output = []
        for r in self.robots:
            output.extend(r.get_position())

        return output

    def flush(self):
        # If no frames are waiting, there is nothing for the model to analyze.
        if not self.pending_frames:
            return

        # Run inference on every waiting frame, using the configured batch size.
        information = get_labels(self.pending_frames, batch_size=self.batch_size)

        self.pending_frames = []

        for predictions in information:
            robot_data = [[p.x, p.y] for p in predictions if p.class_name == "Robot"]

            robot_data = self.fix_blobs(robot_data)
            robot_data = self.assign_robots(robot_data)
            self.data.append(robot_data)

    def save_data(self):
        self.flush()
        self.db.insert_data(self.data, self.table_parameters, self.match.replace(" ", "_").lower())
    
    def clear_console(self):
        os.system("cls" if os.name == "nt" else "clear")