from components import MatchObject
# Used by DataCollector.flush() to run inference on the frames it has batched.
from image_proccessing import get_labels

class DataCollector:
    def __init__(self, batch_size=20):
        self.match : str = "PRACTICE 1"
        self.current_match_state : str = "Waiting" # States: Waiting, AUTONOMOUS, DRIVER CONTROL, DONE
        self.robots : list[MatchObject] = []
        self.mobile_goals : list[MatchObject] = []
        # Number of sampled frames to collect before calling the inference model.
        self.batch_size = batch_size
        # Frames that have arrived from the queue but have not been analyzed yet.
        self.pending_frames = []
        self.unlabeled_frames = []
        # Final [x, y] coordinate rows that training_data_proccesing.py writes to CSV.
        self.data = []

    def analyze(self, frame):
        # This is called once for each sampled frame received from the image thread.
        self.pending_frames.append(frame)

        # Once a full batch is ready, process it while image collection keeps running.
        if len(self.pending_frames) >= self.batch_size:
            self.unlabeled_frames = self.pending_frames.copy()
            self.pending_frames = []
            self.flush()

    def flush(self):
        # If no frames are waiting, there is nothing for the model to analyze.
        if not self.pending_frames:
            return

        # Run inference on every waiting frame, using the configured batch size.
        information = get_labels(self.unlabeled_frames, batch_size=self.batch_size)

        self.unlabeled_frames = []

        for predictions in information:
            try:
                # Keep the first prediction for each frame, matching the old script behavior.
                info = predictions[0]
                # Store only the x/y coordinates needed for the training CSV.
                self.data.append([info.x, info.y])
            except Exception:
                # Some frames may have no predictions, so skip those rows.
                print("Failed")
