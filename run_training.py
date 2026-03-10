"""Wrapper to run train_model.py with unbuffered output to file."""
import sys
import os
import warnings

# Suppress all warnings
warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"
os.environ["PYTHONUNBUFFERED"] = "1"

# Redirect stdout and stderr to file AND console
log_path = os.path.join(os.path.dirname(__file__), "train_v17_progress.txt")

class Tee:
    def __init__(self, *streams):
        self.streams = streams
    def write(self, data):
        for s in self.streams:
            s.write(data)
            s.flush()
    def flush(self):
        for s in self.streams:
            s.flush()

log_file = open(log_path, "w", encoding="utf-8")
sys.stdout = Tee(sys.__stdout__, log_file)
sys.stderr = Tee(sys.__stderr__, log_file)

# Now run the actual training
sys.argv = ["train_model.py", "--multi-symbol"]
exec(open("train_model.py", encoding="utf-8").read())
