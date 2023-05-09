import torch

READS_GROUP = "Raw/Reads"
GLOBAL_KEY = "UniqueGlobalKey"
QUEUE_BORDER_SIZE = 1000
F5BATCH_QUEUE_BORDER_SIZE = 100
SLEEP_TIME = 1
USE_CUDA = torch.cuda.is_available()

BASE2INT = {
    "A": 0, "C": 1, "G": 2, "T": 3, "N": 4,
    "0": 0, "1": 1, "2": 2, "3": 3, "4": 4,
    'a': 0, 'c': 1, 'g': 2, 't': 3, 'n': 4
}
INT2BASE = {0: "A", 1: "C", 2: "G", 3: "T", 4: "N"}
