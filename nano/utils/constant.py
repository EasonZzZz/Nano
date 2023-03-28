import torch

READS_GROUP = "Raw/Reads"
GLOBAL_KEY = "UniqueGlobalKey"
QUEUE_BORDER_SIZE = 1000
F5BATCH_QUEUE_BORDER_SIZE = 100
SLEEP_TIME = 1
USE_CUDA = torch.cuda.is_available()
