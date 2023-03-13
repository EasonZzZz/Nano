import glob
import multiprocessing.queues
import os

base_pairs = {
    'A': 'T', 'C': 'G', 'G': 'C', 'T': 'A', 'N': 'N'
}
base2int = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'N': 4}
int2base = {0: 'A', 1: 'C', 2: 'G', 3: 'T', 4: 'N'}


def display_args(args):
    """
    Display the arguments.
    """
    print("#" * 80)
    print("Arguments:")
    for arg in vars(args):
        print("\t{}: {}".format(arg, getattr(args, arg)))
    print("#" * 80)


def complement_sequence(sequence):
    """
    Complement a sequence.
    :param sequence: The sequence to complement.
    :return: The complement of the sequence.
    """
    try:
        return "".join([base_pairs[base] for base in sequence])
    except KeyError:
        raise ValueError("Invalid base in sequence: {}".format(sequence))


def get_fast5s(fast5_dir, recursive=True):
    """
    Get all fast5 files in a directory.
    :param fast5_dir: Path to the directory containing fast5 files.
    :param recursive: Whether to search for fast5 files recursively.
    :return: A list of paths to fast5 files.
    """
    fast5_dir = os.path.abspath(fast5_dir)
    if recursive:
        fast5s = glob.glob(os.path.join(fast5_dir, "**", "*.fast5"), recursive=True)
    else:
        fast5s = glob.glob(os.path.join(fast5_dir, "*.fast5"))
    return fast5s


class SharedCounter(object):
    """
    A synchronized shared counter.
    """

    def __init__(self, n=0):
        self.count = multiprocessing.Value("i", n)

    def increment(self, n=1):
        with self.count.get_lock():
            self.count.value += n

    @property
    def value(self):
        with self.count.get_lock():
            return self.count.value


class Queue(multiprocessing.queues.Queue):
    """
    A synchronized queue.
    """

    def __init__(self, *args, **kwargs):
        super(Queue, self).__init__(*args, ctx=multiprocessing.get_context(), **kwargs)
        self._size = SharedCounter(0)

    def put(self, *args, **kwargs):
        super(Queue, self).put(*args, **kwargs)
        self._size.increment(1)

    def get(self, *args, **kwargs):
        value = super(Queue, self).get(*args, **kwargs)
        self._size.increment(-1)
        return value

    @property
    def size(self):
        return self._size.value

    def empty(self) -> bool:
        return self.size == 0
