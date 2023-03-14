import glob
import os
import multiprocessing
import multiprocessing.queues as mpq

base_pairs = {
    'A': 'T', 'C': 'G', 'G': 'C', 'T': 'A', 'N': 'N'
}
base2int = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'N': 4}
int2base = {0: 'A', 1: 'C', 2: 'G', 3: 'T', 4: 'N'}
iupac_alphabets = {
    'A': ['A'], 'C': ['C'], 'G': ['G'], 'T': ['T'], 'N': ['A', 'C', 'G', 'T'],
    'R': ['A', 'G'], 'Y': ['C', 'T'], 'S': ['G', 'C'], 'W': ['A', 'T'],
    'K': ['G', 'T'], 'M': ['A', 'C'], 'B': ['C', 'G', 'T'], 'D': ['A', 'G', 'T'],
    'H': ['A', 'C', 'T'], 'V': ['A', 'C', 'G'], 'X': ['A', 'C', 'G', 'T']
}


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


def _convert_motif_seq(motif_seq, is_dna=True):
    outbases = []
    for base in motif_seq:
        if is_dna:
            outbases.append(iupac_alphabets[base.upper()])
        else:
            pass

    def recursive_permute(base_list):
        if len(base_list) == 1:
            return base_list[0]
        elif len(base_list) == 2:
            pseqs = []
            for base in base_list[0]:
                for base2 in base_list[1]:
                    pseqs.append(base + base2)
            return pseqs
        else:
            pseqs = recursive_permute(base_list[1:])
            pseq_list = [base_list[0], pseqs]
            return recursive_permute(pseq_list)

    return recursive_permute(outbases)


def get_motif_seqs(motifs, is_dna=True):
    """
    Get all motifs.
    :param motifs: The motifs to extract features for.
    :param is_dna: Whether the motifs are DNA motifs.
    """
    ori_motif_seqs = motifs.strip().split(",")
    motif_seqs = []
    for ori_motif in ori_motif_seqs:
        motif_seqs += _convert_motif_seq(ori_motif, is_dna=is_dna)
    return motif_seqs


def get_ref_loc_of_methyl_site(seq, motif_set, mod_locs=0):
    """
    Get the read locations of the modified sites.
    :param seq: The sequence to search.
    :param motif_set: The set of motifs to search for.
    :param mod_locs: The location of the modified base.
    :return: A list of the read locations of the modified sites.
    """
    seq_len = len(seq)
    motif_set = set(motif_set)
    motif_len = len(list(motif_set)[0])
    sites = []
    for i in range(seq_len - motif_len + 1):
        if seq[i:i + motif_len] in motif_set:
            sites.append(i + mod_locs)
    return sites


class Queue(mpq.Queue):
    """ A portable implementation of multiprocessing.Queue.
    Because of multithreading / multiprocessing semantics, Queue.qsize() may
    raise the NotImplementedError exception on Unix platforms like Mac OS X
    where sem_getvalue() is not implemented. This subclass addresses this
    problem by using a synchronized shared counter (initialized to zero) and
    increasing / decreasing its value every time the put() and get() methods
    are called, respectively. This not only prevents NotImplementedError from
    being raised, but also allows us to implement a reliable version of both
    qsize() and empty().
    """

    def __init__(self, maxsize=-1, block=True, timeout=None):
        self.block = block
        self.timeout = timeout
        super().__init__(maxsize, ctx=multiprocessing.get_context())
        self.size = SharedCounter(0)

    def __getstate__(self):
        return super().__getstate__() + (self.size,)

    def __setstate__(self, state):
        super().__setstate__(state[:-1])
        self.size = state[-1]

    def put(self, *args, **kwargs):
        super(Queue, self).put(*args, **kwargs)
        self.size.increment(1)

    def get(self, *args, **kwargs):
        item = super(Queue, self).get(*args, **kwargs)
        self.size.increment(-1)
        return item

    def qsize(self):
        """ Reliable implementation of multiprocessing.Queue.qsize() """
        return self.size.value

    def empty(self):
        """ Reliable implementation of multiprocessing.Queue.empty() """
        return not self.qsize()

    def clear(self):
        """ Remove all elements from the Queue. """
        while not self.empty():
            self.get()


class SharedCounter(object):
    """ A synchronized shared counter.
    The locking done by multiprocessing.Value ensures that only a single
    process or thread may read or write the in-memory ctypes object. However,
    in order to do n += 1, Python performs a read followed by a write, so a
    second process may read the old value before the new one is written by the
    first process. The solution is to use a multiprocessing.Lock to guarantee
    the atomicity of the modifications to Value.
    This class comes almost entirely from Eli Bendersky's blog:
    http://eli.thegreenplace.net/2012/01/04/shared-counter-with-pythons-multiprocessing/
    """

    def __init__(self, n=0):
        self.count = multiprocessing.Value('i', n)

    def increment(self, n=1):
        """ Increment the counter by n (default = 1) """
        with self.count.get_lock():
            self.count.value += n

    @property
    def value(self):
        """ Return the value of the counter """
        return self.count.value
