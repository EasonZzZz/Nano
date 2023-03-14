import pandas as pd

from nano import extract_features
import matplotlib.pyplot as plt

# raw, events, info = extract_features.get_raw_signal(
#     fast5_file="../data/single/0/0a379e9d-8d76-4a0e-bac0-341f81cc349f.fast5",
#     corrected_group="RawGenomeCorrected_000",
#     basecall_subgroup="BaseCalled_template",
# )
#
# if raw is not None:
#     plt.plot(raw)
#     plt.show()
# print(events)
# print(info)


# from nano.utils import process_utils
# fast5s = process_utils.get_fast5s(fast5_dir="../data/single", recursive=False)
# print(len(fast5s))
#
# queue = process_utils.Queue()
# print(queue.size)

# from nano.utils.ref_helper import DNAReference
#
# ref = DNAReference('../data/ref_960.fa')
# print(ref.get_chrom_names())
# print(ref.get_chrom_length(ref.get_chrom_names()[0]))
#
# df = pd.DataFrame()
# features = pd.DataFrame({
#     "a": [1, 2, 3],
#     "b": [4, 5, 6],
# })
# df = pd.concat([df, features], axis=0)
# print(df)

from nano.utils.process_utils import Queue

queue = Queue()
queue.put(1)
queue.put(2)
queue.put(3)
print(queue.get())
print(queue.get())
print(queue.get())
print(queue.get())
