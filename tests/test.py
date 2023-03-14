from nano import extract_features
import matplotlib.pyplot as plt

raw, events, info = extract_features.get_raw_signal(
    fast5_file="../fast5/single/0/0a6006d9-460a-4eb8-88cd-5be1196f3230.fast5",
    corrected_group="RawGenomeCorrected_000",
    basecall_subgroup="BaseCalled_template",
)

if raw is not None:
    plt.plot(raw)
    plt.show()
print(events)
print(info)


# from nano.utils import process_utils
# fast5s = process_utils.get_fast5s(fast5_dir="../fast5/single", recursive=False)
# print(len(fast5s))
#
# queue = process_utils.Queue()
# print(queue.size)
