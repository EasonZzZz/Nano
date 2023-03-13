from nano import extract_features

raw, events = extract_features.get_raw_signal(
    fast5_file="../fast5/single/0/0a379e9d-8d76-4a0e-bac0-341f81cc349f.fast5",
    corrected_group="RawGenomeCorrected_000",
    basecall_subgroup="BaseCalled_template",
)

print(raw)
print(events)


from nano.utils import process_utils
fast5s = process_utils.get_fast5s(fast5_dir="../fast5/single", recursive=False)
print(len(fast5s))

queue = process_utils.Queue()
print(queue.size)
