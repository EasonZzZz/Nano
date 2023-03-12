import os
import h5py

f = h5py.File('fast5/0a0ace70-81bb-4b8d-901c-16bf3173c89f.fast5', 'r')
reads = next(iter(f['/Raw/Reads'].values()))
all_raw_signal = reads['Signal'][:]
print(len(all_raw_signal))
