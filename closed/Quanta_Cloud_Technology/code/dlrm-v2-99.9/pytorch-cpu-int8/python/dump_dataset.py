import sys
import os
import numpy as np

data_path = sys.argv[1]
output_path = sys.argv[2]

tar_fea = 1   # single target
den_fea = 13  # 13 dense  features
spa_fea = 26  # 26 sparse features
tot_fea = tar_fea + den_fea + spa_fea

data_file = open(os.path.join(data_path, 'terabyte_processed_test.bin'), 'rb')
raw_data = data_file.read()
array = np.frombuffer(raw_data, dtype=np.int32).reshape(-1, tot_fea)

x_int_batch = array[:, 1:14]
x_cat_batch = array[:, 14:]
y_batch = array[:, 0].reshape(-1)

np.save(os.path.join(output_path, "x_int_batch.npy"), x_int_batch)
np.save(os.path.join(output_path, "x_cat_batch.npy"), x_cat_batch)
np.save(os.path.join(output_path, "y_batch.npy"), y_batch)

# process feature count file
day_day_count = np.load(os.path.join(data_path, 'day_day_count.npz'))['total_per_file']
day_fea_count = np.load(os.path.join(data_path, 'day_fea_count.npz'))['counts']
np.save(os.path.join(output_path, 'day_day_count.npy'), day_day_count)
np.save(os.path.join(output_path, 'day_fea_count.npy'), day_fea_count)
