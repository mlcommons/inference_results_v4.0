# Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


from __future__ import absolute_import, division, print_function, unicode_literals

import os
import numpy as np
from torch.utils.data import Dataset
import torch
import time
import math
from tqdm import tqdm
import argparse

multi_hot_sizes=[3,2,1,2,6,1,1,1,1,7,3,8,1,6,9,5,1,1,1,12,100,27,10,3,1,1]

class DataLoader:
    """
    DataLoader dedicated for the Criteo Terabyte Click Logs dataset
    """

    def __init__(
            self,
            data_filename,
            data_directory,
            days,
            batch_size,
            max_ind_range=-1,
            split="train",
            drop_last_batch=False
    ):
        self.data_filename = data_filename
        self.data_directory = data_directory
        self.days = days
        self.batch_size = batch_size
        self.max_ind_range = max_ind_range

        total_file = os.path.join(
            data_directory,
            data_filename + "_day_count_v2.npz"
        )
        
        with np.load(total_file) as data:
            total_per_file = data["total_per_file"][np.array(days)]

        self.length = sum(total_per_file)
        if split == "test" or split == "val":
            self.length = int(np.ceil(self.length / 2.))
        self.split = split
        self.drop_last_batch = drop_last_batch

    def __iter__(self):
        return iter(
            _batch_generator(
                self.data_filename, self.data_directory, self.days,
                self.batch_size, self.split, self.drop_last_batch, self.max_ind_range
            )
        )

    def __len__(self):
        if self.drop_last_batch:
            return self.length // self.batch_size
        else:
            return math.ceil(self.length / self.batch_size)

def _transform_features(
        x_int_batch, x_cat_batch, y_batch, max_ind_range, flag_input_torch_tensor=True
):

    if max_ind_range > 0:
        x_cat_batch = x_cat_batch % max_ind_range

    sparse = []
    select_index = []
    for i in range(26+1):
        select_index.append(sum(multi_hot_sizes[:i]))
   
    if flag_input_torch_tensor:
        dense = x_int_batch.type(torch.float32).clone().detach()
        labels = y_batch.type(torch.int32).clone().detach()
        # sparse = x_cat_batch.detach().clone().type(torch.int32)
        for i in range(26):
            sparse.append(x_cat_batch[:,select_index[i]:select_index[i+1]].type(torch.int32).clone().flatten().detach())
    else:
        dense = torch.tensor(x_int_batch, dtype=torch.float32)
        sparse = torch.tensor(x_cat_batch, dtype=torch.int32)
        labels = torch.tensor(y_batch, dtype=torch.int32)

    return dense, tuple(sparse), labels.view(-1, 1)


def _batch_generator(
        data_filename, data_directory, days, batch_size, split, drop_last, max_ind_range
):
    previous_file = None
    for day in days:
        filepath = os.path.join(
            data_directory,
            data_filename + "_{}_sparse_multi_hot.npz".format(day)
        )

        # print('Loading file: ', filepath)
        with np.load(filepath) as data:
            x_int = data["X_int"]
            x_cat = data["X_cat"]
            y = data["y"]

        samples_in_file = y.shape[0]
        batch_start_idx = 0
        if split == "test" or split == "val":
            length = int(np.ceil(samples_in_file / 2.))
            if split == "test":
                samples_in_file = length
            elif split == "val":
                batch_start_idx = samples_in_file - length

        while batch_start_idx < samples_in_file - batch_size:

            missing_samples = batch_size
            if previous_file is not None:
                missing_samples -= previous_file['y'].shape[0]

            current_slice = slice(batch_start_idx, batch_start_idx + missing_samples)

            x_int_batch = x_int[current_slice]
            x_cat_batch = x_cat[current_slice]
            y_batch = y[current_slice]

            if previous_file is not None:
                x_int_batch = np.concatenate(
                    [previous_file['x_int'], x_int_batch],
                    axis=0
                )
                x_cat_batch = np.concatenate(
                    [previous_file['x_cat'], x_cat_batch],
                    axis=0
                )
                y_batch = np.concatenate([previous_file['y'], y_batch], axis=0)
                previous_file = None

            if x_int_batch.shape[0] != batch_size:
                raise ValueError('should not happen')

            yield _transform_features(x_int_batch, x_cat_batch, y_batch, max_ind_range)

            batch_start_idx += missing_samples
        if batch_start_idx != samples_in_file:
            current_slice = slice(batch_start_idx, samples_in_file)
            if previous_file is not None:
                previous_file = {
                    'x_int' : np.concatenate(
                        [previous_file['x_int'], x_int[current_slice]],
                        axis=0
                    ),
                    'x_cat' : np.concatenate(
                        [previous_file['x_cat'], x_cat[current_slice]],
                        axis=0
                    ),
                    'y' : np.concatenate([previous_file['y'], y[current_slice]], axis=0)
                }
            else:
                previous_file = {
                    'x_int' : x_int[current_slice],
                    'x_cat' : x_cat[current_slice],
                    'y' : y[current_slice]
                }

    if not drop_last:
        yield _transform_features(
            previous_file['x_int'],
            previous_file['x_cat'],
            previous_file['y'],
            max_ind_range
        )


def _test():
    generator = _batch_generator(
        data_filename='day',
        data_directory='./input',
        days=range(23),
        split="train",
        batch_size=2048,
        drop_last=True,
        max_ind_range=-1
    )
    t1 = time.time()
    for x_int, lS_o, x_cat, y in generator:
        t2 = time.time()
        time_diff = t2 - t1
        t1 = t2
        print(
            "time {} x_int.shape: {} lS_o.shape: {} x_cat.shape: {} y.shape: {}".format(
                time_diff, x_int.shape, lS_o.shape, x_cat.shape, y.shape
            )
        )


class CriteoBinDataset(Dataset):
    """Binary version of criteo dataset."""

    def __init__(self, data_file,
                 batch_size=1, max_ind_range=-1, runner=False, bytes_per_feature=4):
        # dataset
        self.tar_fea = 1   # single target
        self.den_fea = 13  # 13 dense  features
        self.spa_fea = sum(multi_hot_sizes)  # 26 sparse features
        self.tad_fea = self.tar_fea + self.den_fea
        self.tot_fea = self.tad_fea + self.spa_fea

        self.batch_size = batch_size
        self.max_ind_range = max_ind_range
        self.bytes_per_entry = (bytes_per_feature * self.tot_fea)

        self.num_entries = math.ceil((os.path.getsize(data_file + "/terabyte_processed_test_v2_label_sparse.bin") + os.path.getsize(data_file + "/terabyte_processed_test_v2_dense.bin")) / self.bytes_per_entry)

        self.arr_label_sparse = np.memmap(data_file + "/terabyte_processed_test_v2_label_sparse.bin", np.int32, 'r', shape=(self.num_entries, self.spa_fea + 1))
        self.arr_dense = np.memmap(data_file + "/terabyte_processed_test_v2_dense.bin", np.float32, 'r', shape=(self.num_entries, self.den_fea))

        print('data file:', data_file, 'number of batches:', self.num_entries)
        # self.file = open(data_file, 'rb')

    def __len__(self):
        return self.num_entries

    def __getitem__(self, idx):
        self.file.seek(idx * self.bytes_per_entry, 0)
        raw_data = self.file.read(self.bytes_per_entry)
        array = np.frombuffer(raw_data, dtype=np.int32)
        tensor = torch.from_numpy(array).view((-1, self.tot_fea))

        return _transform_features(x_int_batch=tensor[:, 1:14],
                                   x_cat_batch=tensor[:, 14:],
                                   y_batch=tensor[:, 0],
                                   max_ind_range=self.max_ind_range,
                                   flag_input_torch_tensor=True)

    def read(self, idx, bs):
        dense = self.arr_dense[idx:idx+bs,:].copy()
        sparse = self.arr_label_sparse[idx:idx+bs,1:].copy()
        label = self.arr_label_sparse[idx:idx+bs,0:1].copy()
         
        return _transform_features(torch.from_numpy(dense).type(torch.float32),
                                   torch.from_numpy(sparse).type(torch.int32),
                                   torch.from_numpy(label).type(torch.int32),
                                   max_ind_range=self.max_ind_range,
                                   flag_input_torch_tensor=True)

    def __del__(self):
        # self.file.close()
        return


def numpy_to_binary(input_files, output_file_path, split='train'):
    output_file_path_label_sparse = output_file_path + "/terabyte_processed_test_v2_label_sparse.bin"
    output_file_path_dense = output_file_path + "/terabyte_processed_test_v2_dense.bin"
    
    with open(output_file_path_label_sparse, 'wb') as output_file:
        if split == 'train':
            for input_file in input_files:
                print('Processing file: ', input_file)
                np_data = np.load(input_file)
                np_data = np.concatenate([np_data['y'].reshape(-1, 1),
                                          np_data['X_int'],
                                          np_data['X_cat']], axis=1)
                np_data = np_data.astype(np.int32)

                output_file.write(np_data.tobytes())
        else:
            assert len(input_files) == 3
            np_sparse_data = np.load(input_files[0])
            # np_dense_data = np.load(input_files[1])
            np_labels_data = np.load(input_files[2])
            concat_array = np.concatenate([np_labels_data, np_sparse_data[str(0)]], axis=1, dtype=np.int32)

            print("concat sparse indices")
            for i in range(25):
                concat_array = np.concatenate([concat_array,
                                          np_sparse_data[str(i+1)]], axis=1, dtype=np.int32)
            np_data = concat_array.astype(np.int32)

            samples_in_file = np_data.shape[0]
            midpoint = int(np.ceil(samples_in_file / 2.))
            if split == "test":
                begin = 0
                end = midpoint
            elif split == "val":
                begin = midpoint
                end = samples_in_file
            else:
                raise ValueError('Unknown split value: ', split)
            
            print("writing output to file")
            output_file.write(np_data[begin:end].tobytes())
    
    with open(output_file_path_dense, 'wb') as output_file:
        assert len(input_files) == 3
        np_dense_data = np.load(input_files[1])
        samples_in_file = np_dense_data.shape[0]
        midpoint = int(np.ceil(samples_in_file / 2.))
        if split == "test":
            begin = 0
            end = midpoint
        elif split == "val":
            begin = midpoint
            end = samples_in_file
        else:
            raise ValueError('Unknown split value: ', split)
        np_data = np_dense_data.astype(np.float32)
        print("writing dense output to file")
        output_file.write(np_data[begin:end].tobytes())

def _preprocess(args):
    train_files = ['{}_{}_sparse_multi_hot.npz'.format(args.input_data_prefix, day) for
                   day in range(0, 23)]

    test_valid_file = args.input_data_prefix + '_23_sparse_multi_hot.npz'

    os.makedirs(args.output_directory, exist_ok=True)
    for split in ['train', 'val', 'test']:
        print('Running preprocessing for split =', split)

        output_file = os.path.join(args.output_directory,
                                   '{}_data_v2.bin'.format(split))

        input_files = train_files if split == 'train' else [test_valid_file]
        numpy_to_binary(input_files=input_files,
                        output_file_path=output_file,
                        split=split)


def _test_bin():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_directory', required=True)
    parser.add_argument('--input_data_prefix', required=True)
    parser.add_argument('--split', choices=['train', 'test', 'val'],
                        required=True)
    args = parser.parse_args()

    _preprocess(args)

    binary_data_file = os.path.join(args.output_directory,
                                    '{}_data_v2.bin'.format(args.split))

    counts_file = os.path.join(args.output_directory, 'day_fea_count_v2.npz')
    dataset_binary = CriteoBinDataset(data_file=binary_data_file,
                                            counts_file=counts_file,
                                            batch_size=2048,)
    from dlrm_data_pytorch import CriteoDataset
    from dlrm_data_pytorch import collate_wrapper_criteo_offset as collate_wrapper_criteo

    binary_loader = torch.utils.data.DataLoader(
        dataset_binary,
        batch_size=None,
        shuffle=False,
        num_workers=0,
        collate_fn=None,
        pin_memory=False,
        drop_last=False,
    )

    original_dataset = CriteoDataset(
        dataset='terabyte',
        max_ind_range=10 * 1000 * 1000,
        sub_sample_rate=1,
        randomize=True,
        split=args.split,
        raw_path=args.input_data_prefix,
        pro_data='dummy_string',
        memory_map=True
    )

    original_loader = torch.utils.data.DataLoader(
        original_dataset,
        batch_size=2048,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_wrapper_criteo,
        pin_memory=False,
        drop_last=False,
    )

    assert len(dataset_binary) == len(original_loader)
    for i, (old_batch, new_batch) in tqdm(enumerate(zip(original_loader,
                                                        binary_loader)),
                                          total=len(dataset_binary)):

        for j in range(len(new_batch)):
            if not np.array_equal(old_batch[j], new_batch[j]):
                raise ValueError('FAILED: Datasets not equal')
        if i > len(dataset_binary):
            break
    print('PASSED')


if __name__ == '__main__':
    _test()
    _test_bin()