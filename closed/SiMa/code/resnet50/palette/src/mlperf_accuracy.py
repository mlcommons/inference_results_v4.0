#!/usr/bin/env python3

import os
import numpy as np
import argparse


# ./mla-rt-develop-914a0a7 /mnt/mlperf_sr/batch24/init.lm 
# ./mla-rt-develop-914a0a7 -I4d -v -d ./mla_driver.bin /mnt/mlperf_sr/batch24/run.lm --ifm ifm.0:/mnt/mlperf/batch24.dat -r /mnt/mlperf/ofm::rn_out.dat


def ex(cmd):
    print('*' * 80)
    print(cmd)
    print('*' * 80)
    os.system(cmd)



def setup_board():
    ex('''sshpass -p xxxx ssh board "echo xxxx | sudo -S bash -c \\"sed -iE 's/ALL=(ALL) ALL/ALL=(ALL) NOPASSWD: ALL/g' /etc/sudoers\\"" ''')


def remote_execute(cmd):
    ex('''sshpass -p xxxx ssh board sudo {}'''.format(cmd))


def load_output(output_file):
    return numpy.fromfile(output_file, dtype=np.int8)


def test_24():
    remote_execute('/mnt/mlperf/accuracy_debug/test_24.sh')
    outs = analyze_output()
    print(outs)


def test_8():
    remote_execute('/mnt/mlperf/accuracy_debug/test_8.sh')
    outs = analyze_output()
    print(outs)


def test_1():
    remote_execute('/mnt/mlperf/accuracy_debug/test_1.sh')
    outs = analyze_output()
    print(outs)



def analyze_output(output_file='/srv/nfs4/share/mlperf/accuracy_debug/out.dat'):
    data = np.fromfile(output_file, dtype=np.int8)
    bs = data.shape[0] // 1001
    return data.reshape(bs, 1001)


def main():
    parser = argparse.ArgumentParser(
                        prog = 'ProgramName',
                        description = 'What the program does',
                        epilog = 'Text at the bottom of help')
    parser.add_argument('lm_file')
    parser.add_argument('input_file')

    output_file = '/tmp/model_out.bin'
    args = parser.parse_args()
    execute_mla_rt(args.lm_file, args.input_file, output_file)

    print(load_output(output_file))


if __name__ == '__main__':
    setup_board()
    # remote_execute("echo 'ehllo'")
    test_8()
    anlayze_output('/srv/nfs4/share/mlperf/accuracy_debug/out.dat')
    # main()