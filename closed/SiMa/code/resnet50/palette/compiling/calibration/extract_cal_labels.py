import numpy as np
with open('/project/davinci_users/software/victor.bittorf/mlperf/amy/mlperf_resnet50/calibration/cal_map.txt') as f:
    lines = f.read()
    l = list(map(lambda l: l.split('\t'), lines.split('\n')))
    labels = []
    for filename, label in l:
        labels.append(int(label))
    labels = np.array(labels).astype(np.int32)
    labels.tofile('/project/davinci_users/software/victor.bittorf/mlperf/amy/mlperf_resnet50/calibration/mlperf_resnet50_cal_labels_int32.dat')
    print(labels)