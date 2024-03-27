| Model     | Scenario     |   Accuracy |        QPS | Latency (in ms)   | Power Efficiency (in samples/J)   |
|-----------|--------------|------------|------------|-------------------|-----------------------------------|
| resnet50  | singlestream |     75.864 |   2450.98  | 0.408             |                                   |
| resnet50  | multistream  |     75.864 |  14869.9   | 0.538             |                                   |
| resnet50  | server       |     75.864 | 151979     | -                 |                                   |
| resnet50  | offline      |     75.864 | 157977     | -                 |                                   |
| retinanet | server       |     37.234 |   1899.71  | -                 |                                   |
| retinanet | offline      |     37.234 |   2272.64  | -                 |                                   |
| retinanet | singlestream |     37.234 |    101.616 | 9.841             |                                   |
| retinanet | multistream  |     37.234 |    203.035 | 39.402            |                                   |