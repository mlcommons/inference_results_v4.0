| Model     | Scenario   |   Accuracy |       QPS | Latency (in ms)   | Power Efficiency (in samples/J)   | TEST01   | TEST05   | TEST04   |
|-----------|------------|------------|-----------|-------------------|-----------------------------------|----------|----------|----------|
| resnet50  | offline    |     75.864 | 153133    | -                 |                                   | passed   | passed   | passed   |
| resnet50  | server     |     75.864 | 148979    | -                 |                                   | passed   | passed   | passed   |
| retinanet | offline    |     37.234 |   2493.86 | -                 |                                   | passed   | passed   |          |
| retinanet | server     |     37.234 |   2199.05 | -                 |                                   | passed   | passed   |          |