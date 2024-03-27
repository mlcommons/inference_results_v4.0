| Model     | Scenario     |   Accuracy |      QPS | Latency (in ms)   |   Power Efficiency (in samples/J) | TEST01   | TEST05   | TEST04   |
|-----------|--------------|------------|----------|-------------------|-----------------------------------|----------|----------|----------|
| retinanet | offline      |    37.219  |  124.311 | -                 |                             4.617 | passed   | passed   |          |
| retinanet | singlestream |    37.239  |   52.427 | 19.074            |                             3.057 | passed   | passed   |          |
| retinanet | multistream  |    37.239  |   82.585 | 96.87             |                             4.056 | passed   | passed   |          |
| resnet50  | singlestream |    75.892  |  870.322 | 1.149             |                            71.622 | passed   | passed   | passed   |
| resnet50  | offline      |    75.892  | 6941.64  | -                 |                           250.482 | passed   | passed   | passed   |
| bert-99   | singlestream |    90.0716 |   79.879 | 12.519            |                             3.474 | passed   | passed   |          |
| bert-99   | offline      |    90.0716 |  254.362 | -                 |                             9.706 | passed   | passed   |          |