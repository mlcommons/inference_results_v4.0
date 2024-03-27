| Model        | Scenario     | Accuracy                               |       QPS | Latency (in ms)   | Power Efficiency (in samples/J)   | TEST01   | TEST05   | TEST04   |
|--------------|--------------|----------------------------------------|-----------|-------------------|-----------------------------------|----------|----------|----------|
| gptj-99.9    | server       | (43.0003, 20.1351, 29.9644, 4005160.0) |     7.696 | -                 |                                   |          |          |          |
| gptj-99.9    | offline      | (43.04, 20.1313, 29.9733, 4022811.0)   |     9.269 | -                 |                                   |          |          |          |
| gptj-99.9    | singlestream | (43.0133, 20.1143, 29.9807, 4030164.0) |     0.83  | 1204.832          |                                   |          |          |          |
| retinanet    | server       | 37.357                                 |  1598.62  | -                 |                                   | passed   | passed   |          |
| retinanet    | offline      | 37.313                                 |  1719.03  | -                 |                                   | passed   | passed   |          |
| retinanet    | multistream  | 37.312                                 |  1251.37  | 6.393             |                                   | passed   | passed   |          |
| retinanet    | singlestream | 37.309                                 |   429.738 | 2.327             |                                   | passed   | passed   |          |
| bert-99.9    | server       | 90.89075                               |  2932.14  | -                 |                                   | passed   | passed   |          |
| bert-99.9    | offline      | 90.8917                                |  3269.47  | -                 |                                   | passed   | passed   |          |
| bert-99      | server       | 90.25898                               |  7693.92  | -                 |                                   | passed   | passed   |          |
| bert-99      | offline      | 90.15484                               |  8212.08  | -                 |                                   | passed   | passed   |          |
| bert-99      | singlestream | 90.26682                               |   963.391 | 1.038             |                                   | passed   | passed   |          |
| resnet50     | server       | 76.138                                 | 74728.4   | -                 |                                   | passed   | passed   | passed   |
| resnet50     | offline      | 76.078                                 | 89133.6   | -                 |                                   | passed   | passed   | passed   |
| resnet50     | multistream  | 76.078                                 | 20050.1   | 0.399             |                                   | passed   | passed   | passed   |
| resnet50     | singlestream | 76.078                                 |  2906.98  | 0.344             |                                   | passed   | passed   | passed   |
| gptj-99      | server       | (43.0003, 20.1351, 29.9644, 4005160.0) |     7.696 | -                 |                                   |          |          |          |
| gptj-99      | offline      | (43.04, 20.1313, 29.9733, 4022811.0)   |     9.257 | -                 |                                   |          |          |          |
| gptj-99      | singlestream | (43.0133, 20.1143, 29.9807, 4030164.0) |     0.83  | 1204.944          |                                   |          |          |          |
| 3d-unet-99.9 | offline      | 0.86236                                |     8.245 | -                 |                                   | passed   | passed   |          |
| 3d-unet-99.9 | singlestream | 0.86236                                |     2.294 | 436.009           |                                   | passed   | passed   |          |
| rnnt         | server       | 92.55677                               | 26994.4   | -                 |                                   | passed   | passed   |          |
| rnnt         | offline      | 92.55226                               | 30285.4   | -                 |                                   | passed   | passed   |          |
| rnnt         | singlestream | 92.56128                               |    96.674 | 10.344            |                                   | passed   | passed   |          |