| Model        | Scenario     | Accuracy                               |       QPS | Latency (in ms)   |   Power Efficiency (in samples/J) | TEST01   | TEST05   | TEST04   |
|--------------|--------------|----------------------------------------|-----------|-------------------|-----------------------------------|----------|----------|----------|
| resnet50     | offline      | 76.078                                 | 45267     | -                 |                            74.97  | passed   | passed   | passed   |
| resnet50     | singlestream | 76.078                                 |  3401.36  | 0.294             |                            10.761 | passed   | passed   | passed   |
| resnet50     | multistream  | 76.078                                 | 18018     | 0.444             |                            42.868 | passed   | passed   | passed   |
| 3d-unet-99.9 | offline      | 0.86236                                |     4.153 | -                 |                             0.007 | passed   | passed   |          |
| 3d-unet-99.9 | singlestream | 0.86236                                |     2.307 | 433.455           |                             0.008 | passed   | passed   |          |
| retinanet    | offline      | 37.354                                 |   871.749 | -                 |                             1.45  | passed   | passed   |          |
| retinanet    | singlestream | 37.318                                 |   589.275 | 1.697             |                             1.127 | passed   | passed   |          |
| retinanet    | multistream  | 37.326                                 |   714.222 | 11.201            |                             1.29  | passed   | passed   |          |
| gptj-99.9    | offline      | (43.04, 20.1313, 29.9733, 4022811.0)   |     4.662 | -                 |                             0.009 |          |          |          |
| gptj-99.9    | singlestream | (43.0133, 20.1143, 29.9807, 4030164.0) |     0.83  | 1204.262          |                             0.003 |          |          |          |
| rnnt         | offline      | 92.55451                               | 15225.6   | -                 |                            25.297 | passed   | passed   |          |
| rnnt         | singlestream | 92.56579                               |    99.651 | 10.035            |                             0.415 | passed   | passed   |          |
| bert-99      | offline      | 90.15484                               |  4102.44  | -                 |                             6.668 | passed   | passed   |          |
| bert-99      | singlestream | 90.26682                               |   983.284 | 1.017             |                             2.203 | passed   | passed   |          |