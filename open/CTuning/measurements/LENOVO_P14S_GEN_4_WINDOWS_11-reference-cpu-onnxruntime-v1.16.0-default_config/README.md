| Model    | Scenario     |   Accuracy |    QPS | Latency (in ms)   | Power Efficiency (in samples/J)   |
|----------|--------------|------------|--------|-------------------|-----------------------------------|
| resnet50 | singlestream |     76.456 | 36.926 | 27.081            |                                   |
| resnet50 | multistream  |     76.456 | 30     | 266.664           |                                   |
| resnet50 | offline      |     76.456 | 35.49  | -                 |                                   |