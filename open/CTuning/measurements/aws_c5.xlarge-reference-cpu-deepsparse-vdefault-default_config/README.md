| Model                                               | Scenario   |   Accuracy |    QPS | Latency (in ms)   | Power Efficiency (in samples/J)   |
|-----------------------------------------------------|------------|------------|--------|-------------------|-----------------------------------|
| bert-base-pruned90-none-bert-99                     | offline    |    88.4204 | 13.1   | -                 |                                   |
| mobilebert-14layer_pruned50-none-vnni-bert-99       | offline    |    90.4308 | 32.988 | -                 |                                   |
| mobilebert-base_quant-none-bert-99                  | offline    |    90.7887 | 17.06  | -                 |                                   |
| mobilebert-none-base-none-bert-99                   | offline    |    90.8907 | 16.984 | -                 |                                   |
| mobilebert-14layer_pruned50_quant-none-vnni-bert-99 | offline    |    90.4046 | 47.192 | -                 |                                   |
| bert-base-pruned95_obs_quant-none-bert-99           | offline    |    87.8921 | 23.114 | -                 |                                   |
| obert-base-pruned90-none-bert-99                    | offline    |    88.3108 | 12.957 | -                 |                                   |