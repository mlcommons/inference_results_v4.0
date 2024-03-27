| Model                                               | Scenario   |   Accuracy |    QPS | Latency (in ms)   | Power Efficiency (in samples/J)   |
|-----------------------------------------------------|------------|------------|--------|-------------------|-----------------------------------|
| bert-base-pruned90-none-bert-99                     | offline    |    88.4204 |  9.665 | -                 |                                   |
| mobilebert-14layer_pruned50-none-vnni-bert-99       | offline    |    90.4308 | 24.644 | -                 |                                   |
| mobilebert-base_quant-none-bert-99                  | offline    |    90.8127 | 14.253 | -                 |                                   |
| mobilebert-none-base-none-bert-99                   | offline    |    90.8907 | 14.004 | -                 |                                   |
| mobilebert-14layer_pruned50_quant-none-vnni-bert-99 | offline    |    90.3672 | 31.266 | -                 |                                   |
| bert-base-pruned95_obs_quant-none-bert-99           | offline    |    87.8857 | 19.09  | -                 |                                   |
| obert-base-pruned90-none-bert-99                    | offline    |    88.3108 |  9.509 | -                 |                                   |