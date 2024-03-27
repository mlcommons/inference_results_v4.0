| Model                                               | Scenario     |   Accuracy |     QPS | Latency (in ms)   |   Power Efficiency (in samples/J) |
|-----------------------------------------------------|--------------|------------|---------|-------------------|-----------------------------------|
| obert-large-base-none-bert-99                       | offline      |    89.6519 |  12.078 | -                 |                             0.04  |
| obert-large-base-none-bert-99                       | singlestream |    89.6519 |   7.023 | 142.395           |                             0.033 |
| obert-base-pruned90-none-bert-99                    | offline      |    88.3108 |  97.675 | -                 |                             0.325 |
| obert-base-pruned90-none-bert-99                    | singlestream |    88.3108 |  56.351 | 17.746            |                             0.284 |
| obert-large-pruned97-quant-none-bert-99             | offline      |    90.1797 | 104.403 | -                 |                             0.354 |
| obert-large-pruned97-quant-none-bert-99             | singlestream |    90.1797 |  47.396 | 21.099            |                             0.248 |
| mobilebert-none-base-none-bert-99                   | offline      |    90.8907 | 112.774 | -                 |                             0.375 |
| mobilebert-none-base-none-bert-99                   | singlestream |    90.8907 |  56.472 | 17.708            |                             0.246 |
| bert-large-pruned80_quant-none-vnni-bert-99         | offline      |    90.2684 |  91.781 | -                 |                             0.288 |
| bert-large-pruned80_quant-none-vnni-bert-99         | singlestream |    90.2684 |  34.604 | 28.898            |                             0.186 |
| obert-large-pruned95-none-vnni-bert-99              | offline      |    90.1783 |  38.34  | -                 |                             0.133 |
| obert-large-pruned95-none-vnni-bert-99              | singlestream |    90.1783 |  21.925 | 45.61             |                             0.119 |
| bert-base-pruned90-none-bert-99                     | offline      |    88.4204 |  98.547 | -                 |                             0.329 |
| bert-base-pruned90-none-bert-99                     | singlestream |    88.4204 |  56.625 | 17.66             |                             0.285 |
| mobilebert-14layer_pruned50_quant-none-vnni-bert-99 | offline      |    90.4046 | 362.956 | -                 |                             1.35  |
| mobilebert-14layer_pruned50_quant-none-vnni-bert-99 | singlestream |    90.4046 | 183.251 | 5.457             |                             0.98  |
| mobilebert-14layer_pruned50-none-vnni-bert-99       | offline      |    90.4308 | 211.088 | -                 |                             0.732 |
| mobilebert-14layer_pruned50-none-vnni-bert-99       | singlestream |    90.4308 | 101.194 | 9.882             |                             0.49  |
| bert-large-base-none-bert-99                        | offline      |    89.6519 |  12.073 | -                 |                             0.04  |
| bert-large-base-none-bert-99                        | singlestream |    89.6519 |   7.014 | 142.566           |                             0.033 |
| obert-large-pruned95_quant-none-vnni-bert-99        | offline      |    90.0299 | 104.046 | -                 |                             0.363 |
| obert-large-pruned95_quant-none-vnni-bert-99        | singlestream |    90.0299 |  42.922 | 23.298            |                             0.255 |
| mobilebert-base_quant-none-bert-99                  | offline      |    90.7887 | 212.981 | -                 |                             0.77  |
| mobilebert-base_quant-none-bert-99                  | singlestream |    90.7959 | 104.384 | 9.58              |                             0.503 |
| obert-large-pruned97-none-bert-99                   | offline      |    90.1356 |  40.203 | -                 |                             0.142 |
| obert-large-pruned97-none-bert-99                   | singlestream |    90.1356 |  23.955 | 41.745            |                             0.133 |