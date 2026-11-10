# Triage with MegaDetector on unseen reserves

Reserves: CDB, KRU, PLN. 129,152 sequences (82,872 empty, 41,094 with a FaunaVision animal, 5,574 with a priority species). A sequence is kept if any frame has an animal box at or above the threshold.

| Threshold | Empty removed | Animals kept | Priority kept | Frames left to review |
|---|---|---|---|---|
| 0.05 | 89.8% | 99.9% | 99.8% | 53.0% |
| 0.1 | 89.8% | 99.9% | 99.8% | 53.0% |
| 0.2 | 94.7% | 99.9% | 99.7% | 49.0% |
| 0.3 | 96.7% | 99.8% | 99.5% | 47.4% |
| 0.5 | 98.5% | 99.5% | 98.7% | 45.9% |
| 0.7 | 99.3% | 98.1% | 95.2% | 44.6% |

## Priority species kept at threshold 0.2

| Species | Sequences | Kept |
|---|---|---|
| White rhinoceros | 983 | 99.3% |
| Lion | 223 | 99.6% |
| Black rhinoceros | 308 | 99.7% |
| African elephant | 2985 | 99.7% |
| Livestock / domestic | 171 | 100.0% |
| Cheetah | 15 | 100.0% |
| Leopard | 83 | 100.0% |
| Brown hyena | 400 | 100.0% |
| Spotted hyena | 395 | 100.0% |
| African wild dog | 11 | 100.0% |

Caveat: MegaDetector was partly trained on Snapshot Safari data, so these numbers are likely optimistic.
