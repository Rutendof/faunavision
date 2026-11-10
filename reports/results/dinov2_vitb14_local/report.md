# Results: dinov2_vitb14_local

Validation = held-out cameras in training reserves. Test = unseen reserves (Pilanesberg, Kruger, Camdeboo).

| Head | Split | Top-1 | Top-5 | Macro-F1 | Balanced acc | ECE | Near-miss share of errors |
|---|---|---|---|---|---|---|---|
| Linear | val | 83.9% | 96.1% | 81.7% | 81.6% | 2.5% | 41.7% |
| Linear | test | 73.2% | 91.1% | 72.0% | 72.6% | 3.7% | 33.6% |
| k-NN | val | 84.0% | 96.1% | 81.3% | 82.0% | 8.0% | 41.3% |
| k-NN | test | 70.5% | 88.0% | 68.5% | 68.6% | 5.1% | 39.3% |

Calibration: test ECE 4.5% before, 3.7% after temperature scaling (T = 0.77).

## Lookalike pairs (test, linear head)

Among images of either species assigned to one of the two, the share given the right one.

| Pair | Accuracy | n |
|---|---|---|
| White rhinoceros vs Black rhinoceros | 55.2% | 29 |
| Plains zebra vs Mountain zebra | 100.0% | 28 |
| Blue wildebeest vs Black wildebeest | 87.0% | 23 |
| Spotted hyena vs Brown hyena | 80.0% | 15 |
| Bontebok vs Blesbok | 100.0% | 14 |
| Greater kudu vs Nyala | 72.0% | 25 |
| Common duiker vs Steenbok | 100.0% | 15 |
| Black-backed jackal vs Cape fox | 93.3% | 15 |

## Per-class recall (test, linear head)

| Class | Recall | F1 | Test images |
|---|---|---|---|
| Roan antelope | 0.0% | 0.0% | 1 |
| Grey rhebok | 13.3% | 19.0% | 15 |
| Klipspringer | 20.0% | 23.1% | 15 |
| Livestock / domestic | 26.7% | 34.8% | 15 |
| Southern reedbuck | 26.7% | 23.5% | 15 |
| Spotted hyena | 26.7% | 40.0% | 15 |
| Mountain reedbuck | 33.3% | 32.3% | 15 |
| Sable antelope | 33.3% | 50.0% | 3 |
| Tsessebe | 40.0% | 52.2% | 15 |
| Oribi | 46.7% | 46.7% | 15 |
| Steenbok | 46.7% | 51.9% | 15 |
| Bushbuck | 50.0% | 58.8% | 10 |
| Black rhinoceros | 53.3% | 48.5% | 15 |
| Brown hyena | 53.3% | 59.3% | 15 |
| Common duiker | 53.3% | 48.5% | 15 |
| Leopard | 53.3% | 66.7% | 15 |
| White rhinoceros | 53.3% | 42.1% | 15 |
| Impala | 60.0% | 69.2% | 15 |
| Greater kudu | 60.0% | 62.1% | 15 |
| Nyala | 60.0% | 60.0% | 15 |
| Cheetah | 64.3% | 58.1% | 14 |
| Black wildebeest | 66.7% | 66.7% | 15 |
| Blue wildebeest | 66.7% | 52.6% | 15 |
| Mongoose (any) | 66.7% | 71.4% | 15 |
| Bushpig | 73.3% | 59.5% | 15 |
| Hare (any) | 73.3% | 81.5% | 15 |
| Hippopotamus | 73.3% | 81.5% | 15 |
| Red hartebeest | 73.3% | 78.6% | 15 |
| Serval | 73.3% | 84.6% | 15 |
| African civet | 80.0% | 85.7% | 15 |
| Lion | 80.0% | 80.0% | 15 |
| Meerkat | 80.0% | 85.7% | 15 |
| African wildcat | 84.6% | 88.0% | 13 |
| Black-backed jackal | 86.7% | 72.2% | 15 |
| African buffalo | 86.7% | 83.9% | 15 |
| African elephant | 86.7% | 92.9% | 15 |
| Genet | 86.7% | 89.7% | 15 |
| Plains zebra | 86.7% | 92.9% | 15 |
| Tortoise | 90.0% | 94.7% | 10 |
| African wild dog | 90.9% | 95.2% | 11 |
| Aardvark | 93.3% | 70.0% | 15 |
| Aardwolf | 93.3% | 87.5% | 15 |
| Chacma baboon | 93.3% | 84.8% | 15 |
| Blesbok | 93.3% | 93.3% | 15 |
| Common eland | 93.3% | 90.3% | 15 |
| Gemsbok | 93.3% | 90.3% | 15 |
| Giraffe | 93.3% | 96.6% | 15 |
| Honey badger | 93.3% | 84.8% | 15 |
| Cape porcupine | 93.3% | 93.3% | 15 |
| Springbok | 93.3% | 93.3% | 15 |
| Warthog | 93.3% | 93.3% | 15 |
| Waterbuck | 93.3% | 87.5% | 15 |
| Bat-eared fox | 100.0% | 100.0% | 15 |
| Blue crane | 100.0% | 100.0% | 4 |
| Cape fox | 100.0% | 33.3% | 1 |
| Caracal | 100.0% | 96.8% | 15 |
| Kori bustard | 100.0% | 96.8% | 15 |
| Mountain zebra | 100.0% | 100.0% | 15 |
| Common ostrich | 100.0% | 100.0% | 15 |
| Secretarybird | 100.0% | 100.0% | 6 |
| Springhare | 100.0% | 93.8% | 15 |
| Vervet monkey | 100.0% | 96.8% | 15 |
