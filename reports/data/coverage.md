# Camera-trap dataset coverage

- Test reserves (never seen in training): CDB, KRU, PLN
- One frame per sequence, MegaDetector confidence >= 0.5
- Caps: 4000 train / 1500 val and test images per class
- Totals: 154,502 images; train 102,523, test 27,020, val 24,959

| Cluster | Class | Priority | Train | Val | Test | Note |
|---|---|---|---|---|---|---|
| birds | Common ostrich |  | 3915 | 401 | 1342 |  |
| birds | Kori bustard |  | 1472 | 75 | 23 |  |
| birds | Helmeted guineafowl |  | 431 | 29 | 0 | absent from test reserves |
| birds | Blue crane |  | 106 | 49 | 4 |  |
| birds | Secretarybird |  | 65 | 5 | 6 | too few to train: needs iNaturalist top-up |
| bovids_other | African buffalo |  | 4000 | 626 | 1073 |  |
| bovids_other | Giraffe |  | 4000 | 1500 | 1500 |  |
| canids | Black-backed jackal |  | 2145 | 733 | 588 |  |
| canids | Bat-eared fox |  | 751 | 138 | 137 |  |
| canids | Cape fox |  | 295 | 32 | 1 |  |
| canids | African wild dog | yes | 58 | 15 | 11 | too few to train: needs iNaturalist top-up |
| damaliscus | Bontebok |  | 4000 | 850 | 0 | absent from test reserves |
| damaliscus | Red hartebeest |  | 3866 | 545 | 1075 |  |
| damaliscus | Blesbok |  | 180 | 1 | 962 |  |
| damaliscus | Tsessebe |  | 17 | 40 | 71 | too few to train: needs iNaturalist top-up |
| domestic | Livestock / domestic | yes | 4000 | 1188 | 169 |  |
| gazelle_like | Impala |  | 4000 | 1500 | 1500 |  |
| gazelle_like | Springbok |  | 4000 | 774 | 1500 |  |
| hippotragus | Gemsbok |  | 4000 | 1500 | 868 |  |
| hippotragus | Sable antelope |  | 927 | 69 | 3 |  |
| hippotragus | Roan antelope |  | 55 | 5 | 1 | too few to train: needs iNaturalist top-up |
| hyaenids | Spotted hyena | yes | 2242 | 505 | 390 |  |
| hyaenids | Brown hyena | yes | 889 | 128 | 393 |  |
| hyaenids | Aardwolf |  | 222 | 124 | 66 |  |
| kobus_reedbucks | Waterbuck |  | 651 | 128 | 454 |  |
| kobus_reedbucks | Southern reedbuck |  | 272 | 46 | 29 |  |
| kobus_reedbucks | Mountain reedbuck |  | 132 | 10 | 140 |  |
| large_cats | Lion | yes | 832 | 161 | 222 |  |
| large_cats | Leopard | yes | 650 | 74 | 83 |  |
| large_cats | Cheetah | yes | 52 | 7 | 14 | too few to train: needs iNaturalist top-up |
| megaherbivores | African elephant | yes | 4000 | 1500 | 1500 |  |
| megaherbivores | Hippopotamus |  | 841 | 86 | 409 |  |
| primates | Chacma baboon |  | 4000 | 1500 | 496 |  |
| primates | Vervet monkey |  | 3115 | 412 | 865 |  |
| reptiles | Tortoise |  | 141 | 65 | 10 |  |
| rhinos | White rhinoceros | yes | 1317 | 90 | 969 |  |
| rhinos | Black rhinoceros | yes | 868 | 63 | 307 |  |
| small_antelope | Common duiker |  | 4000 | 1082 | 1003 |  |
| small_antelope | Steenbok |  | 4000 | 651 | 317 |  |
| small_antelope | Grey rhebok |  | 445 | 68 | 20 |  |
| small_antelope | Klipspringer |  | 110 | 9 | 23 |  |
| small_antelope | Cape grysbok |  | 95 | 23 | 0 | too few to train: needs iNaturalist top-up; absent from test reserves |
| small_antelope | Oribi |  | 19 | 9 | 21 | too few to train: needs iNaturalist top-up |
| small_carnivores | Mongoose (any) |  | 449 | 61 | 38 |  |
| small_carnivores | Genet |  | 265 | 25 | 74 |  |
| small_carnivores | African civet |  | 112 | 37 | 53 |  |
| small_carnivores | Honey badger |  | 84 | 24 | 18 | too few to train: needs iNaturalist top-up |
| small_carnivores | Meerkat |  | 45 | 5 | 21 | too few to train: needs iNaturalist top-up |
| small_cats | African wildcat |  | 420 | 21 | 13 |  |
| small_cats | Caracal |  | 126 | 13 | 88 |  |
| small_cats | Serval |  | 47 | 7 | 64 | too few to train: needs iNaturalist top-up |
| small_mammals | Hare (any) |  | 1281 | 130 | 400 |  |
| small_mammals | Cape porcupine |  | 1062 | 195 | 133 |  |
| small_mammals | Aardvark |  | 470 | 78 | 70 |  |
| small_mammals | Springhare |  | 143 | 15 | 60 |  |
| spiral_horned | Common eland |  | 4000 | 952 | 964 |  |
| spiral_horned | Greater kudu |  | 4000 | 1500 | 1500 |  |
| spiral_horned | Nyala |  | 696 | 113 | 109 |  |
| spiral_horned | Bushbuck |  | 329 | 35 | 10 |  |
| suids | Warthog |  | 4000 | 757 | 1165 |  |
| suids | Bushpig |  | 555 | 102 | 21 |  |
| wildebeest | Blue wildebeest |  | 4000 | 1051 | 1027 |  |
| wildebeest | Black wildebeest |  | 1263 | 201 | 294 |  |
| zebras | Mountain zebra |  | 4000 | 1321 | 833 |  |
| zebras | Plains zebra |  | 4000 | 1500 | 1500 |  |
