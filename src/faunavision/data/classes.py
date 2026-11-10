"""FaunaVision class list for camera-trap triage.

Maps raw Snapshot Safari category names to our classes. Each class belongs
to a confusion cluster (a group of lookalikes we report on together) and may
be a *priority* class that a ranger should be alerted about.

Design choices:
- Lookalike pairs that matter for conservation get their own cluster
  (white vs black rhino, plains vs mountain zebra, ...).
- Sex or age labels are merged into the species (lionfemale -> lion).
- Groups with too few images to learn species are merged to a coarser class
  (all mongooses -> mongoose) rather than dropped.
- Domestic animals are kept as one class: livestock in a reserve is a
  real triage signal.
- People and vehicles are not in the labels (removed for privacy). The
  detector (MegaDetector) handles those.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ClassDef:
    name: str
    cluster: str
    priority: bool = False


# fmt: off
_CLASSES: dict[str, tuple[str, str, bool, list[str]]] = {
    # class:            (cluster,            common name,               priority, raw LILA categories)
    "white_rhino":      ("rhinos",           "White rhinoceros",        True,  ["rhinoceroswhite"]),
    "black_rhino":      ("rhinos",           "Black rhinoceros",        True,  ["rhinocerosblack"]),
    "plains_zebra":     ("zebras",           "Plains zebra",            False, ["zebraplains", "zebraburchells"]),
    "mountain_zebra":   ("zebras",           "Mountain zebra",          False, ["zebramountain"]),
    "blue_wildebeest":  ("wildebeest",       "Blue wildebeest",         False, ["wildebeestblue"]),
    "black_wildebeest": ("wildebeest",       "Black wildebeest",        False, ["wildebeestblack"]),
    "spotted_hyena":    ("hyaenids",         "Spotted hyena",           True,  ["hyenaspotted"]),
    "brown_hyena":      ("hyaenids",         "Brown hyena",             True,  ["hyenabrown"]),
    "aardwolf":         ("hyaenids",         "Aardwolf",                False, ["aardwolf"]),
    "bontebok":         ("damaliscus",       "Bontebok",                False, ["bontebok"]),
    "blesbok":          ("damaliscus",       "Blesbok",                 False, ["blesbok"]),
    "tsessebe":         ("damaliscus",       "Tsessebe",                False, ["tsessebe"]),
    "red_hartebeest":   ("damaliscus",       "Red hartebeest",          False, ["hartebeestred", "hartebeest"]),
    "kudu":             ("spiral_horned",    "Greater kudu",            False, ["kudu"]),
    "eland":            ("spiral_horned",    "Common eland",            False, ["eland"]),
    "nyala":            ("spiral_horned",    "Nyala",                   False, ["nyala"]),
    "bushbuck":         ("spiral_horned",    "Bushbuck",                False, ["bushbuck"]),
    "common_duiker":    ("small_antelope",   "Common duiker",           False, ["duikercommongrey"]),
    "steenbok":         ("small_antelope",   "Steenbok",                False, ["steenbok"]),
    "grysbok":          ("small_antelope",   "Cape grysbok",            False, ["grysbok"]),
    "klipspringer":     ("small_antelope",   "Klipspringer",            False, ["klipspringer"]),
    "grey_rhebok":      ("small_antelope",   "Grey rhebok",             False, ["rhebokgrey"]),
    "oribi":            ("small_antelope",   "Oribi",                   False, ["oribi"]),
    "reedbuck":         ("kobus_reedbucks",  "Southern reedbuck",       False, ["reedbuck"]),
    "mountain_reedbuck":("kobus_reedbucks",  "Mountain reedbuck",       False, ["reedbuckmountain"]),
    "waterbuck":        ("kobus_reedbucks",  "Waterbuck",               False, ["waterbuck"]),
    "sable":            ("hippotragus",      "Sable antelope",          False, ["sable"]),
    "roan":             ("hippotragus",      "Roan antelope",           False, ["roan"]),
    "gemsbok":          ("hippotragus",      "Gemsbok",                 False, ["oryx"]),
    "lion":             ("large_cats",       "Lion",                    True,  ["lionfemale", "lionmale", "lioncub"]),
    "leopard":          ("large_cats",       "Leopard",                 True,  ["leopard"]),
    "cheetah":          ("large_cats",       "Cheetah",                 True,  ["cheetah"]),
    "caracal":          ("small_cats",       "Caracal",                 False, ["caracal"]),
    "african_wildcat":  ("small_cats",       "African wildcat",         False, ["africanwildcat"]),
    "serval":           ("small_cats",       "Serval",                  False, ["serval"]),
    "black_backed_jackal": ("canids",        "Black-backed jackal",     False, ["jackalblackbacked"]),
    "bat_eared_fox":    ("canids",           "Bat-eared fox",           False, ["foxbateared"]),
    "cape_fox":         ("canids",           "Cape fox",                False, ["foxcape"]),
    "wild_dog":         ("canids",           "African wild dog",        True,  ["wilddog"]),
    "genet":            ("small_carnivores", "Genet",                   False, ["genetcommonsmallspotted", "genetcapelargespotted"]),
    "civet":            ("small_carnivores", "African civet",           False, ["civet"]),
    "honey_badger":     ("small_carnivores", "Honey badger",            False, ["honeybadger"]),
    "mongoose":         ("small_carnivores", "Mongoose (any)",          False, ["mongoose", "mongooseunknown", "mongooseyellow", "mongoosebanded",
                                                                                "mongoosesmallcapegrey", "mongooseegyptianlargegrey", "mongooseslender",
                                                                                "mongoosewhitetailed", "mongoosedwarf", "mongoosecapesmallgrey",
                                                                                "mongoosewatermarsh", "mongoosewater"]),
    "meerkat":          ("small_carnivores", "Meerkat",                 False, ["meerkat"]),
    "elephant":         ("megaherbivores",   "African elephant",        True,  ["elephant"]),
    "hippo":            ("megaherbivores",   "Hippopotamus",            False, ["hippopotamus"]),
    "buffalo":          ("bovids_other",     "African buffalo",         False, ["buffalo"]),
    "giraffe":          ("bovids_other",     "Giraffe",                 False, ["giraffe"]),
    "impala":           ("gazelle_like",     "Impala",                  False, ["impala"]),
    "springbok":        ("gazelle_like",     "Springbok",               False, ["springbok"]),
    "warthog":          ("suids",            "Warthog",                 False, ["warthog"]),
    "bushpig":          ("suids",            "Bushpig",                 False, ["bushpig"]),
    "baboon":           ("primates",         "Chacma baboon",           False, ["baboon"]),
    "vervet_monkey":    ("primates",         "Vervet monkey",           False, ["monkeyvervet"]),
    "porcupine":        ("small_mammals",    "Cape porcupine",          False, ["porcupine"]),
    "aardvark":         ("small_mammals",    "Aardvark",                False, ["aardvark"]),
    "hare":             ("small_mammals",    "Hare (any)",              False, ["hareunknown", "harecape", "harescrub"]),
    "springhare":       ("small_mammals",    "Springhare",              False, ["springhare"]),
    "ostrich":          ("birds",            "Common ostrich",          False, ["ostrich"]),
    "kori_bustard":     ("birds",            "Kori bustard",            False, ["bustardkori"]),
    "secretarybird":    ("birds",            "Secretarybird",           False, ["secretarybird"]),
    "blue_crane":       ("birds",            "Blue crane",              False, ["craneblue"]),
    "helmeted_guineafowl": ("birds",         "Helmeted guineafowl",     False, ["guineafowl"]),
    "tortoise":         ("reptiles",         "Tortoise",                False, ["tortoise"]),
    "domestic_animal":  ("domestic",         "Livestock / domestic",    True,  ["domesticanimal", "cattle", "horsedomestic", "goatdomestic", "dogdomestic"]),
}
# fmt: on

EMPTY_CATEGORY = "empty"

CLASSES: dict[str, ClassDef] = {
    name: ClassDef(name=name, cluster=cluster, priority=priority)
    for name, (cluster, _common, priority, _raw) in _CLASSES.items()
}
COMMON_NAMES: dict[str, str] = {name: v[1] for name, v in _CLASSES.items()}
RAW_TO_CLASS: dict[str, str] = {
    raw: name for name, v in _CLASSES.items() for raw in v[3]
}


def map_category(raw: str) -> str | None:
    """Our class for a raw LILA category, or None if it is not used."""
    return RAW_TO_CLASS.get(raw)
