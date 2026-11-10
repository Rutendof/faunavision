from faunavision.data.species_selection import (
    build_clusters,
    parse_species_counts,
    pick_balanced,
    with_lineage,
)


def _row(tid, genus_id, family_id, obs, cls="Aves"):
    return {
        "taxon_id": tid,
        "name": f"sp{tid}",
        "common_name": None,
        "iconic": cls,
        "observations": obs,
        "class_id": 1,
        "class": cls,
        "order_id": 2,
        "order": "O",
        "family_id": family_id,
        "family": f"fam{family_id}",
        "genus_id": genus_id,
        "genus": f"gen{genus_id}",
    }


def test_parse_keeps_only_species_above_threshold():
    results = [
        {"count": 500, "taxon": {"id": 1, "name": "A a", "rank": "species", "ancestor_ids": [9, 1]}},
        {"count": 50, "taxon": {"id": 2, "name": "A b", "rank": "species", "ancestor_ids": [9, 2]}},
        {"count": 900, "taxon": {"id": 3, "name": "A c d", "rank": "subspecies", "ancestor_ids": []}},
    ]
    out = parse_species_counts(results, min_observations=100)
    assert [s.taxon_id for s in out] == [1]


def test_with_lineage_resolves_ranks_from_ancestors():
    sp = parse_species_counts(
        [{"count": 500, "taxon": {"id": 49_001, "name": "Balearica regulorum", "rank": "species",
                                  "ancestor_ids": [1, 3, 23, 49, 49_001]}}],
        min_observations=1,
    )
    taxa = {
        3: {"id": 3, "rank": "class", "name": "Aves"},
        23: {"id": 23, "rank": "family", "name": "Gruidae"},
        49: {"id": 49, "rank": "genus", "name": "Balearica"},
    }
    row = with_lineage(sp, taxa)[0]
    assert (row["class"], row["family"], row["genus"]) == ("Aves", "Gruidae", "Balearica")
    assert row["order"] is None


def test_genus_cluster_preferred_and_species_not_reused():
    rows = [_row(i, genus_id=10, family_id=100, obs=1000 - i) for i in range(4)]
    clusters = build_clusters(rows, min_species=3, max_species=5)
    assert len(clusters) == 1
    assert clusters[0].level == "genus"
    assert len(clusters[0].members) == 4


def test_family_fallback_uses_leftover_genera():
    rows = [
        _row(1, genus_id=10, family_id=100, obs=500),
        _row(2, genus_id=11, family_id=100, obs=400),
        _row(3, genus_id=12, family_id=100, obs=300),
    ]
    clusters = build_clusters(rows, min_species=3)
    assert [c.level for c in clusters] == ["family"]


def test_max_species_keeps_most_observed_and_counts_rest():
    rows = [_row(i, genus_id=10, family_id=100, obs=100 * i) for i in range(1, 8)]
    c = build_clusters(rows, min_species=3, max_species=5)[0]
    assert [m["taxon_id"] for m in c.members] == [7, 6, 5, 4, 3]
    assert c.unused_relatives == 2
    assert c.min_observations == 300


def test_pick_balanced_prefers_genus_over_bigger_family_cluster():
    family_rows = [_row(i, genus_id=20 + i, family_id=1, obs=9000) for i in range(3)]
    genus_rows = [_row(10 + i, genus_id=30, family_id=2, obs=500) for i in range(3)]
    clusters = build_clusters(family_rows + genus_rows)
    assert clusters[0].level == "family"  # ranked first by observations...
    assert pick_balanced(clusters, n=1)[0].level == "genus"  # ...but genus is suggested


def test_pick_balanced_round_robins_classes():
    birds = [_row(i, genus_id=10 + i // 3, family_id=1, obs=5000 - i) for i in range(9)]
    mammals = [_row(100 + i, genus_id=50, family_id=2, obs=400, cls="Mammalia") for i in range(3)]
    clusters = build_clusters(birds + mammals)
    picked = pick_balanced(clusters, n=2)
    assert {c.taxon_class for c in picked} == {"Aves", "Mammalia"}
