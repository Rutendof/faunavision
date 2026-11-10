"""Turn iNaturalist species counts into candidate "confusion clusters".

A confusion cluster is a small group (3-5) of closely related species that
people genuinely mix up: same genus first, same family as a fallback. Training
and evaluating on clusters makes the fine-grained claim real instead of
relying on 40 unrelated, easy-to-separate species.

Everything here is pure (no network) so it can be unit tested.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

LINEAGE_RANKS = ("class", "order", "family", "genus")


@dataclass(frozen=True)
class Species:
    taxon_id: int
    name: str
    common_name: str | None
    iconic: str | None
    observations: int
    ancestor_ids: tuple[int, ...]


@dataclass
class Cluster:
    level: str  # "genus" or "family"
    taxon_id: int
    name: str
    taxon_class: str | None
    members: list[dict] = field(default_factory=list)
    unused_relatives: int = 0  # qualifying species left out by max_species

    @property
    def min_observations(self) -> int:
        return min(m["observations"] for m in self.members)

    @property
    def total_observations(self) -> int:
        return sum(m["observations"] for m in self.members)


def parse_species_counts(results: list[dict], min_observations: int) -> list[Species]:
    """Keep species-rank taxa with enough research-grade observations."""
    out = []
    for r in results:
        t = r["taxon"]
        if t.get("rank") != "species" or r["count"] < min_observations:
            continue
        out.append(
            Species(
                taxon_id=t["id"],
                name=t["name"],
                common_name=t.get("preferred_common_name"),
                iconic=t.get("iconic_taxon_name"),
                observations=r["count"],
                ancestor_ids=tuple(t.get("ancestor_ids", ())),
            )
        )
    return out


def with_lineage(species: list[Species], taxa_by_id: dict[int, dict]) -> list[dict]:
    """Attach class/order/family/genus (id and name) to each species."""
    rows = []
    for sp in species:
        row = {
            "taxon_id": sp.taxon_id,
            "name": sp.name,
            "common_name": sp.common_name,
            "iconic": sp.iconic,
            "observations": sp.observations,
        }
        for rank in LINEAGE_RANKS:
            row[f"{rank}_id"] = None
            row[rank] = None
        for anc in sp.ancestor_ids:
            t = taxa_by_id.get(anc)
            if t and t.get("rank") in LINEAGE_RANKS:
                row[f"{t['rank']}_id"] = t["id"]
                row[t["rank"]] = t["name"]
        rows.append(row)
    return rows


def build_clusters(
    rows: list[dict], min_species: int = 3, max_species: int = 5
) -> list[Cluster]:
    """Group species by genus, then leftovers by family.

    A species is used in at most one cluster. Within a cluster, the
    most-observed species are kept (up to ``max_species``).
    """
    clusters: list[Cluster] = []
    used: set[int] = set()

    for level in ("genus", "family"):
        groups: dict[int, list[dict]] = defaultdict(list)
        for row in rows:
            if row["taxon_id"] in used or row.get(f"{level}_id") is None:
                continue
            groups[row[f"{level}_id"]].append(row)

        for group_id, members in groups.items():
            if len(members) < min_species:
                continue
            members = sorted(members, key=lambda r: r["observations"], reverse=True)
            chosen = members[:max_species]
            clusters.append(
                Cluster(
                    level=level,
                    taxon_id=group_id,
                    name=chosen[0][level],
                    taxon_class=chosen[0]["class"],
                    members=chosen,
                    unused_relatives=len(members) - len(chosen),
                )
            )
            used.update(m["taxon_id"] for m in chosen)

    # Strongest clusters first: the rarest member decides how much data we have.
    clusters.sort(key=lambda c: (c.min_observations, c.total_observations), reverse=True)
    return clusters


def pick_balanced(clusters: list[Cluster], n: int) -> list[Cluster]:
    """Pick ``n`` clusters round-robin across classes (birds, mammals, ...).

    Stops one class (usually birds) from taking every slot. Within a class,
    genus clusters come before family clusters: congeners are the real
    lookalikes, while a family cluster (e.g. impala vs buffalo) often is not.
    Assumes ``clusters`` is already sorted best-first.
    """
    by_class: dict[str | None, list[Cluster]] = defaultdict(list)
    for c in sorted(clusters, key=lambda c: c.level != "genus"):  # stable sort
        by_class[c.taxon_class].append(c)
    queues = sorted(by_class.values(), key=lambda q: q[0].min_observations, reverse=True)

    picked: list[Cluster] = []
    while len(picked) < n and any(queues):
        for q in queues:
            if q and len(picked) < n:
                picked.append(q.pop(0))
    return picked
