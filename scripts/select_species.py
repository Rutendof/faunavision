"""Propose confusion clusters of lookalike species for FaunaVision.

Example:
    python scripts/select_species.py --place-id 113055 --min-observations 300

Writes to reports/species_selection/:
    candidates.csv   every qualifying species with its lineage
    clusters.csv     every proposed cluster, one row per member species
    report.md        readable summary plus a suggested balanced starting set

The output is a proposal. The final species list should be reviewed by hand:
check that the members really look alike and that photos are usable.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from faunavision.data.inat_api import InatClient  # noqa: E402
from faunavision.data.species_selection import (  # noqa: E402
    build_clusters,
    parse_species_counts,
    pick_balanced,
    with_lineage,
)

# Licences that allow non-commercial reuse with attribution. Images stay
# local and are never redistributed with the repo.
DEFAULT_LICENSES = ["cc0", "cc-by", "cc-by-nc", "cc-by-sa", "cc-by-nc-sa"]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--place-id", type=int, default=113055, help="iNaturalist place (113055 = Southern Africa)")
    p.add_argument("--iconic-taxa", default="Aves,Mammalia,Reptilia,Amphibia")
    p.add_argument("--min-observations", type=int, default=300)
    p.add_argument("--min-species", type=int, default=3)
    p.add_argument("--max-species", type=int, default=5)
    p.add_argument("--suggest", type=int, default=10, help="clusters in the suggested starting set")
    p.add_argument("--out", type=Path, default=ROOT / "reports" / "species_selection")
    p.add_argument("--cache", type=Path, default=ROOT / "data" / "cache" / "inat_api")
    args = p.parse_args()

    client = InatClient(args.cache)
    print("Fetching species counts ...")
    raw = client.species_counts(
        place_id=args.place_id,
        iconic_taxa=args.iconic_taxa.split(","),
        licenses=DEFAULT_LICENSES,
        min_observations=args.min_observations,
    )
    species = parse_species_counts(raw, args.min_observations)
    print(f"{len(species)} species with >= {args.min_observations} observations")

    ancestor_ids = {a for sp in species for a in sp.ancestor_ids}
    print(f"Resolving {len(ancestor_ids)} ancestor taxa ...")
    taxa = client.taxa(ancestor_ids)
    rows = with_lineage(species, taxa)

    clusters = build_clusters(rows, args.min_species, args.max_species)
    suggested = pick_balanced(clusters, args.suggest)
    suggested_ids = {c.taxon_id for c in suggested}

    args.out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).sort_values("observations", ascending=False).to_csv(
        args.out / "candidates.csv", index=False
    )
    cluster_rows = [
        {
            "cluster": c.name,
            "level": c.level,
            "class": c.taxon_class,
            "suggested": c.taxon_id in suggested_ids,
            **m,
        }
        for c in clusters
        for m in c.members
    ]
    pd.DataFrame(cluster_rows).to_csv(args.out / "clusters.csv", index=False)
    (args.out / "report.md").write_text(render_report(args, rows, clusters, suggested), encoding="utf-8")
    print(f"{len(clusters)} clusters proposed; report at {args.out / 'report.md'}")


def render_report(args, rows, clusters, suggested) -> str:
    n_species = sum(len(c.members) for c in suggested)
    lines = [
        "# Species selection report",
        "",
        f"- Place id: {args.place_id}; groups: {args.iconic_taxa}",
        f"- Research-grade, openly licensed observations; minimum {args.min_observations} per species",
        f"- {len(rows)} qualifying species, {len(clusters)} candidate clusters",
        "",
        "Counts are observations, not images. Each observation has one or more photos.",
        "",
        f"## Suggested starting set ({len(suggested)} clusters, {n_species} species)",
        "",
    ]
    lines += _cluster_tables(suggested)
    lines += ["## All candidate clusters", ""]
    lines.append("| Cluster | Level | Class | Species | Rarest member obs |")
    lines.append("|---|---|---|---|---|")
    for c in clusters:
        lines.append(
            f"| {c.name} | {c.level} | {c.taxon_class} | {len(c.members)} | {c.min_observations} |"
        )
    return "\n".join(lines) + "\n"


def _cluster_tables(clusters) -> list[str]:
    out = []
    for c in clusters:
        extra = f" (+{c.unused_relatives} more relatives available)" if c.unused_relatives else ""
        out += [f"### {c.name} ({c.level}, {c.taxon_class}){extra}", ""]
        out += ["| Species | Common name | Observations |", "|---|---|---|"]
        for m in c.members:
            link = f"[*{m['name']}*](https://www.inaturalist.org/taxa/{m['taxon_id']})"
            out.append(f"| {link} | {m['common_name'] or ''} | {m['observations']:,} |")
        out.append("")
    return out


if __name__ == "__main__":
    main()
