import json
import zipfile

import pandas as pd

from faunavision.data.classes import CLASSES, RAW_TO_CLASS, map_category
from faunavision.data.lila import read_labels, read_md_best_boxes
from faunavision.data.splits import assign_splits, build_manifest, coverage


def test_class_mapping_is_consistent():
    assert map_category("lionmale") == "lion"
    assert map_category("zebraburchells") == "plains_zebra"
    assert map_category("empty") is None
    assert set(RAW_TO_CLASS.values()) == set(CLASSES)
    assert CLASSES["black_rhino"].priority and CLASSES["black_rhino"].cluster == "rhinos"


def _write_coco(tmp_path):
    coco = {
        "images": [
            {"id": "KRU/a.jpg", "location": "L1", "seq_id": "s1", "frame_num": 0},
            {"id": "KRU/b.jpg", "location": "L1", "seq_id": "s2", "frame_num": 0},
            {"id": "SER/c.jpg", "location": "L9", "seq_id": "s3", "frame_num": 0},
        ],
        "annotations": [
            {"image_id": "KRU/a.jpg", "category_id": 1},
            {"image_id": "KRU/b.jpg", "category_id": 1},
            {"image_id": "KRU/b.jpg", "category_id": 2},
            {"image_id": "SER/c.jpg", "category_id": 1},
        ],
        "categories": [{"id": 1, "name": "impala"}, {"id": 2, "name": "kudu"}],
    }
    path = tmp_path / "meta.zip"
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("meta.json", json.dumps(coco))
    return path


def test_read_labels_streams_zip_and_filters_projects(tmp_path):
    images = read_labels(_write_coco(tmp_path), projects={"KRU"})
    assert set(images) == {"KRU/a.jpg", "KRU/b.jpg"}
    assert images["KRU/b.jpg"]["labels"] == {"impala", "kudu"}
    assert images["KRU/a.jpg"]["location"] == "L1"


def test_read_md_best_boxes_picks_most_confident_animal(tmp_path):
    md = {"images": [
        {"file": "KRU/a.jpg", "detections": [
            {"category": "1", "conf": 0.6, "bbox": [0, 0, 0.1, 0.1]},
            {"category": "1", "conf": 0.9, "bbox": [0.5, 0.5, 0.2, 0.2]},
            {"category": "2", "conf": 0.8, "bbox": [0, 0, 1, 1]},
        ]},
        {"file": "KRU/b.jpg", "detections": []},
        {"file": "other.jpg", "detections": []},
    ]}
    path = tmp_path / "md.json"
    path.write_text(json.dumps(md))
    out = read_md_best_boxes(path, keep={"KRU/a.jpg", "KRU/b.jpg"})
    assert out["KRU/a.jpg"]["md_conf"] == 0.9
    assert out["KRU/a.jpg"]["bbox"] == (0.5, 0.5, 0.2, 0.2)
    assert out["KRU/a.jpg"]["md_n_animals"] == 2 and out["KRU/a.jpg"]["md_person"]
    assert out["KRU/b.jpg"]["md_conf"] == 0.0 and "other.jpg" not in out


def _index():
    rows = []
    for proj, locs in {"KRU": ["k1"], "MAD": [f"m{i}" for i in range(40)]}.items():
        for loc in locs:
            for seq in range(3):
                for frame, conf in enumerate([0.7, 0.95, 0.6]):
                    rows.append({
                        "image_id": f"{proj}/{loc}/{seq}/{frame}.jpg", "project": proj, "location": loc,
                        "seq_id": f"{proj}#{loc}#{seq}", "frame_num": frame, "n_labels": 1,
                        "cls": "impala" if seq < 2 else "kudu", "md_conf": conf,
                    })
    return pd.DataFrame(rows)


def test_splits_hold_out_reserves_and_whole_cameras():
    idx = _index()
    split = assign_splits(idx, test_projects={"KRU"})
    assert set(split[idx.project == "KRU"]) == {"test"}
    per_location = idx.assign(split=split).groupby("location")["split"].nunique()
    assert (per_location == 1).all()
    assert set(split) == {"train", "val", "test"}


def test_square_crop_box_is_square_padded_and_clipped():
    from faunavision.data.crops import square_crop_box

    l, t, r, b = square_crop_box(0.4, 0.4, 0.2, 0.1, 1000, 1000)
    assert (r - l) == (b - t) == 240  # longer side 200px + 2 x 10% padding
    l, t, r, b = square_crop_box(0.95, 0.0, 0.05, 0.05, 1000, 1000)
    assert l >= 0 and t == 0 and r <= 1000


def test_manifest_one_best_frame_per_sequence_and_caps():
    m = build_manifest(_index(), test_projects={"KRU"}, cap_train=10)
    assert m["seq_id"].is_unique
    assert (m["frame_num"] == 1).all()  # frame 1 has the highest md_conf
    assert (m[m.split == "train"].groupby("cls").size() <= 10).all()
    cov = coverage(m)
    assert list(cov.columns) == ["train", "val", "test"]
