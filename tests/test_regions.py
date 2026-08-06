from pathlib import Path

from dogtracker_pc.regions import (
    add_manual_region,
    delete_manual_region,
    load_manual_regions,
    point_in_polygon,
)

SQUARE = [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)]


def test_point_in_polygon_inside():
    assert point_in_polygon(5, 5, SQUARE) is True


def test_point_in_polygon_outside():
    assert point_in_polygon(20, 20, SQUARE) is False


def test_point_in_polygon_degenerate_returns_false():
    assert point_in_polygon(1, 1, [(0, 0), (1, 1)]) is False


def test_add_and_load_roundtrip(tmp_path: Path):
    assert load_manual_regions(tmp_path) == []
    region = add_manual_region(tmp_path, SQUARE)
    assert region.points == tuple(SQUARE)

    loaded = load_manual_regions(tmp_path)
    assert len(loaded) == 1
    assert loaded[0].id == region.id
    assert loaded[0].points == tuple(SQUARE)


def test_add_multiple_regions_persist_independently(tmp_path: Path):
    a = add_manual_region(tmp_path, SQUARE)
    b = add_manual_region(tmp_path, [(1, 1), (2, 1), (2, 2)])
    loaded = load_manual_regions(tmp_path)
    assert {r.id for r in loaded} == {a.id, b.id}


def test_delete_manual_region(tmp_path: Path):
    region = add_manual_region(tmp_path, SQUARE)
    assert delete_manual_region(tmp_path, region.id) is True
    assert load_manual_regions(tmp_path) == []


def test_delete_missing_region_returns_false(tmp_path: Path):
    add_manual_region(tmp_path, SQUARE)
    assert delete_manual_region(tmp_path, "does-not-exist") is False


def test_load_ignores_corrupt_cache(tmp_path: Path):
    cache_dir = tmp_path / ".dogtracker_cache"
    cache_dir.mkdir()
    (cache_dir / "manual_regions.json").write_text("not json")
    assert load_manual_regions(tmp_path) == []
