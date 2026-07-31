import numpy as np

from dogtracker_pc.analysis import build_heatmap, find_areas, segment_visits
from dogtracker_pc.detect import Detection


def _det(filename, ts, x, y):
    return Detection(filename, ts, 320, 240, x, y, 40, 30, 0.9)


def test_segment_visits_groups_nearby_slow_movement():
    dets = [
        _det("a.jpg", 0, 100, 100),
        _det("b.jpg", 500, 103, 101),
        _det("c.jpg", 1000, 106, 99),
    ]
    visits = segment_visits(dets, distance_threshold_px=40, time_gap_threshold_ms=3000)
    assert len(visits) == 1
    assert visits[0].frame_count == 3
    assert visits[0].start_ts == 0
    assert visits[0].end_ts == 1000


def test_segment_visits_splits_on_large_jump():
    dets = [
        _det("a.jpg", 0, 100, 100),
        _det("b.jpg", 500, 260, 200),  # far away -> new visit
    ]
    visits = segment_visits(dets, distance_threshold_px=40, time_gap_threshold_ms=3000)
    assert len(visits) == 2
    assert visits[0].frame_count == 1
    assert visits[1].centroid_x == 260


def test_segment_visits_splits_on_time_gap_even_if_close():
    dets = [
        _det("a.jpg", 0, 100, 100),
        _det("b.jpg", 10_000, 101, 100),  # same spot, but a big gap
    ]
    visits = segment_visits(dets, distance_threshold_px=40, time_gap_threshold_ms=3000)
    assert len(visits) == 2


def test_segment_visits_empty_input():
    assert segment_visits([]) == []


def test_segment_visits_single_detection():
    visits = segment_visits([_det("a.jpg", 0, 1, 2)])
    assert len(visits) == 1
    assert visits[0].duration_ms == 0
    assert visits[0].frame_count == 1


def test_build_heatmap_empty_is_fully_transparent():
    img = build_heatmap([], 32, 24)
    assert img.size == (32, 24)
    assert img.mode == "RGBA"
    alpha = np.array(img)[:, :, 3]
    assert alpha.max() == 0


def test_build_heatmap_peaks_at_densest_point():
    dets = [_det("a.jpg", i * 500, 10, 10) for i in range(5)] + [_det("b.jpg", 0, 25, 20)]
    img = build_heatmap(dets, 32, 24, blur_radius=0)
    alpha = np.array(img)[:, :, 3]
    # The 5-detection cluster at (10, 10) must be more opaque than the
    # single detection at (25, 20).
    assert alpha[10, 10] > alpha[20, 25]


def test_build_heatmap_clamps_out_of_bounds_centers():
    dets = [_det("a.jpg", 0, -50, 1000)]
    img = build_heatmap(dets, 32, 24, blur_radius=0)  # must not raise/crash
    assert img.size == (32, 24)


def _visits_from(*groups):
    """Build detections from [(x, y), ...] groups (one visit per group, 3s apart) and segment them."""
    dets = []
    ts = 0
    for gi, points in enumerate(groups):
        for x, y in points:
            dets.append(_det(f"g{gi}.jpg", ts, x, y))
            ts += 500
        ts += 10_000  # force a new visit between groups
    return segment_visits(dets, distance_threshold_px=15, time_gap_threshold_ms=3000)


def test_find_areas_groups_revisits_to_the_same_spot():
    # Three separate visits to ~(100, 100), one faraway visit at (400, 400).
    visits = _visits_from([(100, 100), (102, 101)], [(101, 99)], [(99, 100)], [(400, 400)])
    areas = find_areas(visits, area_radius_px=20)
    assert len(areas) == 2
    top = areas[0]
    assert top.visit_count == 3
    assert abs(top.centroid_x - 100) < 3
    assert abs(top.centroid_y - 100) < 3
    assert len(top.visit_indices) == 3


def test_find_areas_ranks_by_visit_count_first():
    # One spot visited twice (short dwells) beats a spot visited once with a longer dwell.
    visits = _visits_from([(10, 10)], [(10, 10)], [(500, 500), (500, 500), (500, 500), (500, 500)])
    areas = find_areas(visits, area_radius_px=15)
    assert areas[0].visit_count == 2
    assert round(areas[0].centroid_x) == 10


def test_find_areas_empty_input():
    assert find_areas([]) == []


def test_find_areas_average_duration():
    visits = _visits_from([(0, 0)], [(0, 0)])
    areas = find_areas(visits, area_radius_px=15)
    assert len(areas) == 1
    area = areas[0]
    assert area.total_duration_ms == sum(visits[i].duration_ms for i in area.visit_indices)
    assert area.avg_duration_ms == area.total_duration_ms / area.visit_count
