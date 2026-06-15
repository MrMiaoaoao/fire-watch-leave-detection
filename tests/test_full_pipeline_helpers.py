import unittest
from collections import deque

from full_pipeline import (
    TrackState,
    box_iou_contain,
    deduplicate_detections,
    deduplicate_persons,
    find_overlaps,
    vote_color,
)


class FullPipelineHelperTest(unittest.TestCase):
    def test_vote_color_requires_minimum_valid_votes(self):
        self.assertIsNone(vote_color(["red", "red", "red"], vote_thres=3))
        self.assertEqual(vote_color(["red", "red", "red", "red"], vote_thres=3), "red")
        self.assertEqual(vote_color(["yellow", "yellow", "yellow", "yellow"], vote_thres=3), "yellow")

    def test_track_state_updates_box_and_last_frame(self):
        state = TrackState(1, 2, 3, 4, 5, deque(maxlen=3))
        state.color_history.extend(["red", None, "yellow"])

        state.update_box(10, 20, 30, 40, 50)

        self.assertEqual((state.x1, state.y1, state.x2, state.y2), (10, 20, 30, 40))
        self.assertEqual(state.last_frame, 50)
        self.assertEqual(list(state.color_history), ["red", None, "yellow"])

    def test_box_iou_contain(self):
        iou, contain = box_iou_contain([0, 0, 100, 100], [50, 50, 150, 150])

        self.assertAlmostEqual(iou, 2500 / 17500, places=4)
        self.assertAlmostEqual(contain, 0.25, places=4)

    def test_find_overlaps_for_dict_detections(self):
        detections = [
            {"bbox": [0, 0, 100, 100], "track_id": 1, "color": "red"},
            {"bbox": [10, 10, 90, 90], "track_id": 2, "color": "red"},
            {"bbox": [200, 200, 260, 260], "track_id": 3, "color": "yellow"},
        ]

        overlaps = find_overlaps(detections, iou_thres=0.15, contain_thres=0.40)

        self.assertEqual(len(overlaps), 1)
        self.assertEqual(overlaps[0]["tid_i"], 1)
        self.assertEqual(overlaps[0]["tid_j"], 2)

    def test_deduplicate_persons_keeps_higher_confidence(self):
        persons = [
            (0, 0, 100, 100, 0.50, 1),
            (10, 10, 90, 90, 0.90, 2),
            (200, 200, 260, 260, 0.40, 3),
        ]

        kept = deduplicate_persons(persons)

        self.assertEqual([p[5] for p in kept], [2, 3])

    def test_deduplicate_detections_keeps_higher_confidence(self):
        detections = [
            {"bbox": [0, 0, 100, 100], "track_id": 1, "color": "red", "conf": 0.50},
            {"bbox": [10, 10, 90, 90], "track_id": 2, "color": "red", "conf": 0.90},
            {"bbox": [200, 200, 260, 260], "track_id": 3, "color": "yellow", "conf": 0.40},
        ]

        kept = deduplicate_detections(detections)

        self.assertEqual([d["track_id"] for d in kept], [2, 3])


if __name__ == "__main__":
    unittest.main()
