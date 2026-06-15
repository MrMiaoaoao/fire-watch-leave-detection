import unittest
from collections import deque

from full_pipeline import TrackState, vote_color


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


if __name__ == "__main__":
    unittest.main()
