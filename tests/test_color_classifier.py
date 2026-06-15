import unittest

import cv2
import numpy as np

from color_classifier import ALL_ROIS, classify, classify_batch_scored


def synthetic_person(base_bgr, accent_bgr=None):
    image = np.full((220, 120, 3), (35, 35, 35), dtype=np.uint8)
    for y in range(image.shape[0]):
        image[y, :, :] = np.clip(image[y, :, :] + y // 12, 0, 255)
    image[24:145, 18:102] = base_bgr
    if accent_bgr is not None:
        image[55:68, 18:102] = accent_bgr
        image[100:113, 18:102] = accent_bgr
    cv2.rectangle(image, (0, 0), (119, 219), (55, 55, 55), 3)
    return image


class ColorClassifierTest(unittest.TestCase):
    def test_v264_roi_contract(self):
        self.assertEqual(ALL_ROIS, [
            (0.18, 0.65, 0.20, 0.80),
            (0.10, 0.50, 0.18, 0.82),
            (0.15, 0.60, 0.08, 0.55),
            (0.15, 0.60, 0.45, 0.92),
        ])

    def test_yellow_base_classifies_as_worker(self):
        crop = synthetic_person((0, 230, 230))
        result = classify(crop)
        self.assertIsNotNone(result)
        self.assertEqual(result["color"], "yellow")
        self.assertGreater(result["yellow_ratio"], 0.1)

    def test_red_base_with_yellow_stripes_classifies_as_supervisor(self):
        crop = synthetic_person((0, 0, 220), accent_bgr=(0, 230, 230))
        result = classify(crop)
        self.assertIsNotNone(result)
        self.assertEqual(result["color"], "red")

    def test_blank_crop_returns_none(self):
        crop = np.full((160, 80, 3), 40, dtype=np.uint8)
        self.assertIsNone(classify(crop))

    def test_batch_scored_preserves_length(self):
        crops = [synthetic_person((0, 230, 230)), synthetic_person((0, 0, 220), accent_bgr=(0, 230, 230))]
        results = classify_batch_scored(crops)
        self.assertEqual(len(results), 2)
        self.assertEqual([r["color"] for r in results], ["yellow", "red"])


if __name__ == "__main__":
    unittest.main()
