import unittest

from leave_detector import LeaveDetector


class LeaveDetectorTest(unittest.TestCase):
    def test_warn_after_confirmed_absence_timeout(self):
        detector = LeaveDetector(timeout=1.0, fps=10.0, warmup=0.0, present_confirm_s=0.3, clear_confirm_s=0.5)

        for _ in range(3):
            self.assertIsNone(detector.update(red_count=1))

        alert = None
        for _ in range(20):
            alert = detector.update(red_count=0)
            if alert:
                break

        self.assertIsNotNone(alert)
        self.assertEqual(alert["type"], "warn")
        self.assertTrue(detector.alert_active)
        self.assertEqual(detector.report()["summary"]["total_alerts"], 1)

    def test_short_false_red_does_not_clear_active_alert(self):
        detector = LeaveDetector(timeout=1.0, fps=10.0, warmup=0.0, present_confirm_s=0.3, clear_confirm_s=0.5)

        for _ in range(3):
            detector.update(red_count=1)
        for _ in range(20):
            if detector.update(red_count=0):
                break

        self.assertTrue(detector.alert_active)
        for _ in range(4):
            self.assertIsNone(detector.update(red_count=1))
        self.assertTrue(detector.alert_active)
        self.assertEqual([a["type"] for a in detector.alerts], ["warn"])

    def test_clear_after_continuous_confirmed_red(self):
        detector = LeaveDetector(timeout=1.0, fps=10.0, warmup=0.0, present_confirm_s=0.3, clear_confirm_s=0.5)

        for _ in range(3):
            detector.update(red_count=1)
        for _ in range(20):
            if detector.update(red_count=0):
                break

        for _ in range(5):
            self.assertIsNone(detector.update(red_count=1))

        self.assertFalse(detector.alert_active)
        self.assertEqual([a["type"] for a in detector.alerts], ["warn", "clear"])


if __name__ == "__main__":
    unittest.main()
