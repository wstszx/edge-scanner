import concurrent.futures
import unittest

import scanner


class ScannerRegressionTests(unittest.TestCase):
    def tearDown(self) -> None:
        scanner._set_current_request_logger(None)
        with scanner._REQUEST_TRACE_LOCK:
            scanner._REQUEST_TRACE_ACTIVE.clear()

    def test_kelly_stake_guard_paths_return_triplet(self) -> None:
        guard_cases = [
            (0.5, 2.0, 0.0, 0.25),
            (0.5, 1.0, 1000.0, 0.25),
            (0.1, 2.0, 1000.0, 0.25),
        ]
        for args in guard_cases:
            with self.subTest(args=args):
                self.assertEqual(scanner._kelly_stake(*args), (0.0, 0.0, 0.0))

    def test_kelly_stake_positive_path_returns_triplet(self) -> None:
        full_pct, fraction_pct, stake = scanner._kelly_stake(0.6, 2.2, 1000.0, 0.5)
        self.assertGreater(full_pct, 0.0)
        self.assertGreater(fraction_pct, 0.0)
        self.assertGreater(stake, 0.0)

    def test_select_request_logger_uses_current_thread_context(self) -> None:
        logger_a = scanner._ScanRequestLogger("scan-a")
        logger_b = scanner._ScanRequestLogger("scan-b")
        scanner._set_current_request_logger(logger_a)
        selected = scanner._select_request_logger([logger_a, logger_b])
        self.assertIs(selected, logger_a)

    def test_select_request_logger_does_not_cross_scan_when_context_mismatch(self) -> None:
        logger_a = scanner._ScanRequestLogger("scan-a")
        logger_b = scanner._ScanRequestLogger("scan-b")
        scanner._set_current_request_logger(logger_a)
        self.assertIsNone(scanner._select_request_logger([logger_b]))

        scanner._set_current_request_logger(None)
        self.assertIs(scanner._select_request_logger([logger_b]), logger_b)
        self.assertIsNone(scanner._select_request_logger([logger_a, logger_b]))

    def test_submit_with_request_logger_propagates_context_to_worker(self) -> None:
        logger = scanner._ScanRequestLogger("scan-main")
        scanner._set_current_request_logger(logger)
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = scanner._submit_with_request_logger(
                executor,
                scanner._current_request_logger,
            )
            self.assertIs(future.result(), logger)


if __name__ == "__main__":
    unittest.main()
