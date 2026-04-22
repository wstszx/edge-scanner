import copy
import threading
import time
import unittest
from unittest.mock import MagicMock, patch

import app as app_module


class _ImmediateThread:
    def __init__(self, target=None, daemon=None):
        self._target = target
        self.daemon = daemon

    def start(self) -> None:
        if self._target is not None:
            self._target()


class _DeferredThread:
    def __init__(self, target=None, daemon=None):
        self._target = target
        self.daemon = daemon
        self.started = False

    def start(self) -> None:
        self.started = True

    def run_now(self) -> None:
        if self._target is not None:
            self._target()


class ScanInputValidationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = app_module.app.test_client()
        self._env_save_scan = app_module.ENV_SAVE_SCAN
        self._background_started = app_module._BACKGROUND_SERVICES_STARTED
        self._scan_jobs = copy.deepcopy(app_module._SCAN_JOBS)
        self._current_scan_job_id = app_module._CURRENT_SCAN_JOB_ID
        self._latest_scan_job_id = app_module._LATEST_SCAN_JOB_ID

    def tearDown(self) -> None:
        app_module.ENV_SAVE_SCAN = self._env_save_scan
        app_module._BACKGROUND_SERVICES_STARTED = self._background_started
        with app_module._SCAN_JOB_LOCK:
            app_module._SCAN_JOBS.clear()
            app_module._SCAN_JOBS.update(copy.deepcopy(self._scan_jobs))
            app_module._CURRENT_SCAN_JOB_ID = self._current_scan_job_id
            app_module._LATEST_SCAN_JOB_ID = self._latest_scan_job_id

    def _assert_scan_job_created(self, response, mocked_start, expected_job_id='job-123'):
        self.assertEqual(response.status_code, 202)
        payload = response.get_json() or {}
        self.assertTrue(payload.get('success'))
        self.assertEqual(payload.get('job_id'), expected_job_id)
        mocked_start.assert_called_once()
        return mocked_start.call_args.args[0]

    def test_scan_returns_job_payload_for_valid_request(self) -> None:
        thread_holder = {}

        def _thread_factory(target=None, daemon=None):
            thread = _DeferredThread(target=target, daemon=daemon)
            thread_holder['thread'] = thread
            return thread

        with (
            patch.object(app_module, '_start_scan_job', return_value={'success': True, 'job_id': 'job-123', 'status': 'running'}) as mocked_start,
            patch.object(app_module.threading, 'Thread', side_effect=_thread_factory),
        ):
            response = self.client.post('/scan', json={'sports': ['basketball_nba']})

        self.assertEqual(response.status_code, 202)
        payload = response.get_json() or {}
        self.assertTrue(payload.get('success'))
        self.assertEqual(payload.get('job_id'), 'job-123')
        self.assertEqual(payload.get('status'), 'running')
        mocked_start.assert_called_once()

    def test_scan_returns_conflict_with_running_job_id(self) -> None:
        with patch.object(
            app_module,
            '_start_scan_job',
            return_value={'success': False, 'error': 'Scan already in progress', 'error_code': 409, 'job_id': 'job-running'},
        ):
            response = self.client.post('/scan', json={'sports': ['basketball_nba']})

        self.assertEqual(response.status_code, 409)
        payload = response.get_json() or {}
        self.assertFalse(payload.get('success'))
        self.assertEqual(payload.get('job_id'), 'job-running')
        self.assertEqual(payload.get('error'), 'Scan already in progress')

    def test_scan_job_status_endpoint_returns_partial_snapshot(self) -> None:
        with patch.object(
            app_module,
            '_get_scan_job_snapshot',
            return_value={
                'job_id': 'job-123',
                'status': 'running',
                'partial_result': {
                    'success': True,
                    'partial': True,
                    'arbitrage': {'opportunities': [{'event': 'A vs B'}], 'opportunities_count': 1},
                },
            },
        ):
            response = self.client.get('/scan/jobs/job-123')

        self.assertEqual(response.status_code, 200)
        payload = response.get_json() or {}
        self.assertTrue(payload.get('success'))
        self.assertEqual(payload.get('job', {}).get('job_id'), 'job-123')
        self.assertEqual(payload.get('job', {}).get('status'), 'running')
        partial = (payload.get('job', {}) or {}).get('partial_result') or {}
        self.assertTrue(partial.get('partial'))
        self.assertEqual((((partial.get('arbitrage') or {}).get('opportunities')) or [])[0].get('event'), 'A vs B')

    def test_scan_current_job_endpoint_returns_latest_job(self) -> None:
        with patch.object(
            app_module,
            '_get_current_or_latest_scan_job_snapshot',
            return_value={'job_id': 'job-999', 'status': 'completed', 'final_result': {'success': True}},
        ):
            response = self.client.get('/scan/jobs/current')

        self.assertEqual(response.status_code, 200)
        payload = response.get_json() or {}
        self.assertTrue(payload.get('success'))
        self.assertEqual(payload.get('job', {}).get('job_id'), 'job-999')
        self.assertEqual(payload.get('job', {}).get('status'), 'completed')

    def test_start_scan_job_runs_background_scan_and_sets_final_result(self) -> None:
        result_payload = {
            'success': True,
            'scan_time': '2026-02-22T12:00:00Z',
            'arbitrage': {'opportunities': [{'event': 'A vs B'}], 'opportunities_count': 1},
            'middles': {'opportunities': [], 'opportunities_count': 0},
            'plus_ev': {'opportunities': [], 'opportunities_count': 0},
        }
        thread_holder = {}

        def _thread_factory(target=None, daemon=None):
            thread = _DeferredThread(target=target, daemon=daemon)
            thread_holder['thread'] = thread
            return thread

        with (
            patch.object(app_module, '_execute_scan_payload', return_value=result_payload),
            patch.object(app_module.threading, 'Thread', side_effect=_thread_factory),
        ):
            started = app_module._start_scan_job({'sports': ['basketball_nba']})
            thread_holder['thread'].run_now()

        self.assertTrue(started.get('success'))
        snapshot = app_module._get_scan_job_snapshot(started.get('job_id')) or {}
        self.assertEqual(snapshot.get('status'), 'completed')
        final_result = snapshot.get('final_result') or {}
        self.assertTrue(final_result.get('success'))
        self.assertEqual((((final_result.get('arbitrage') or {}).get('opportunities')) or [])[0].get('event'), 'A vs B')

    def test_start_scan_job_allows_only_one_runner_under_concurrent_admission(self) -> None:
        real_thread = threading.Thread
        results = []
        start_barrier = threading.Barrier(2)
        original_register = app_module._register_scan_job

        def _thread_factory(target=None, daemon=None):
            return _DeferredThread(target=target, daemon=daemon)

        def _delayed_register(payload, *args, **kwargs):
            time.sleep(0.01)
            return original_register(payload, *args, **kwargs)

        with (
            patch.object(app_module, '_register_scan_job', side_effect=_delayed_register),
            patch.object(app_module.threading, 'Thread', side_effect=_thread_factory),
        ):
            def _worker() -> None:
                start_barrier.wait()
                results.append(app_module._start_scan_job({'sports': ['basketball_nba']}))

            first = real_thread(target=_worker)
            second = real_thread(target=_worker)
            first.start()
            second.start()
            first.join()
            second.join()

        successes = [item for item in results if item.get('success')]
        conflicts = [item for item in results if item.get('error_code') == 409]

        self.assertEqual(len(successes), 1)
        self.assertEqual(len(conflicts), 1)
        self.assertEqual(len(app_module._SCAN_JOBS), 1)

    def test_update_scan_job_prunes_old_finished_jobs_beyond_history_limit(self) -> None:
        with patch.object(app_module, 'SCAN_JOB_HISTORY_LIMIT', 2, create=True):
            first = app_module._register_scan_job({'sports': ['basketball_nba']})
            second = app_module._register_scan_job({'sports': ['basketball_nba']})
            third = app_module._register_scan_job({'sports': ['basketball_nba']})

            app_module._update_scan_job(
                first['job_id'],
                status='completed',
                finished_at='2026-02-22T12:00:01Z',
                final_result={'success': True, 'scan_time': '2026-02-22T12:00:01Z'},
            )
            app_module._update_scan_job(
                second['job_id'],
                status='completed',
                finished_at='2026-02-22T12:00:02Z',
                final_result={'success': True, 'scan_time': '2026-02-22T12:00:02Z'},
            )
            app_module._update_scan_job(
                third['job_id'],
                status='completed',
                finished_at='2026-02-22T12:00:03Z',
                final_result={'success': True, 'scan_time': '2026-02-22T12:00:03Z'},
            )

        self.assertNotIn(first['job_id'], app_module._SCAN_JOBS)
        self.assertIn(second['job_id'], app_module._SCAN_JOBS)
        self.assertIn(third['job_id'], app_module._SCAN_JOBS)
        self.assertEqual(app_module._LATEST_SCAN_JOB_ID, third['job_id'])
        latest_snapshot = app_module._get_current_or_latest_scan_job_snapshot() or {}
        self.assertEqual(latest_snapshot.get('job_id'), third['job_id'])

    def test_start_scan_job_progress_updates_partial_result(self) -> None:
        thread_holder = {}

        def _thread_factory(target=None, daemon=None):
            thread = _DeferredThread(target=target, daemon=daemon)
            thread_holder['thread'] = thread
            return thread

        def _fake_execute(payload, save_scan_override=None, background=False, progress_callback=None):
            progress_callback({
                'type': 'provider_completed',
                'sport_key': 'basketball_nba',
                'provider_key': 'sx_bet',
                'provider': 'SX Bet',
                'ms': 123.0,
                'events_returned': 2,
                'error': None,
            })
            progress_callback({
                'type': 'sport_completed',
                'sport_key': 'basketball_nba',
                'result': {
                    'arb_opportunities': [{'event': 'A vs B', 'roi_percent': 1.5, 'stakes': {'guaranteed_profit': 1.0}}],
                    'middle_opportunities': [],
                    'plus_ev_opportunities': [],
                    'events_scanned': 1,
                    'total_profit': 1.0,
                    'successful': 1,
                },
            })
            return {
                'success': True,
                'scan_time': '2026-02-22T12:00:00Z',
                'arbitrage': {'opportunities': [{'event': 'A vs B'}], 'opportunities_count': 1},
                'middles': {'opportunities': [], 'opportunities_count': 0},
                'plus_ev': {'opportunities': [], 'opportunities_count': 0},
            }

        with (
            patch.object(app_module, '_execute_scan_payload', side_effect=_fake_execute),
            patch.object(app_module.threading, 'Thread', side_effect=_thread_factory),
        ):
            started = app_module._start_scan_job({'sports': ['basketball_nba']})
            thread_holder['thread'].run_now()

        snapshot = app_module._get_scan_job_snapshot(started.get('job_id')) or {}
        progress = snapshot.get('progress') or {}
        self.assertEqual(progress.get('providers_completed'), 1)
        self.assertEqual(progress.get('sports_completed'), 1)
        partial_result = snapshot.get('partial_result') or {}
        self.assertTrue(partial_result.get('partial'))
        self.assertEqual((((partial_result.get('arbitrage') or {}).get('opportunities')) or [])[0].get('event'), 'A vs B')

    def test_start_scan_job_scan_started_progress_sets_totals(self) -> None:
        thread_holder = {}

        def _thread_factory(target=None, daemon=None):
            thread = _DeferredThread(target=target, daemon=daemon)
            thread_holder['thread'] = thread
            return thread

        def _fake_execute(payload, save_scan_override=None, background=False, progress_callback=None):
            progress_callback({
                'type': 'scan_started',
                'sports_total': 1,
                'providers_total': 2,
            })
            return {
                'success': True,
                'scan_time': '2026-02-22T12:00:00Z',
                'arbitrage': {'opportunities': [], 'opportunities_count': 0},
                'middles': {'opportunities': [], 'opportunities_count': 0},
                'plus_ev': {'opportunities': [], 'opportunities_count': 0},
            }

        with (
            patch.object(app_module, '_execute_scan_payload', side_effect=_fake_execute),
            patch.object(app_module.threading, 'Thread', side_effect=_thread_factory),
        ):
            started = app_module._start_scan_job({'sports': ['basketball_nba']})
            thread_holder['thread'].run_now()

        snapshot = app_module._get_scan_job_snapshot(started.get('job_id')) or {}
        progress = snapshot.get('progress') or {}
        self.assertEqual(progress.get('sports_total'), 1)
        self.assertEqual(progress.get('providers_total'), 2)

    def test_start_scan_job_provider_progress_can_publish_partial_result_before_sport_completion(self) -> None:
        thread_holder = {}

        def _thread_factory(target=None, daemon=None):
            thread = _DeferredThread(target=target, daemon=daemon)
            thread_holder['thread'] = thread
            return thread

        def _fake_execute(payload, save_scan_override=None, background=False, progress_callback=None):
            progress_callback({
                'type': 'provider_completed',
                'sport_key': 'basketball_nba',
                'provider_key': 'sx_bet',
                'provider': 'SX Bet',
                'ms': 123.0,
                'events_returned': 2,
                'error': None,
                'result': {
                    'arb_opportunities': [{'event': 'Early Arb', 'roi_percent': 1.2, 'stakes': {'guaranteed_profit': 0.8}}],
                    'middle_opportunities': [],
                    'plus_ev_opportunities': [],
                },
            })
            return {
                'success': True,
                'scan_time': '2026-02-22T12:00:00Z',
                'arbitrage': {'opportunities': [{'event': 'Early Arb'}], 'opportunities_count': 1},
                'middles': {'opportunities': [], 'opportunities_count': 0},
                'plus_ev': {'opportunities': [], 'opportunities_count': 0},
            }

        with (
            patch.object(app_module, '_execute_scan_payload', side_effect=_fake_execute),
            patch.object(app_module.threading, 'Thread', side_effect=_thread_factory),
        ):
            started = app_module._start_scan_job({'sports': ['basketball_nba']})
            thread_holder['thread'].run_now()

        snapshot = app_module._get_scan_job_snapshot(started.get('job_id')) or {}
        partial_result = snapshot.get('partial_result') or {}
        self.assertTrue(partial_result.get('partial'))
        self.assertEqual((((partial_result.get('arbitrage') or {}).get('opportunities')) or [])[0].get('event'), 'Early Arb')

    def test_start_scan_job_provider_partial_progress_accumulates_before_provider_completion(self) -> None:
        thread_holder = {}

        def _thread_factory(target=None, daemon=None):
            thread = _DeferredThread(target=target, daemon=daemon)
            thread_holder['thread'] = thread
            return thread

        def _fake_execute(payload, save_scan_override=None, background=False, progress_callback=None):
            progress_callback({
                'type': 'provider_partial',
                'sport_key': 'basketball_nba',
                'provider_key': 'polymarket',
                'provider': 'Polymarket',
                'result': {
                    'arb_opportunities': [{'event': 'Early Partial Arb', 'roi_percent': 1.1, 'stakes': {'guaranteed_profit': 0.7}}],
                    'middle_opportunities': [],
                    'plus_ev_opportunities': [],
                },
            })
            progress_callback({
                'type': 'provider_completed',
                'sport_key': 'basketball_nba',
                'provider_key': 'polymarket',
                'provider': 'Polymarket',
                'ms': 456.0,
                'events_returned': 3,
                'error': None,
                'result': {
                    'arb_opportunities': [
                        {'event': 'Early Partial Arb', 'roi_percent': 1.1, 'stakes': {'guaranteed_profit': 0.7}},
                        {'event': 'Late Provider Arb', 'roi_percent': 0.9, 'stakes': {'guaranteed_profit': 0.5}},
                    ],
                    'middle_opportunities': [],
                    'plus_ev_opportunities': [],
                },
            })
            return {
                'success': True,
                'scan_time': '2026-02-22T12:00:00Z',
                'arbitrage': {'opportunities': [{'event': 'Early Partial Arb'}, {'event': 'Late Provider Arb'}], 'opportunities_count': 2},
                'middles': {'opportunities': [], 'opportunities_count': 0},
                'plus_ev': {'opportunities': [], 'opportunities_count': 0},
            }

        with (
            patch.object(app_module, '_execute_scan_payload', side_effect=_fake_execute),
            patch.object(app_module.threading, 'Thread', side_effect=_thread_factory),
        ):
            started = app_module._start_scan_job({'sports': ['basketball_nba']})
            thread_holder['thread'].run_now()

        snapshot = app_module._get_scan_job_snapshot(started.get('job_id')) or {}
        partial_result = snapshot.get('partial_result') or {}
        arbitrage_items = (((partial_result.get('arbitrage') or {}).get('opportunities')) or [])
        self.assertEqual([item.get('event') for item in arbitrage_items], ['Early Partial Arb', 'Late Provider Arb'])

    def test_start_scan_job_provider_progress_accumulates_partial_results(self) -> None:
        thread_holder = {}

        def _thread_factory(target=None, daemon=None):
            thread = _DeferredThread(target=target, daemon=daemon)
            thread_holder['thread'] = thread
            return thread

        def _fake_execute(payload, save_scan_override=None, background=False, progress_callback=None):
            progress_callback({
                'type': 'provider_completed',
                'sport_key': 'basketball_nba',
                'provider_key': 'sx_bet',
                'provider': 'SX Bet',
                'ms': 123.0,
                'events_returned': 2,
                'error': None,
                'result': {
                    'arb_opportunities': [{'event': 'First Arb', 'roi_percent': 1.2, 'stakes': {'guaranteed_profit': 0.8}}],
                    'middle_opportunities': [],
                    'plus_ev_opportunities': [],
                },
            })
            first_snapshot = app_module._get_scan_job_snapshot(started.get('job_id')) or {}
            first_partial = (((first_snapshot.get('partial_result') or {}).get('arbitrage') or {}).get('opportunities')) or []
            self.assertEqual([item.get('event') for item in first_partial], ['First Arb'])

            progress_callback({
                'type': 'provider_completed',
                'sport_key': 'basketball_nba',
                'provider_key': 'betdex',
                'provider': 'BetDEX',
                'ms': 456.0,
                'events_returned': 3,
                'error': None,
                'result': {
                    'arb_opportunities': [{'event': 'Second Arb', 'roi_percent': 0.9, 'stakes': {'guaranteed_profit': 0.5}}],
                    'middle_opportunities': [],
                    'plus_ev_opportunities': [],
                },
            })
            return {
                'success': True,
                'scan_time': '2026-02-22T12:00:00Z',
                'arbitrage': {'opportunities': [{'event': 'First Arb'}, {'event': 'Second Arb'}], 'opportunities_count': 2},
                'middles': {'opportunities': [], 'opportunities_count': 0},
                'plus_ev': {'opportunities': [], 'opportunities_count': 0},
            }

        with (
            patch.object(app_module, '_execute_scan_payload', side_effect=_fake_execute),
            patch.object(app_module.threading, 'Thread', side_effect=_thread_factory),
        ):
            started = app_module._start_scan_job({'sports': ['basketball_nba']})
            thread_holder['thread'].run_now()

        snapshot = app_module._get_scan_job_snapshot(started.get('job_id')) or {}
        partial_result = snapshot.get('partial_result') or {}
        arbitrage_items = (((partial_result.get('arbitrage') or {}).get('opportunities')) or [])
        self.assertEqual([item.get('event') for item in arbitrage_items], ['First Arb', 'Second Arb'])

    def test_start_scan_job_provider_progress_preserves_negative_roi_partial_result(self) -> None:
        thread_holder = {}

        def _thread_factory(target=None, daemon=None):
            thread = _DeferredThread(target=target, daemon=daemon)
            thread_holder['thread'] = thread
            return thread

        def _fake_execute(payload, save_scan_override=None, background=False, progress_callback=None):
            progress_callback({
                'type': 'provider_completed',
                'sport_key': 'basketball_nba',
                'provider_key': 'sx_bet',
                'provider': 'SX Bet',
                'ms': 123.0,
                'events_returned': 2,
                'error': None,
                'result': {
                    'arb_opportunities': [{'event': 'Near Arb', 'roi_percent': -0.8, 'stakes': {'guaranteed_profit': -0.8}}],
                    'middle_opportunities': [],
                    'plus_ev_opportunities': [],
                },
            })
            return {
                'success': True,
                'scan_time': '2026-02-22T12:00:00Z',
                'arbitrage': {'opportunities': [{'event': 'Near Arb', 'roi_percent': -0.8}], 'opportunities_count': 1},
                'middles': {'opportunities': [], 'opportunities_count': 0},
                'plus_ev': {'opportunities': [], 'opportunities_count': 0},
            }

        with (
            patch.object(app_module, '_execute_scan_payload', side_effect=_fake_execute),
            patch.object(app_module.threading, 'Thread', side_effect=_thread_factory),
        ):
            started = app_module._start_scan_job({'sports': ['basketball_nba']})
            thread_holder['thread'].run_now()

        snapshot = app_module._get_scan_job_snapshot(started.get('job_id')) or {}
        partial_result = snapshot.get('partial_result') or {}
        arbitrage_items = (((partial_result.get('arbitrage') or {}).get('opportunities')) or [])
        self.assertEqual(len(arbitrage_items), 1)
        self.assertEqual(arbitrage_items[0].get('event'), 'Near Arb')
        self.assertEqual(arbitrage_items[0].get('roi_percent'), -0.8)

    def test_scan_rejects_invalid_json_payload(self) -> None:
        response = self.client.post(
            "/scan",
            data='{"bad":',
            headers={"Content-Type": "application/json"},
        )
        self.assertEqual(response.status_code, 400)
        payload = response.get_json() or {}
        self.assertEqual(payload.get("error"), "Invalid JSON payload")
        self.assertEqual(payload.get("error_code"), 400)
        self.assertFalse(payload.get("success"))

    def test_scan_rejects_non_object_payload(self) -> None:
        response = self.client.post(
            "/scan",
            data='["not-an-object"]',
            headers={"Content-Type": "application/json"},
        )
        self.assertEqual(response.status_code, 400)
        payload = response.get_json() or {}
        self.assertEqual(payload.get("error"), "Scan payload must be a JSON object")
        self.assertEqual(payload.get("error_code"), 400)
        self.assertFalse(payload.get("success"))

    def test_scan_uses_default_sharp_book_for_non_string(self) -> None:
        with patch.object(app_module, "_start_scan_job", return_value={"success": True, "job_id": "job-123", "status": "running"}) as mocked_start:
            response = self.client.post("/scan", json={"sharpBook": 123})
        payload = self._assert_scan_job_created(response, mocked_start)
        self.assertEqual(payload.get("sharpBook"), 123)

    def test_scan_parses_boolean_strings_from_payload(self) -> None:
        app_module.ENV_SAVE_SCAN = True
        with patch.object(app_module, "_start_scan_job", return_value={"success": True, "job_id": "job-123", "status": "running"}) as mocked_start:
            response = self.client.post(
                "/scan",
                json={
                    "saveScan": "false",
                    "allSports": "false",
                    "allMarkets": "false",
                },
            )
        payload = self._assert_scan_job_created(response, mocked_start)
        self.assertEqual(payload.get("allSports"), "false")
        self.assertEqual(payload.get("allMarkets"), "false")

    def test_scan_derives_include_providers_when_empty_list_is_sent(self) -> None:
        with patch.object(app_module, "_start_scan_job", return_value={"success": True, "job_id": "job-123", "status": "running"}) as mocked_start:
            response = self.client.post(
                "/scan",
                json={
                    "bookmakers": ["SX Bet"],
                    "includeProviders": [],
                },
            )
        payload = self._assert_scan_job_created(response, mocked_start)
        self.assertEqual(payload.get("includeProviders"), [])

    def test_scan_accepts_live_mode_without_api_key(self) -> None:
        with patch.object(app_module, "_start_scan_job", return_value={"success": True, "job_id": "job-123", "status": "running"}) as mocked_start:
            response = self.client.post(
                "/scan",
                json={
                    "scanMode": "live",
                    "sports": ["basketball_nba"],
                },
            )
        payload = self._assert_scan_job_created(response, mocked_start)
        self.assertEqual(payload.get("scanMode"), "live")

    def test_scan_rejects_invalid_scan_mode(self) -> None:
        with patch.object(app_module, "run_scan", return_value={"success": True}) as mocked_run_scan:
            response = self.client.post(
                "/scan",
                json={
                    "scanMode": "LiVeSoon",
                    "sports": ["basketball_nba"],
                },
            )

        self.assertEqual(response.status_code, 400)
        payload = response.get_json() or {}
        self.assertFalse(payload.get("success"))
        self.assertEqual(payload.get("error_code"), 400)
        self.assertEqual(payload.get("error"), "Invalid scanMode")
        mocked_run_scan.assert_not_called()

    def test_server_auto_scan_config_rejects_invalid_scan_mode(self) -> None:
        response = self.client.post(
            "/server-auto-scan-config",
            json={
                "enabled": True,
                "intervalMinutes": 5,
                "payload": {
                    "scanMode": "LiVeSoon",
                    "sports": ["basketball_nba"],
                },
            },
        )

        self.assertEqual(response.status_code, 400)
        payload = response.get_json() or {}
        self.assertFalse(payload.get("success"))
        self.assertEqual(payload.get("error_code"), 400)
        self.assertEqual(payload.get("error"), "Invalid scanMode")

    def test_scan_accepts_provider_only_mode_without_api_key(self) -> None:
        with (
            patch.object(app_module, "ENV_PROVIDER_ONLY_MODE", True),
            patch.object(app_module, "_start_scan_job", return_value={"success": True, "job_id": "job-123", "status": "running"}) as mocked_start,
        ):
            response = self.client.post(
                "/scan",
                json={
                    "bookmakers": ["SX Bet"],
                    "includeProviders": [],
                },
            )

        payload = self._assert_scan_job_created(response, mocked_start)
        self.assertEqual(payload.get("bookmakers"), ["SX Bet"])

    def test_scan_derives_regions_from_selected_platforms_when_omitted(self) -> None:
        with (
            patch.object(app_module, "ENV_PROVIDER_ONLY_MODE", False),
            patch.object(app_module, "_start_scan_job", return_value={"success": True, "job_id": "job-123", "status": "running"}) as mocked_start,
        ):
            response = self.client.post(
                "/scan",
                json={
                    "bookmakers": ["betfair_ex_uk", "SX Bet"],
                    "sharpBook": "pinnacle",
                },
            )
        payload = self._assert_scan_job_created(response, mocked_start)
        self.assertEqual(payload.get("sharpBook"), "pinnacle")

    def test_scan_clamps_kelly_fraction_to_zero_and_one(self) -> None:
        with patch.object(app_module, "_start_scan_job", return_value={"success": True, "job_id": "job-123", "status": "running"}) as mocked_start:
            high_response = self.client.post("/scan", json={"kellyFraction": 5})
            low_response = self.client.post("/scan", json={"kellyFraction": -3})

        self.assertEqual(high_response.status_code, 202)
        self.assertEqual(low_response.status_code, 202)
        self.assertEqual(mocked_start.call_count, 2)

    def test_execute_scan_payload_saves_history_from_nested_result_shape(self) -> None:
        result_payload = {
            "success": True,
            "scan_time": "2026-02-22T12:00:00Z",
            "arbitrage": {"opportunities": [{"event": "A vs B"}]},
            "middles": {"opportunities": [{"event": "C vs D"}]},
            "plus_ev": {"opportunities": [{"event": "E vs F"}]},
        }
        history_manager = MagicMock()
        notifier = MagicMock()
        notifier.is_configured = False
        with (
            patch.object(app_module, "run_scan", return_value=result_payload),
            patch.object(app_module, "get_history_manager", return_value=history_manager),
            patch.object(app_module, "get_notifier", return_value=notifier),
        ):
            payload = app_module._execute_scan_payload({})

        self.assertEqual(payload.get("scan_time"), "2026-02-22T12:00:00Z")
        history_manager.save_opportunities.assert_called_once()
        history_manager.save_scan_summary.assert_called_once()
        history_payload = history_manager.save_opportunities.call_args.args[0]
        self.assertEqual(len(history_payload.get("opportunities") or []), 1)
        self.assertEqual(len(history_payload.get("middles") or []), 1)
        self.assertEqual(len(history_payload.get("plus_ev") or []), 1)

    def test_execute_scan_payload_preserves_negative_roi_arbitrage_results_from_run_scan(self) -> None:
        result_payload = {
            "success": True,
            "scan_time": "2026-02-22T12:00:00Z",
            "arbitrage": {
                "opportunities": [
                    {
                        "event": "A vs B",
                        "roi_percent": -0.85,
                        "stakes": {"guaranteed_profit": -0.85},
                    }
                ]
            },
            "middles": {"opportunities": []},
            "plus_ev": {"opportunities": []},
        }
        history_manager = MagicMock()
        notifier = MagicMock()
        notifier.is_configured = False
        with (
            patch.object(app_module, "run_scan", return_value=result_payload),
            patch.object(app_module, "get_history_manager", return_value=history_manager),
            patch.object(app_module, "get_notifier", return_value=notifier),
        ):
            payload = app_module._execute_scan_payload({})

        arbitrage_items = (((payload.get("arbitrage") or {}).get("opportunities")) or [])
        self.assertEqual(len(arbitrage_items), 1)
        self.assertEqual(arbitrage_items[0].get("roi_percent"), -0.85)
        history_payload = history_manager.save_opportunities.call_args.args[0]
        saved_opportunities = history_payload.get("opportunities") or []
        self.assertEqual(len(saved_opportunities), 1)
        self.assertEqual(saved_opportunities[0].get("roi_percent"), -0.85)

    def test_execute_scan_payload_ignores_history_save_failures(self) -> None:
        result_payload = {
            "success": True,
            "scan_time": "2026-02-22T12:00:00Z",
            "arbitrage": {"opportunities": [{"event": "A vs B"}]},
            "middles": {"opportunities": []},
            "plus_ev": {"opportunities": []},
        }
        history_manager = MagicMock()
        history_manager.save_opportunities.side_effect = RuntimeError("disk full")
        notifier = MagicMock()
        notifier.is_configured = False

        with (
            patch.object(app_module, "run_scan", return_value=result_payload),
            patch.object(app_module, "get_history_manager", return_value=history_manager),
            patch.object(app_module, "get_notifier", return_value=notifier),
        ):
            payload = app_module._execute_scan_payload({})

        self.assertTrue(payload.get("success"))
        history_manager.save_opportunities.assert_called_once()
        history_manager.save_scan_summary.assert_called_once()

    def test_execute_scan_payload_ignores_notification_failures(self) -> None:
        result_payload = {
            "success": True,
            "scan_time": "2026-02-22T12:00:00Z",
            "arbitrage": {"opportunities": [{"event": "A vs B"}]},
            "middles": {"opportunities": []},
            "plus_ev": {"opportunities": []},
        }
        history_manager = MagicMock()
        notifier = MagicMock()
        notifier.is_configured = True
        notifier.notify_opportunities.side_effect = RuntimeError("notify failed")

        with (
            patch.object(app_module, "run_scan", return_value=result_payload),
            patch.object(app_module, "get_history_manager", return_value=history_manager),
            patch.object(app_module, "get_notifier", return_value=notifier),
            patch.object(app_module.threading, "Thread", side_effect=lambda target=None, daemon=None: _ImmediateThread(target=target, daemon=daemon)),
        ):
            payload = app_module._execute_scan_payload({})

        self.assertTrue(payload.get("success"))
        notifier.notify_opportunities.assert_called_once()

    def test_index_prewarms_background_services(self) -> None:
        with patch.object(app_module, "_start_background_provider_services") as mocked_start:
            response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        mocked_start.assert_called_once_with(wait_timeout=0.0)

    def test_index_hides_soft_bookmakers_from_supported_platforms_list(self) -> None:
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        html = response.get_data(as_text=True)
        self.assertIn('value="sx_bet"', html)
        self.assertIn('value="pinnacle"', html)
        self.assertNotIn('name="regions"', html)
        self.assertNotIn('value="draftkings"', html)
        self.assertNotIn('value="fanduel"', html)

    def test_scan_rejects_unsupported_bookmakers_only(self) -> None:
        with patch.object(app_module, "_start_scan_job", return_value={"success": True, "job_id": "job-123", "status": "running"}) as mocked_start_scan_job:
            response = self.client.post(
                "/scan",
                json={
                    "bookmakers": ["DraftKings", "FanDuel"],
                    "includeProviders": [],
                },
            )
        self.assertEqual(response.status_code, 400)
        payload = response.get_json() or {}
        self.assertFalse(payload.get("success"))
        self.assertEqual(
            payload.get("error"),
            "No supported arbitrage platforms were selected",
        )
        mocked_start_scan_job.assert_not_called()

    def test_provider_runtime_returns_status_payload(self) -> None:
        runtime_payload = {
            "enabled": True,
            "started": True,
            "ready": True,
            "status": {
                "market_connected": True,
                "market_last_message_at": "2026-03-22T12:00:00Z",
            },
        }
        with (
            patch.object(app_module, "_start_background_provider_services") as mocked_start,
            patch.object(app_module, "_provider_runtime_status", return_value=runtime_payload),
        ):
            response = self.client.get("/provider-runtime/polymarket")
        self.assertEqual(response.status_code, 200)
        mocked_start.assert_called_once_with(wait_timeout=0.0)
        payload = response.get_json() or {}
        self.assertTrue(payload.get("success"))
        self.assertEqual(payload.get("provider_key"), "polymarket")
        self.assertTrue(payload.get("ready"))
        self.assertEqual(
            ((payload.get("status") or {}).get("market_last_message_at")),
            "2026-03-22T20:00:00+08:00",
        )

    def test_provider_runtime_rejects_unknown_provider(self) -> None:
        with patch.object(app_module, "_start_background_provider_services") as mocked_start:
            response = self.client.get("/provider-runtime/unknown")
        self.assertEqual(response.status_code, 404)
        mocked_start.assert_called_once_with(wait_timeout=0.0)
        payload = response.get_json() or {}
        self.assertFalse(payload.get("success"))


if __name__ == "__main__":
    unittest.main()
