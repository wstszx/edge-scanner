from __future__ import annotations

import copy
import unittest
from unittest.mock import patch

from providers import sx_bet


def _deepcopy(value):
    return copy.deepcopy(value)


async def _fake_shared_client(*args, **kwargs):
    return object()


class SXBetProviderTests(unittest.IsolatedAsyncioTestCase):
    async def test_fetch_events_async_chunks_live_state_requests_when_fixture_batch_exceeds_api_limit(self) -> None:
        fixtures_payload = []
        odds_payload = {}
        for index in range(21):
            event_id = f'L2000{index:02d}'
            market_hash = f'0x{index:064x}'
            fixtures_payload.append(
                {
                    'id': event_id,
                    'eventId': event_id,
                    'teamOne': f'Home Team {index}',
                    'teamTwo': f'Away Team {index}',
                    'gameTime': 1775362800 + index,
                    'sportId': 3,
                    'sportLabel': 'Baseball',
                    'leagueId': 171,
                    'leagueLabel': 'MLB',
                    'live_state': {
                        'is_live': False,
                        'status': 'scheduled',
                        'provider_status': 'active',
                        'market_status': 'active',
                        'live_enabled': True,
                    },
                    'markets': [
                        {
                            'marketHash': market_hash,
                            'type': 52,
                            'outcomeOneName': f'Home Team {index}',
                            'outcomeTwoName': f'Away Team {index}',
                            'status': 'ACTIVE',
                            'liveEnabled': True,
                            'mainLine': True,
                            'bestOddsOutcomeOne': 2.0,
                            'bestOddsOutcomeTwo': 2.0,
                        }
                    ],
                }
            )
            odds_payload[market_hash] = {
                'marketHash': market_hash,
                'percentageOdds': '50000000000000000000',
                'price': 2.0,
            }

        fixture_status_calls = []
        live_scores_calls = []

        async def _fake_request_json_async(client, path, params=None, retries=None, backoff_seconds=None):
            if path in {'fixture/status', 'live-scores'}:
                raw_ids = str((params or {}).get('sportXEventIds') or '')
                event_ids = [item for item in raw_ids.split(',') if item]
                self.assertLessEqual(len(event_ids), 20)
                if path == 'fixture/status':
                    fixture_status_calls.append(list(event_ids))
                    return {
                        'status': 'success',
                        'data': {event_id: {'status': 1} for event_id in event_ids},
                    }, 0
                live_scores_calls.append(list(event_ids))
                return {'status': 'success', 'data': []}, 0
            if path == 'orders/odds/best':
                return {'status': 'success', 'data': dict(odds_payload)}, 0
            if path == 'orders':
                return {'status': 'success', 'data': []}, 0
            raise AssertionError((path, params))

        with (
            patch.object(sx_bet, 'get_shared_client', new=_fake_shared_client),
            patch.object(sx_bet, '_request_json_async', side_effect=_fake_request_json_async),
            patch.object(sx_bet, '_sx_best_odds_ws_enabled', return_value=False),
            patch.object(
                sx_bet,
                '_load_upcoming_fixtures_async',
                return_value=(
                    _deepcopy(fixtures_payload),
                    {'fixture_source': 'markets_active', 'cache': 'miss', 'pages_fetched': 1, 'retries_used': 0},
                ),
            ),
        ):
            events = await sx_bet.fetch_events_async(
                'baseball_mlb',
                ['h2h'],
                ['us'],
                bookmakers=['sx_bet'],
                context={'live': True, 'scan_mode': 'live'},
            )

        self.assertEqual(len(events), 21)
        self.assertEqual(len(fixture_status_calls), 2)
        self.assertEqual(len(live_scores_calls), 2)
        self.assertEqual(sum(len(batch) for batch in fixture_status_calls), 21)
        self.assertEqual(sum(len(batch) for batch in live_scores_calls), 21)

    async def test_fetch_events_async_marks_fixture_live_when_fixture_status_confirms_in_progress(self) -> None:
        fixtures_payload = [
            {
                'id': 'L17911867',
                'eventId': 'L17911867',
                'teamOne': 'Home Team',
                'teamTwo': 'Away Team',
                'gameTime': 1775362800,
                'sportId': 2,
                'sportLabel': 'Ice Hockey',
                'leagueId': 10,
                'leagueLabel': 'NHL',
                'live_state': {
                    'is_live': False,
                    'status': 'scheduled',
                    'provider_status': 'active',
                    'market_status': 'active',
                    'live_enabled': True,
                },
                'markets': [
                    {
                        'marketHash': '0xabc',
                        'type': 52,
                        'outcomeOneName': 'Home Team',
                        'outcomeTwoName': 'Away Team',
                        'status': 'ACTIVE',
                        'liveEnabled': True,
                        'mainLine': True,
                        'bestOddsOutcomeOne': 2.0,
                        'bestOddsOutcomeTwo': 2.0,
                    }
                ],
            }
        ]
        fixture_status_payload = {
            'status': 'success',
            'data': {
                'L17911867': {'status': 2}
            },
        }

        async def _fake_request_json_async(client, path, params=None, retries=None, backoff_seconds=None):
            if path == 'fixture/status':
                return _deepcopy(fixture_status_payload), 0
            if path == 'live-scores':
                return {'status': 'success', 'data': []}, 0
            if path == 'orders/odds/best':
                return {'status': 'success', 'data': {'0xabc': {'marketHash': '0xabc', 'percentageOdds': '50000000000000000000', 'price': 2.0}}}, 0
            if path == 'orders':
                return {'status': 'success', 'data': []}, 0
            raise AssertionError((path, params))

        with (
            patch.object(sx_bet, 'get_shared_client', new=_fake_shared_client),
            patch.object(sx_bet, '_request_json_async', side_effect=_fake_request_json_async),
            patch.object(sx_bet, '_sx_best_odds_ws_enabled', return_value=False),
            patch.object(sx_bet, '_load_upcoming_fixtures_async', return_value=(_deepcopy(fixtures_payload), {'fixture_source': 'markets_active', 'cache': 'miss', 'pages_fetched': 1, 'retries_used': 0})),
        ):
            events = await sx_bet.fetch_events_async(
                'icehockey_nhl',
                ['h2h'],
                ['us'],
                bookmakers=['sx_bet'],
                context={'live': True, 'scan_mode': 'live'},
            )

        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].get('live_state', {}).get('is_live'), True)
        self.assertEqual(events[0].get('live_state', {}).get('status'), 'live')

    async def test_fetch_events_async_marks_fixture_live_when_live_scores_exist(self) -> None:
        fixtures_payload = [
            {
                'id': 'L17911867',
                'eventId': 'L17911867',
                'teamOne': 'Home Team',
                'teamTwo': 'Away Team',
                'gameTime': 1775362800,
                'sportId': 2,
                'sportLabel': 'Ice Hockey',
                'leagueId': 10,
                'leagueLabel': 'NHL',
                'live_state': {
                    'is_live': False,
                    'status': 'scheduled',
                    'provider_status': 'active',
                    'market_status': 'active',
                    'live_enabled': True,
                },
                'markets': [
                    {
                        'marketHash': '0xabc',
                        'type': 52,
                        'outcomeOneName': 'Home Team',
                        'outcomeTwoName': 'Away Team',
                        'status': 'ACTIVE',
                        'liveEnabled': True,
                        'mainLine': True,
                        'bestOddsOutcomeOne': 2.0,
                        'bestOddsOutcomeTwo': 2.0,
                    }
                ],
            }
        ]
        fixture_status_payload = {
            'status': 'success',
            'data': {
                'L17911867': {'status': 1}
            },
        }
        live_scores_payload = {
            'status': 'success',
            'data': [
                {
                    'sportXEventId': 'L17911867',
                    'teamOneScore': 1,
                    'teamTwoScore': 0,
                    'currentPeriod': '2',
                    'periodTime': '10:22',
                }
            ],
        }

        async def _fake_request_json_async(client, path, params=None, retries=None, backoff_seconds=None):
            if path == 'fixture/status':
                return _deepcopy(fixture_status_payload), 0
            if path == 'live-scores':
                return _deepcopy(live_scores_payload), 0
            if path == 'orders/odds/best':
                return {'status': 'success', 'data': {'0xabc': {'marketHash': '0xabc', 'percentageOdds': '50000000000000000000', 'price': 2.0}}}, 0
            if path == 'orders':
                return {'status': 'success', 'data': []}, 0
            raise AssertionError((path, params))

        with (
            patch.object(sx_bet, 'get_shared_client', new=_fake_shared_client),
            patch.object(sx_bet, '_request_json_async', side_effect=_fake_request_json_async),
            patch.object(sx_bet, '_sx_best_odds_ws_enabled', return_value=False),
            patch.object(sx_bet, '_load_upcoming_fixtures_async', return_value=(_deepcopy(fixtures_payload), {'fixture_source': 'markets_active', 'cache': 'miss', 'pages_fetched': 1, 'retries_used': 0})),
        ):
            events = await sx_bet.fetch_events_async(
                'icehockey_nhl',
                ['h2h'],
                ['us'],
                bookmakers=['sx_bet'],
                context={'live': True, 'scan_mode': 'live'},
            )

        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].get('live_state', {}).get('is_live'), True)
        self.assertEqual(events[0].get('live_state', {}).get('status'), 'live')


if __name__ == '__main__':
    unittest.main()
