from __future__ import annotations

import unittest
from unittest.mock import AsyncMock, Mock, patch

import httpx
import requests

import scanner
from providers import _async_http as async_http


class _ProviderError(Exception):
    def __init__(self, message: str, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class _FakeAsyncClient:
    def __init__(self, outcomes: list[object]) -> None:
        self._outcomes = list(outcomes)
        self.calls: list[dict] = []

    async def request(self, method: str, url: str, **kwargs) -> httpx.Response:
        self.calls.append({"method": method, "url": url, **kwargs})
        if not self._outcomes:
            raise AssertionError("Missing mocked async HTTP outcome")
        outcome = self._outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        if not isinstance(outcome, httpx.Response):
            raise AssertionError(f"Unsupported outcome type: {type(outcome)!r}")
        return outcome


def _httpx_json_response(status_code: int, payload: object, url: str = "https://example.test/api") -> httpx.Response:
    return httpx.Response(
        status_code=status_code,
        json=payload,
        request=httpx.Request("GET", url),
    )


def _httpx_text_response(status_code: int, text: str, url: str = "https://example.test/api") -> httpx.Response:
    return httpx.Response(
        status_code=status_code,
        content=text.encode("utf-8"),
        request=httpx.Request("GET", url),
    )


class ScannerDataFetchTests(unittest.TestCase):
    def test_request_wraps_timeout_as_scanner_error(self) -> None:
        with patch.object(scanner.requests, "get", side_effect=requests.Timeout("timed out")):
            with self.assertRaises(scanner.ScannerError) as ctx:
                scanner._request("https://example.test/sports", {"regions": "us"})
        self.assertIn("Network error", str(ctx.exception))
        self.assertIsNone(ctx.exception.status_code)

    def test_request_uses_json_error_message_and_status_code(self) -> None:
        response = Mock(status_code=429)
        response.json.return_value = {"message": "Rate limited"}
        response.text = ""
        with patch.object(scanner.requests, "get", return_value=response):
            with self.assertRaises(scanner.ScannerError) as ctx:
                scanner._request("https://example.test/odds", {"markets": "h2h"})
        self.assertEqual(ctx.exception.status_code, 429)
        self.assertEqual(str(ctx.exception), "Rate limited")

    def test_api_key_pool_rotates_key_on_429_then_succeeds(self) -> None:
        calls: list[dict] = []
        expected_response = object()

        def _fake_request(url: str, params: dict) -> object:
            calls.append({"url": url, "params": dict(params)})
            if params.get("apiKey") == "key-a":
                raise scanner.ScannerError("Rate limited", status_code=429)
            return expected_response

        pool = scanner.ApiKeyPool([" key-a ", "key-b"])
        with patch.object(scanner, "_request", side_effect=_fake_request):
            response = pool.request("https://example.test/odds", {"regions": "us"})

        self.assertIs(response, expected_response)
        self.assertEqual(pool.calls_made, 2)
        self.assertEqual([item["params"]["apiKey"] for item in calls], ["key-a", "key-b"])

    def test_api_key_pool_does_not_rotate_non_retriable_error(self) -> None:
        pool = scanner.ApiKeyPool(["key-a", "key-b"])
        with patch.object(
            scanner,
            "_request",
            side_effect=scanner.ScannerError("Server error", status_code=500),
        ):
            with self.assertRaises(scanner.ScannerError) as ctx:
                pool.request("https://example.test/odds", {"regions": "us"})
        self.assertEqual(ctx.exception.status_code, 500)
        self.assertEqual(pool.calls_made, 1)

    def test_fetch_sports_parse_error_message(self) -> None:
        fake_response = Mock()
        fake_response.json.side_effect = ValueError("not json")
        fake_pool = Mock()
        fake_pool.request.return_value = fake_response

        with self.assertRaises(scanner.ScannerError) as ctx:
            scanner.fetch_sports(fake_pool)
        self.assertEqual(str(ctx.exception), "Failed to parse sports list")

    def test_fetch_odds_for_market_batch_splits_invalid_market(self) -> None:
        invalid_markets: list[str] = []

        def _fake_fetch_odds(
            api_pool: object,
            sport_key: str,
            markets: list[str],
            regions: list[str],
            bookmakers=None,
        ) -> list[dict]:
            if "player_props" in markets:
                raise scanner.ScannerError("Invalid market", status_code=422)
            return [
                {
                    "id": "evt-1",
                    "sport_key": sport_key,
                    "home_team": "A",
                    "away_team": "B",
                    "commence_time": "2026-03-20T00:00:00Z",
                    "bookmakers": [
                        {
                            "key": "book-a",
                            "markets": [{"key": market, "outcomes": []} for market in markets],
                        }
                    ],
                }
            ]

        with (
            patch.object(scanner, "fetch_odds_for_sport", side_effect=_fake_fetch_odds),
            patch.object(scanner, "ODDS_API_INVALID_MARKET_STATUS_CODES", {422}),
        ):
            merged = scanner._fetch_odds_for_market_batch(
                api_pool=Mock(),
                sport_key="basketball_nba",
                markets=["h2h", "totals", "player_props"],
                regions=["us"],
                bookmakers=None,
                invalid_markets=invalid_markets,
            )

        self.assertEqual(invalid_markets, ["player_props"])
        self.assertEqual(len(merged), 1)
        market_keys = {
            m.get("key")
            for m in (merged[0].get("bookmakers") or [])[0].get("markets", [])
            if isinstance(m, dict)
        }
        self.assertEqual(market_keys, {"h2h", "totals"})

    def test_purebet_get_json_retries_parse_error_then_returns_payload(self) -> None:
        first = Mock(status_code=200)
        first.json.side_effect = ValueError("bad json")
        second = Mock(status_code=200)
        second.json.return_value = {"events": []}

        with (
            patch.object(scanner.requests, "get", side_effect=[first, second]) as mocked_get,
            patch.object(scanner.time, "sleep"),
        ):
            payload, retries_used = scanner._purebet_get_json(
                "https://v3api.purebet.io/events",
                params={"live": "false"},
                headers={"User-Agent": "test"},
                retries=1,
                backoff_seconds=0.01,
                timeout=30,
            )

        self.assertEqual(payload, {"events": []})
        self.assertEqual(retries_used, 1)
        self.assertEqual(mocked_get.call_count, 2)

    def test_purebet_get_json_retries_429_then_returns_payload(self) -> None:
        first = Mock(status_code=429)
        second = Mock(status_code=200)
        second.json.return_value = [{"id": "evt-1"}]

        with (
            patch.object(scanner.requests, "get", side_effect=[first, second]),
            patch.object(scanner.time, "sleep"),
        ):
            payload, retries_used = scanner._purebet_get_json(
                "https://v3api.purebet.io/events",
                params={"live": "false"},
                headers={"User-Agent": "test"},
                retries=1,
                backoff_seconds=0.0,
                timeout=30,
            )

        self.assertEqual(payload, [{"id": "evt-1"}])
        self.assertEqual(retries_used, 1)


class AsyncHttpFetchTests(unittest.IsolatedAsyncioTestCase):
    async def test_request_json_retries_network_error_then_succeeds(self) -> None:
        url = "https://example.test/api"
        timeout_error = httpx.ReadTimeout("timed out", request=httpx.Request("GET", url))
        client = _FakeAsyncClient([timeout_error, _httpx_json_response(200, {"ok": True}, url)])
        sleeper = AsyncMock()

        with patch("providers._async_http.asyncio.sleep", sleeper):
            payload, retries_used = await async_http.request_json(
                client=client,
                method="GET",
                url=url,
                retries=1,
                backoff_seconds=0.05,
                error_cls=_ProviderError,
                network_error_prefix="Network failed",
                parse_error_message="Parse failed",
                status_error_message=lambda code: f"Status {code}",
            )

        self.assertEqual(payload, {"ok": True})
        self.assertEqual(retries_used, 1)
        self.assertEqual(len(client.calls), 2)
        sleeper.assert_awaited_once()

    async def test_request_json_raises_parse_error_after_retries(self) -> None:
        url = "https://example.test/api"
        client = _FakeAsyncClient(
            [
                _httpx_text_response(200, "not-json", url),
                _httpx_text_response(200, "still-not-json", url),
            ]
        )

        with patch("providers._async_http.asyncio.sleep", AsyncMock()):
            with self.assertRaises(_ProviderError) as ctx:
                await async_http.request_json(
                    client=client,
                    method="GET",
                    url=url,
                    retries=1,
                    backoff_seconds=0.01,
                    error_cls=_ProviderError,
                    network_error_prefix="Network failed",
                    parse_error_message="Parse failed",
                    status_error_message=lambda code: f"Status {code}",
                )

        self.assertEqual(str(ctx.exception), "Parse failed")
        self.assertIsNone(ctx.exception.status_code)

    async def test_request_text_raises_for_non_retriable_status(self) -> None:
        url = "https://example.test/api"
        client = _FakeAsyncClient([_httpx_text_response(404, "missing", url)])

        with self.assertRaises(_ProviderError) as ctx:
            await async_http.request_text(
                client=client,
                method="GET",
                url=url,
                retries=2,
                backoff_seconds=0.01,
                error_cls=_ProviderError,
                network_error_prefix="Network failed",
                status_error_message=lambda code: f"Status {code}",
            )

        self.assertEqual(str(ctx.exception), "Status 404")
        self.assertEqual(ctx.exception.status_code, 404)
        self.assertEqual(len(client.calls), 1)

