from __future__ import annotations

import unittest
from unittest.mock import patch

from providers import betdex


class BetdexSessionAuthTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        betdex.ACCESS_TOKEN_CACHE["token"] = ""
        betdex.ACCESS_TOKEN_CACHE["expires_at"] = 0.0

    async def test_official_credentials_use_monaco_sessions_endpoint(self) -> None:
        calls: list[dict] = []

        async def _fake_request_json_async(
            client,
            url,
            params,
            access_token,
            retries,
            backoff_seconds,
            timeout,
            method="GET",
            json_payload=None,
        ):
            calls.append(
                {
                    "client": client,
                    "url": url,
                    "params": params,
                    "access_token": access_token,
                    "retries": retries,
                    "backoff_seconds": backoff_seconds,
                    "timeout": timeout,
                    "method": method,
                    "json_payload": json_payload,
                }
            )
            return {
                "sessions": [
                    {
                        "accessToken": "official-token",
                        "accessExpiresAt": "2099-01-01T00:00:00Z",
                    }
                ]
            }, 0

        with (
            patch.object(betdex, "BETDEX_APP_ID", "app-1"),
            patch.object(betdex, "BETDEX_API_KEY", "api-key-1"),
            patch.object(betdex, "_request_json_async", side_effect=_fake_request_json_async),
        ):
            token, retries_used, session_cache = await betdex._fetch_access_token_async(
                client=object(),
                retries=2,
                backoff_seconds=0.5,
                timeout=20,
            )

        self.assertEqual(token, "official-token")
        self.assertEqual(retries_used, 0)
        self.assertEqual(session_cache, "miss")
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["url"], betdex._official_session_url())
        self.assertEqual(calls[0]["method"], "POST")
        self.assertEqual(
            calls[0]["json_payload"],
            {"appId": "app-1", "apiKey": "api-key-1"},
        )
        self.assertIsNone(calls[0]["access_token"])
        self.assertEqual(betdex._cached_access_token(), "official-token")

    async def test_public_session_429_raises_actionable_error(self) -> None:
        async def _fake_request_json_async(client, url, params, access_token, retries, backoff_seconds, timeout):
            raise betdex.ProviderError("BetDEX request failed (429)", status_code=429)

        with (
            patch.object(betdex, "BETDEX_APP_ID", ""),
            patch.object(betdex, "BETDEX_API_KEY", ""),
            patch.object(betdex, "_request_json_async", side_effect=_fake_request_json_async),
        ):
            with self.assertRaisesRegex(betdex.ProviderError, "BETDEX_APP_ID and BETDEX_API_KEY") as ctx:
                await betdex._fetch_access_token_async(
                    client=object(),
                    retries=2,
                    backoff_seconds=0.5,
                    timeout=20,
                )

        self.assertEqual(ctx.exception.status_code, 429)

    async def test_partial_official_credentials_fail_fast(self) -> None:
        with (
            patch.object(betdex, "BETDEX_APP_ID", "app-1"),
            patch.object(betdex, "BETDEX_API_KEY", ""),
        ):
            with self.assertRaisesRegex(betdex.ProviderError, "requires both BETDEX_APP_ID and BETDEX_API_KEY"):
                await betdex._fetch_access_token_async(
                    client=object(),
                    retries=2,
                    backoff_seconds=0.5,
                    timeout=20,
                )
