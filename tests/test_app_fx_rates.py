import unittest
from unittest.mock import MagicMock, patch

import requests

import app as app_module


class FxRateRouteTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = app_module.app.test_client()
        self._cached_payload = app_module._FX_RATE_CACHE
        self._cached_expires_at = app_module._FX_RATE_CACHE_EXPIRES_AT

    def tearDown(self) -> None:
        app_module._FX_RATE_CACHE = self._cached_payload
        app_module._FX_RATE_CACHE_EXPIRES_AT = self._cached_expires_at

    def test_fx_rates_returns_reference_payload(self) -> None:
        mocked_response = MagicMock()
        mocked_response.json.return_value = {
            "date": "2026-03-15",
            "rates": {
                "USD": 1.10,
                "CNY": 7.90,
                "HKD": 8.50,
                "GBP": 0.86,
                "JPY": 180.0,
                "KRW": 1700.0,
                "SGD": 1.45,
                "AUD": 1.60,
                "CAD": 1.55,
                "CHF": 0.95,
                "NZD": 1.95,
            },
        }
        mocked_response.raise_for_status.return_value = None

        with patch.object(app_module.requests, "get", return_value=mocked_response) as mocked_get:
            response = self.client.get("/fx-rates")

        self.assertEqual(response.status_code, 200)
        payload = response.get_json() or {}
        self.assertTrue(payload.get("success"))
        self.assertEqual(payload.get("provider"), app_module.FX_RATE_PROVIDER_NAME)
        self.assertEqual(payload.get("reference_currency"), app_module.FX_RATE_REFERENCE_CURRENCY)
        self.assertIn("EUR", payload.get("rates") or {})
        self.assertEqual((payload.get("rates") or {}).get("USD"), 1.10)
        self.assertEqual(payload.get("source_date"), "2026-03-15")
        self.assertFalse(payload.get("stale"))
        self.assertEqual(payload.get("currencies"), list(app_module.ARB_CALC_SUPPORTED_CURRENCIES))
        mocked_get.assert_called_once()

    def test_fx_rates_uses_stale_cache_if_refresh_fails(self) -> None:
        app_module._FX_RATE_CACHE = {
            "provider": app_module.FX_RATE_PROVIDER_NAME,
            "reference_currency": app_module.FX_RATE_REFERENCE_CURRENCY,
            "currencies": list(app_module.ARB_CALC_SUPPORTED_CURRENCIES),
            "rates": {"EUR": 1.0, "USD": 1.12, "CNY": 7.8},
            "source_date": "2026-03-14",
            "fetched_at": "2026-03-15T00:00:00Z",
            "stale": False,
        }
        app_module._FX_RATE_CACHE_EXPIRES_AT = 0.0

        with patch.object(
            app_module.requests,
            "get",
            side_effect=requests.RequestException("upstream down"),
        ):
            response = self.client.get("/fx-rates")

        self.assertEqual(response.status_code, 200)
        payload = response.get_json() or {}
        self.assertTrue(payload.get("success"))
        self.assertTrue(payload.get("stale"))
        self.assertEqual((payload.get("rates") or {}).get("USD"), 1.12)

    def test_index_renders_currency_select_controls(self) -> None:
        with patch.object(app_module, "_start_background_provider_services") as mocked_start:
            response = self.client.get("/")

        self.assertEqual(response.status_code, 200)
        mocked_start.assert_called_once_with(wait_timeout=0.0)
        html = response.get_data(as_text=True)
        self.assertIn('id="arb-calc-total-currency"', html)
        self.assertIn('id="arb-calc-currency-a"', html)
        self.assertIn('id="arb-calc-fx-bar"', html)
        self.assertIn('id="arb-calc-fx-refresh-btn"', html)
        self.assertIn('id="arb-calc-stake-base-a"', html)
        self.assertIn('id="arb-calc-payout-base-a"', html)
        self.assertIn('id="arb-calc-total-note"', html)
        self.assertIn('id="arb-calc-min-payout-currency"', html)
        self.assertIn('id="arb-calc-profit-currency"', html)
        self.assertIn('<option value="CNY"', html)


if __name__ == "__main__":
    unittest.main()
