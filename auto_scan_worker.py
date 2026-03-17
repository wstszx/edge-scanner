from __future__ import annotations

import logging
import signal
import threading

import app as app_module

_STOP_EVENT = threading.Event()


def _handle_signal(signum, frame) -> None:  # pragma: no cover
    _STOP_EVENT.set()


def main() -> None:
    app_module.ensure_data_dir()
    if not logging.getLogger().handlers:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(levelname)s %(message)s",
        )
    else:
        logging.getLogger().setLevel(logging.INFO)

    for sig_name in ("SIGINT", "SIGTERM"):
        sig = getattr(signal, sig_name, None)
        if sig is not None:
            signal.signal(sig, _handle_signal)

    app_module.configure_runtime()
    app_module._start_background_provider_services()
    app_module._start_server_auto_scan()
    logging.info("Auto scan worker started")
    try:
        while not _STOP_EVENT.wait(1.0):
            pass
    finally:
        app_module._stop_server_auto_scan()
        app_module._stop_background_provider_services()


if __name__ == "__main__":
    main()
