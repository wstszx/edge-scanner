from tools import run_batched_dex_scan


def test_build_hunt_args_uses_safe_dex_defaults(tmp_path) -> None:
    args = run_batched_dex_scan.parse_args(["--out-dir", str(tmp_path)])

    hunt_args = run_batched_dex_scan.build_hunt_args(args, stamp="20260527_160000")

    assert "--require-explicit-liquidity" in hunt_args
    assert "--include-all-markets" in hunt_args
    assert hunt_args[hunt_args.index("--provider-sets") + 1] == "all,pairs"
    assert hunt_args[hunt_args.index("--stake") + 1] == "100.0"
    assert hunt_args[hunt_args.index("--min-roi") + 1] == "0.01"
    assert hunt_args[hunt_args.index("--allow-quality") + 1] == "high"
    assert hunt_args[hunt_args.index("--max-quote-skew-seconds") + 1] == "120"
    assert hunt_args[hunt_args.index("--min-executable-stake") + 1] == "25.0"
    assert hunt_args[hunt_args.index("--per-scan-timeout-seconds") + 1] == "35.0"
    assert "basketball_nba" in hunt_args
    assert "basketball_ncaab" in hunt_args
    assert "tennis_atp" in hunt_args
    assert "tennis_wta" in hunt_args
    assert str(tmp_path / "dex_real_opportunity_20260527_160000.json") in hunt_args
    assert str(tmp_path / "dex_real_opportunity_20260527_160000") in hunt_args


def test_main_delegates_to_hunt_with_batch_outputs(monkeypatch, tmp_path) -> None:
    captured = {}

    def fake_hunt_main(argv):
        captured["argv"] = list(argv)
        return 7

    class FakeDateTime:
        @classmethod
        def now(cls):
            return cls()

        def strftime(self, fmt):
            return "20260527_160001"

    monkeypatch.setattr(run_batched_dex_scan.hunt_dex_opportunities, "main", fake_hunt_main)
    monkeypatch.setattr(run_batched_dex_scan, "datetime", FakeDateTime)

    exit_code = run_batched_dex_scan.main(["--sports", "basketball_nba", "--out-dir", str(tmp_path)])

    assert exit_code == 7
    argv = captured["argv"]
    assert argv[argv.index("--sports") + 1] == "basketball_nba"
    assert argv[argv.index("--out") + 1] == str(tmp_path / "dex_real_opportunity_20260527_160001.json")
    assert argv[argv.index("--summary-out") + 1] == str(tmp_path / "dex_real_opportunity_20260527_160001.md")
    assert argv[argv.index("--batch-out-dir") + 1] == str(tmp_path / "dex_real_opportunity_20260527_160001")
