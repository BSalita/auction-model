# BEN settings files of the tests

These are the settings files BEN was started with (`--conf FILE`) in the tests of the main README. They are not
needed to train the models.

Each is BEN's `src/config/GIB-BBO.conf` with a few lines changed, so BEN's license (GPL-3.0) applies to these files,
not this repository's MIT license.

## The two baselines

| File | Lines changed from BEN's `GIB-BBO.conf` |
|------|------|
| `old.conf` | `bidder`, `opponent` and `info` are the networks `GIB-BBO-8730-base` (see "What these files need"); `decl_suit` is the model of 2026-10-16 |
| `old-brillall-noadj.conf` | `old.conf` with `consult_bba`, `use_bba_rollout`, `use_bba_to_count_aces` and `adjust_hcp` set to False, and BEN's `Brill-21GF` networks as `opponent`, `partner`, `opponent_info` and `info`. The baseline of the live tests |

## The tests

Every file is its baseline plus the lines shown, all in `[sampling]`.

| File | Baseline | Lines added |
|------|----------|-------------|
| `old-am-v1.conf` | `old` | `auction_model = /data/runs/v1` |
| `old-am-v1-brill.conf` | `old` | `auction_model = /data/runs/v1-brill` |
| `old-am-guard-v1-brill.conf` | `old` | `auction_model = /data/runs/v1-brill`, `auction_model_guard = 30` |
| `old-guard-v1.conf` | `old` | `auction_model = /data/runs/v1`, `auction_model_deals = False`, `auction_model_guard = 30` |
| `old-guard-v1-brill.conf` | `old` | the same with `v1-brill` |
| `old-truedeal.conf` | `old` | `test_true_deal_file = /var/lib/ben/brill/field.jsonl` |
| `tac.conf` | `old-brillall-noadj` | `auction_model = /data/runs/v1-brill`, `auction_model_deals = False`, `auction_model_tactical = double,sacrifice,interfere,lead`, `auction_model_tactical_trust = 0.2` |
| `tac-even.conf` | `old-brillall-noadj` | as `tac`, with `auction_model_tactical_trust = top` |
| `tac-ds.conf` | `old-brillall-noadj` | as `tac`, with `auction_model_tactical = double,sacrifice`. The setting that was kept |
| `tac-i2.conf` | `old-brillall-noadj` | as `tac`, with `auction_model_tactical = double,sacrifice,interfere2` |
| `tac-d2.conf` | `old-brillall-noadj` | as `tac`, with `auction_model_tactical = double2,sacrifice`; its model path is `/var/lib/ben/auction-model/runs/v1-brill` (it ran on another machine) |
| `tac-s2.conf` | `old-brillall-noadj` | as `tac`, with `auction_model_tactical = double,sacrifice2` |
| `tac-ds-all.conf` | `old-brillall-noadj` | as `tac-ds`, with the model `v1-brill-all` |
| `tac-ds-v3.conf` | `old-brillall-noadj` | as `tac-ds`, with the model `v3-brill` |

## What these files need

- **BEN** with the auction-model keys: BSalita/ben, commit `3038b293` or later on `main`. The test scripts
  (`scripts/match/ab_match.py`, `scripts/match/brill/*.py`) are on its branch `ab-match`.
- **A model of this repository** at the `auction_model` path (a folder holding `model.pt`), this repository's folder in
  the environment variable `AUCTION_MODEL_CODE`, and PyTorch importable by BEN's Python.
- **The networks `GIB-BBO-8730-base`** at `/var/lib/ben/training/GIB-BBO-8730-base/...` (bidding and information
  networks, trained on 2026-11-16 with BEN's training scripts). They are not in BEN's repository or this one. Without
  them, point `bidder`, `opponent` and `info` at BEN's own `GIB-BBO-8744` networks, as in BEN's `GIB-BBO.conf`: the
  tests then run, but against a different baseline than the one in the README.
- The `Brill-21GF` networks and the `decl_suit` model are in BEN's repository (`models/TF2models`).
- Relative paths (`models/...`) are read from BEN's `src` folder.
