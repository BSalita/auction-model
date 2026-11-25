# Auction model

From one hand and the auction so far (any seat, any point of the bidding) the model gives:

- **Double-dummy tricks**: a probability for 0 to 13 tricks for each declarer and strain (20 distributions).
  The expected DD score of any contract, doubled or not, is computed from these.
- **Par score**: a probability for each par value.
- **Sets**: the joint distribution of the suit lengths and the A, K, Q, J of the three hidden hands.
  Spot cards are not modelled. A full set is all three shapes plus the place of every missing honor;
  the model gives its exact probability. A partial set (for example "LHO has the SA and CQ and is 5-3-3-2,
  partner has the SK") is estimated from sampled full sets.
- **DD tricks given a set**, which is what lets the sampled sets be grouped into the few that matter for a
  choice between contracts.

Purpose: BEN's bidding decisions (the Brill events). Training data so far: BBO robot tournaments
(three GIB robots and one human at South).

## Reproducing the models

The trained models and the data they were trained on are **not in this repository**. This section lists what is
needed and the commands that turn it into each model. `v3-brill` is the model in use.

### 1. Software

- Python 3.12 and `pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu128`
  (PyTorch 2.11.0 for CUDA 12.8; the versions used are in `requirements.txt`). Into an empty folder this took
  11 minutes and 7.3 GB.
- A GPU. The models were trained on an RTX 5080 (16 GB); training held 10.3 GB at its peak.
- Nothing else: no environment variable or other setting is needed for the commands below.

This whole section was followed as written on 2026-11-20, from the three sources to the trained `v3` and
`v3-brill`, with a fresh install. The results are under "Check the result".

### 2. Data

Three sources. Nothing else is read.

| Source | Files | Needed for | Where it comes from |
|--------|-------|------------|---------------------|
| BBO robot tournaments, as a table | `BBO\data\bbo_mldf_augmented.parquet`: one row per table played, with the columns `PBN`, `Dealer`, `Vul`, `bid` (the list of calls), `Par_NS` and the 20 double-dummy columns `DD_N_S` .. `DD_W_N` (declarer, strain) | every model | Built by the author from downloaded BBO hand records (events with one human at South and three GIB robots). Not public |
| The same events as raw downloads | `BBO\data\<player>\*.lin` | `v3`, `v3-brill` (the boards dealt by North, which the table lacks) | Downloaded from BBO. Not public |
| Brill daily events | `BRILL\field.jsonl` (one line per event and board: deal, dealer, vulnerability, date, every table's auction) and `BRILL\board-table\dd.json` (double-dummy tricks and par of every deal) | the `-brill` models and all checks on Brill deals | Read from brill.aalborgdata.dk with `scripts/match/brill/fetch_brill_field.py` (needs a Brill sign-in) and `brill_board_table.py`, both on branch `ab-match` of BSalita/ben. The models used the events of 15 August to 17 November 2026 |

In `dd.json` a deal's key is `deal|dealer|vul` as written in `field.jsonl`; its value holds `tricks` (20 numbers
keyed strain + declarer, `SN` = spades by North, `NE` = notrump by East) and `par_score` (North-South's).

Three folders are named by environment variables (`paths.py`; the defaults are the author's machine):

| Variable | Folder |
|----------|--------|
| `AUCTION_MODEL_BBO` | `BBO` above |
| `AUCTION_MODEL_BRILL` | `BRILL` above |
| `AUCTION_MODEL_DATA` | `DATA`: where the training files (`*.npz`) and the models (`runs\NAME\model.pt`) go |

### 3. Build the training files

Run in this folder. `DATA` is the folder of `AUCTION_MODEL_DATA`.

| Step | Command | Writes to `DATA` |
|------|---------|------------------|
| 1 | `python extract_bbo.py` | `bbo_train.npz` (4,992,310 rows), `bbo_test.npz` (108,763 rows: the held-out deals) |
| 2 | `python extract_brill.py` | `brill.npz` (10,197 rows with the events to 18 November 2026) |
| 3 | `python split_brill.py --data DATA --until 20261117` | `brill_train.npz` (8,108 rows, to 5 November), `brill_test.npz` (1,938 rows, 6 to 17 November), `brill_old.npz` (both together), `brill_new.npz` (later events), `mix_train.npz` |
| 4 | copy `bbo_test.npz` to `bbo_heldout.npz` | a second name for the held-out BBO deals, so that a check does not overwrite a run's `eval.txt` |
| 5, for `v3` only | `python extract_bbo_lin.py scan --out DATA\lin --share 0.3`, then `python extract_bbo_lin.py dd --out DATA\lin`, then `python extract_bbo_lin.py build --out DATA\lin` | `bbo_north_train.npz`, `bbo_north_test.npz`, `bbo4_train.npz` (5,918,594 rows), `bbo4_test.npz` (128,603 rows) |

- Step 5: `--share 0.3` reads 30% of the player folders (16,529 of 55,098; the same ones every time), which is what
  the models were trained on: 946,865 rows on 344,238 deals. `scan` took 40 to 46 minutes and `dd` 25 minutes, with
  56 processes each (`--workers`; Windows allows at most 61).
- The held-out BBO deals are picked by polars' `hash`; use the polars version of `requirements.txt` to get the
  same deals.
- Rebuilt this way on 2026-11-20, all thirteen files are equal, array for array, to the ones the models were
  trained on.

### 4. Train

`/data` below is the `DATA` folder. Every setting not shown is `train.py`'s default (batch 768, learning rate 4e-4,
1,000 warm-up steps, width 256, 6 + 4 layers, seed 1).

| Model | Command | Time on the RTX 5080 |
|-------|---------|----------------------|
| `v1` (BBO table) | `python train.py --data /data --run /data/runs/v1 --epochs 3` | 50 minutes |
| `v1-brill` | `python train.py --data /data --run /data/runs/v1-brill --init /data/runs/v1/model.pt --train-file brill_train.npz --test-file brill_test.npz --batch 128 --lr 3e-5 --warmup 20 --epochs 2 --eval-rows 0` | under a minute |
| `v3` (BBO table + North-dealer boards) | `python train.py --data /data --run /data/runs/v3 --epochs 3 --train-file bbo4_train.npz --test-file bbo4_test.npz` | 60 minutes (76 with other work on the machine) |
| `v3-brill` | `python train.py --data /data --run /data/runs/v3-brill --init /data/runs/v3/model.pt --train-file brill_train.npz --test-file brill_test.npz --batch 128 --lr 3e-5 --warmup 20 --epochs 2 --eval-rows 0` | under a minute |
| `v2-big` | `python train.py --data /data --run /data/runs/v2-big --epochs 3 --batch 384 --lr 3e-4 --d 384 --enc 8 --dec 6` | 109 minutes |
| `v1-brill-old`, `v1-brill-all` | `v1-brill`'s command with `--train-file brill_old.npz` or `brill.npz`, and `--test-file brill_new.npz` | under a minute |

The shell scripts run the same commands on the author's machine: `ft4.sh` (`v3`, `v3-brill` and their checks),
`ft3.sh` (`v1-brill-old`, `v1-brill-all`), `ft.sh` and `ft2.sh` (the trials `v1-brill`'s settings were chosen from).
They expect this folder at `/code`, the data at `/data` and PyTorch in `/var/lib/ben/torch-pkgs`: edit their first
lines for another machine.

### 5. Check the result

When training ends, `train.py` writes the held-out table for its `--test-file` to `runs\NAME\eval.txt`. For a
table on another file (written to `runs\NAME\eval_brill_test.txt` here), and a self-test of the inference code:

```
python train.py --data /data --run /data/runs/NAME --eval-only --test-file brill_test.npz --eval-rows 0
python infer.py --run /data/runs/NAME --selftest
```

What the author's models give, in the block "South's viewpoint" of that table (rows `0 calls` and `auction over`):

| Model, held-out file | Error of expected tricks (`DD mae`) | Error of expected par (`par mae`) |
|---|---|---|
| `v1`, `bbo_test.npz` (its `eval.txt`) | 1.727 -> 1.136 | 364.3 -> 247.5 |
| `v1-brill`, `brill_test.npz` (its `eval.txt`) | 1.86 -> 1.19 | 414 -> 253 |
| `v3`, `bbo4_test.npz` (its `eval.txt`) | 1.721 -> 1.129 | 357.1 -> 242.5 |
| `v3-brill`, `brill_test.npz` (its `eval.txt`) | 1.855 -> 1.175 | 411.9 -> 246.4 |

**A second training comes close, not to the same model.** `v3` and `v3-brill` were trained again on 2026-11-20
with the commands above, from rebuilt files and a fresh install:

| Model, held-out file | Rows | Tricks, first run | Tricks, second run | Par, first run | Par, second run |
|---|---|---|---|---|---|
| `v3`, `bbo4_test.npz` | 128,603 | 1.721 -> 1.129 | 1.721 -> 1.130 | 357.1 -> 242.5 | 357.1 -> 242.5 |
| `v3`, `bbo_north_test.npz` | 19,840 | 1.700 -> 1.115 | 1.700 -> 1.115 | 322.3 -> 219.4 | 321.9 -> 219.9 |
| `v3`, `brill_test.npz` | 1,938 | 1.874 -> 1.181 | 1.874 -> 1.182 | 420.0 -> 251.7 | 420.6 -> 251.9 |
| `v3-brill`, `brill_test.npz` | 1,938 | 1.855 -> 1.175 | 1.855 -> 1.177 | 411.9 -> 246.4 | 412.3 -> 246.6 |
| `v3-brill`, `bbo_north_test.npz` | 19,840 | 1.691 -> 1.122 | 1.691 -> 1.121 | 322.2 -> 221.2 | 321.4 -> 221.6 |
| `v3-brill`, `bbo_heldout.npz` | 108,763 | 1.721 -> 1.140 | 1.721 -> 1.140 | 363.6 -> 248.4 | 363.2 -> 248.3 |

- Over every row of those tables (all six auction lengths): on the BBO files the two runs differ by at most 0.002
  tricks and 0.8 points of par; on the 1,938 Brill rows by at most 0.011 tricks and 3.6 points.
- The training losses of the two `v3` runs are equal to the last printed digit for the first 200 steps and differ
  in the third or fourth decimal from then on (at step 24,000 of 24,075: 1.9039 against 1.9039 for the tricks, 2.6104 against 2.6105 for par).
  The weights of the two models are therefore not the same, and the BEN tests of the later sections were not
  repeated with the second model.
- `v1` was stopped at step 7,000 of 20,301 and continued (`train.py --resume`), so it is not one uninterrupted run;
  `v3` is.

### Tests inside BEN

The later sections measure the models in BEN's bidding. Those need more than the models:

- BEN: BSalita/ben, commit `3038b293` or later on `main` (the model's keys in the settings file). The test scripts
  (`scripts/match/ab_match.py`, `scripts/match/brill/brill_replay.py`, `brill_live.py`, `score_brill_replay.py`,
  `brill_autoplay.py`) are on its branch `ab-match`.
- The settings files of every test: the folder `conf` here, with what each one changes and needs (`conf/README.md`).
- The live tests ask Brill's public bidding service for the other three seats' calls, so they depend on that
  service and on the robots it runs that day.

## Files

| File | What |
|------|------|
| `extract_bbo.py` | BBO robot file -> `bbo_train.npz`, `bbo_test.npz` (one row per distinct deal + auction). Windows, `bbo-bidding\.venv` |
| `extract_brill.py` | Brill daily events (`src\data\brill-replay`) -> `brill.npz`, same layout. South is the entrant, the other three are the Brill robots |
| `model.py` | The network, input and target building, legal-choice masks, the set sampler |
| `train.py` | Training and the held-out tables (`--eval-only`, `--test-file`) |
| `eval_sets.py` | Are the probabilities honest: predicted probability against how often true, on held-out deals |
| `infer.py` | `Predictor` / `Position`: DD table, contract scores, par, set lists, partial-set probability, sets that matter |
| `hypothesis.py` | One position: P(a hypothesis about the hidden hands), contracts' value with and without it, value of asking bids |
| `keycards_ask.py` | Value of RKCB at the game-or-slam decisions of `keycards.py`, on the real Brill deals |
| `scoring.py` | Duplicate score table |
| `lead_forecast.py` | Explorer page: tricks after the anticipated opening lead (deals drawn from the model, BEN's lead network and a double-dummy solve of every lead, in a helper process) |
| `post.sh` | Runs the held-out checks when a training run ends |
| `split_brill.py` | Brill deals by date -> `brill_train.npz`, `brill_test.npz`, and the BBO + Brill `mix_train.npz` |
| `ft.sh`, `ft2.sh` | The Brill fine-tuning trials (`train.py --init RUN/model.pt --train-file ...`) |
| `bench.py`, `bench2.py` | Step-time measurements |
| `paths.py` | The data folders of the Windows-side scripts and the environment variables that change them |
| `requirements.txt` | The package versions used |
| `conf\` | The BEN settings files of the tests, and `conf\README.md` on what each changes |

Data and runs are outside this folder: on the author's machine `E:\bridge\data\auction-model` (`runs\<name>\model.pt`,
`train.log`, `eval*.txt`, `sets_*.txt`); elsewhere the folder of `AUCTION_MODEL_DATA`.

## Running on the author's machine

(For another machine see "Reproducing the models" above.) Training needs PyTorch, which is in the `ben-state`
volume. Use a container of its own:

```
wslc run -d --rm --name auction-model --gpus all -m 48G -v ben-state:/var/lib/ben -v "D:\OneDrive\sw\bridge\ML-Contract-Bridge\src\auction-model:/code" -v "E:\bridge\data\auction-model:/data" -w /code --entrypoint sleep ben:latest infinity
```

Inside it (`export PYTHONPATH=/var/lib/ben/torch-pkgs`):

```
python train.py --data /data --run /data/runs/NAME --epochs 3
python train.py --data /data --run /data/runs/NAME --eval-only --test-file brill.npz
python eval_sets.py --run /data/runs/NAME --south-only
python infer.py --run /data/runs/NAME --hand AK52.KQ9742.Q.K2 --seat S --dealer S --vul E_W --auction "1H p 1S p" --contracts "4S:N 4H:S 3N:N"
python infer.py --run /data/runs/NAME --selftest
```

## Conventions

- Cards: index = suit*13 + rank, suits S,H,D,C, ranks A,K,Q,J,T,9..2. Seats N,E,S,W = 0..3.
- Calls: `p`, `d`, `r`, `1C`..`7N` (also `P`, `X`, `XX` in `infer.py`).
- Inside the network everything is relative to the viewpoint seat: me, LHO, partner, RHO.
- Held-out BBO deals: `PBN.hash(seed=20261117) % 47 == 0`, the same rule as the real-GIB bidding networks.
- A set is generated in this order: spade, heart, diamond lengths of LHO and partner (the rest follows),
  then the 16 honors SA..CJ. Choices that cannot be completed to a legal deal are masked out, so every
  sampled set is a legal layout and the probabilities sum to one over legal sets.

## Results (2026-11-18)

Two runs, 3 passes over `bbo_train.npz` each: `v1` (8.3M parameters, d 256, 6 + 4 layers, 50 minutes of training on a
quiet RTX 5080) and `v2-big` (31M, d 384, 8 + 6 layers, batch 384, 109 minutes). Held-out, South's viewpoint, before
any call -> after the whole auction (`eval.txt`, `eval_brill.txt`, `sets_*_south.txt` in each run folder):

| Measure | v1, BBO | v2-big, BBO | v1, Brill | v2-big, Brill |
|---|---|---|---|---|
| Error of expected DD tricks | 1.73 -> 1.14 | 1.73 -> 1.13 | 1.87 -> 1.21 | 1.88 -> 1.21 |
| Error of expected par score | 364 -> 248 | 364 -> 245 | 418 -> 271 | 419 -> 272 |
| Most likely seat of a missing honor is right (sampled sets; chance 33%) | 33% -> 47% | 33% -> 47% | 33% -> 47% | 33% -> 47% |
| Expected suit length off by (cards) | 1.05 -> 0.76 | 1.06 -> 0.75 | 1.03 -> 0.76 | 1.03 -> 0.75 |
| DD tricks exactly right given the true set | 68% | 71% | 68% | 71% |
| Expected DD score of the final contract off by, auction over | 208 | 202 | 203 | 203 |

- Size is not the limit: 3.7 times the parameters changed the direct outputs by about 1%. `v1` is the working model.
- On BBO deals the probabilities are honest (an honor given 80-90% is there 83-84% of the time; a final contract
  given 90-100% to make makes 94.5%). On Brill deals they are overconfident at both ends (80-90% -> 79-80%,
  under 10% -> 10%).
- Joint probabilities carry real structure: "this seat has the honor and k+ cards of its suit" averages 22.9%
  predicted, 22.9% true, against 20.4% if the two parts are multiplied.
- The `hon acc` column of `eval.txt` is measured with the true shapes and earlier honors given (48% -> 58%); it is
  not the accuracy from the auction alone, which is the 33% -> 47% above.
- "Final contract makes" before the auction is over is not a fair test: the contract is chosen by calls the model
  has not seen yet, so contracts it rates unlikely still make often. Only the auction-over rows count.

Training data limits found:

- The BBO source file has no board dealt by North (boards 1, 5, 9, 13 are absent), so from South's seat the model has
  never seen partner deal. On Brill boards dealt by North the honor loss is 10.27 against 9.79 on the others
  (auction over); DD tricks are no worse.
- BBO robot events give South the best hand (99.3% of deals; mean 15.3 points against 8.0-8.5). In the Brill dailies
  South is best on 69% and has under 10 points on 17% of deals (0.4% in BBO). Before any call on Brill deals the model
  sometimes gives one seat over 50% for an honor when the truth is one in three.

### Fine-tuning on Brill deals: `v1-brill`

`split_brill.py` splits `brill.npz` by date: `brill_train.npz` (8,108 rows, 15 August to 5 November 2026) and
`brill_test.npz` (1,938 rows, 6 to 17 November; no deal on both sides). `ft.sh` and `ft2.sh` hold the eleven trial
runs, all starting from `v1` (`train.py --init`). Kept: `runs\v1-brill` = Brill deals only, learning rate 3e-5,
2 passes, batch 128 (132 steps, under a minute). The trials' logs are in `runs\ft-trials`.

South's viewpoint on `brill_test.npz`, before any call -> auction over:

| Measure | `v1` | `v1-brill` |
|---|---|---|
| Error of expected DD tricks | 1.92 -> 1.21 | 1.86 -> 1.19 |
| Error of expected par score | 437 -> 262 | 414 -> 253 |
| Set loss (shape + honors) | 20.19 -> 16.61 | 19.88 -> 16.42 |
| Honors given over 50% before any call (truth is one in three) | 969 of 63,411 | 12 |
| Honor given 80-90% is there, auction over | 76.6% | 81.1% |
| Honor given under 10% is there, auction over | 10.1% | 8.6% |
| DD tricks exactly right given the true set | 68.2% | 67.9% |

- One or two passes at a low rate is all the 8,108 rows carry: at 1e-4 every pass after the first made the held-out
  results worse (12 passes: DD error auction over 1.29, DD given a set 61%).
- Mixing in 200,000 BBO rows (`mix_train.npz`) gained less than Brill alone; dropout 0.1 hurt.
- The recipe was picked on the same 1,938 held-out rows, so its edge over the other short runs is not established;
  the gain over `v1` is common to all of them.
- `v1-brill` has not seen the newest 12 days. A model for use would repeat the recipe on all of `brill.npz`.

## Wiring into BEN's bidder (2026-11-18)

In `src\ben` (BSalita/ben, on `main` since commit `3038b293`): new file `src\auction_model_deals.py`, three optional keys read in
`Sample.from_conf`, and two short branches in `botbidder.py`. With no key set BEN behaves as before.

```
[sampling]
auction_model = /data/runs/v1-brill     # run folder holding model.pt
auction_model_deals = True              # the search's deals come from the model (default when auction_model is set)
auction_model_guard = 30                # passout seat: add a candidate when a contract of ours is expected to
                                        # score this many points more than defending their partscore (empty = off)
test_true_deal_file = /var/lib/ben/brill/field.jsonl   # measurement only: every sample is the real deal
```

The process needs `AUCTION_MODEL_CODE` (this folder) and PyTorch (`/var/lib/ben/torch-pkgs` is added if `import torch`
fails). The model runs on the CPU with one thread: drawing 50 deals takes 0.2 s on a free core.

Scripts here: `ben_samples.py` (BEN's own deals at every South turn of a data file), `sample_check.py` (those deals
and the model's against the true deal), `ben_adapter_test.py` (both configs side by side on a few boards),
`replay_compare.py` (scored replays against a baseline on the same boards). Replays: `ben\scripts\match\brill\brill_replay.py`
in a container with `src\ben` at `/ben`, this folder at `/code` and the data at `/data`; configs in
`/var/lib/ben/am/conf`, replays in `/var/lib/ben/am` and `src\data\brill-replay\runs\am`, scores in
`src\data\brill-replay\am-report`.

**Step 0, deals against the truth** (5,100 South turns of `brill_test.npz`, BEN config `old` = `GIB-BBO-8730-base`,
50 deals each; `runs\v1-brill\sample_check_ben_samples_old.txt`):

| Measure | BEN's deals | Model's deals |
|---|---|---|
| Honor seat error (Brier) | 0.2160 | 0.2119 |
| Honor given under 10% is there | 15.7% | 11.3% |
| Suit length off by (cards) | 0.873 | 0.855 |
| True length in none of the deals | 5.4% | 1.5% |
| Final contract given 80-100% to make does make | 81% | 87% |
| South's third or later turn: points of a hidden hand off by | 1.55 | 1.42 |
| South's third or later turn: final contract tricks off by | 1.05 | 0.96 |

At South's first turn the two are level (BEN's deals are a little better on points and tricks); the model's edge
grows with the auction. BEN returned no deal on 59 of 5,159 turns and fewer than 15 on 2.8%.

**The bbo-bidding table as the deal source** (`bt_check.py`, 2026-11-18, same 5,159 turns; `bt_check.txt` and
`bt_samples.npz` in the data folder; Windows, `bbo-bidding\.venv`, 46 s). The auction so far is looked up in
`bbo_bt_seat1.parquet` (461M auction nodes, hand requirements per seat), the 39 unseen cards are dealt at random and
a deal is kept when the three hidden hands meet the requirements. The custom overlay of bbo-bidding is not applied.

- Of the 4,058 turns with a bid made, 95.5% of the auctions are in the table (87% at South's third or later turn);
  50 deals on 94% of those, none on 2%. A random deal meets the requirements 2.3% of the time (median).
- All three true hidden hands meet the table's requirements on 63% of turns in the table: 84% at South's first
  turn, 68% at the second, 44% at the third or later. Partner (a Brill robot, not GIB) fits least: 79% overall,
  62% at the third or later turn. Most failures are lower bounds on points.
- Against the true deal, on the 4,844 turns where both gave deals (random deals in brackets): honor seat error
  0.2150 against BEN's 0.2165 (0.2267), points of a hidden hand off by 1.85 against 1.79 (2.70), suit length off by
  0.877 against 0.875 (1.036), true length in none of the deals 2.9% against 5.0%. Level with BEN's own deals,
  wider, and behind the auction model (0.2119, 0.855, 1.5%). Where the true hands fit the table its deals beat BEN's
  (0.2098 against 0.2138); where they do not, points are off by 0.24 more.

So the table is a usable deal source but not a better one than BEN's sampler on Brill auctions, and the replay
results below say a better one would not show anyway. Not tested: the table as the bidder of the rest of the auction.

**Step 1, the model as the deal source** (2,556 Brill boards, same boards as the baseline replay `runs\old.jsonl`,
51.77%): `v1` 51.38% (-0.38 +-0.53 points, -0.05 +-0.09 IMPs), `v1-brill` 51.39% (-0.37 +-0.52). The auction
changes on 11% of boards, about as many as between two training runs of one recipe, with as many boards worse as
better. Better deals do not show as better bidding. (`old` is the luckiest of nine sibling runs on these boards,
so any change to it tends to score a few tenths lower.)

**Step 2, the par guard** (`auction_model_guard = 30`, BEN's own deals): South's pass ended the auction on 406 of
the 2,556 boards, 193 of them in the other side's undoubled partscore with a bid of ours available at the 3 level or
lower. The model expected 30 or more points extra from bidding on 21 (`guard_count.py`). The search took the proposed
call on none; one board changed, for the worse, because an extra candidate shifts the comparison between the others.
No effect.

**Headroom: the real deal as every sample** (`test_true_deal_file`, measurement only): 53.78% against 51.77%,
+2.02 +-0.69 points and +0.47 +-0.12 IMPs a board; the auction changes on 450 boards (17.6%), better on 241 and
worse on 132. So perfect knowledge of the hidden honors, tens and shapes is worth 2 points through BEN's search,
and the step from BEN's deals to the model's covers a few percent of the distance from BEN's deals to the truth
(honor error 0.216 -> 0.212, truth 0): a gain of the order of 0.05 points, far below what the replay can see
(0.7). Most of the 2 points is information the auction does not carry, not information the search misuses.
That is the reason step 3 (a call-value output replacing the rollouts) was not built: it can only use the same
information better. The 132 boards that got worse with the real deal show the judging itself errs (EPBot is not
the Brill robots, double dummy is not the play).

## Key cards (2026-11-18)

`keycards.py`: at South's turns of the Brill auctions, which 1 to 5 cards or suit lengths of the hidden hands decide
between two contracts of our side (partial or game, game or slam, suit game or 3NT), how likely the model says they
are there, and whether that is right on the real deals. No change to BEN. The docstring has the definitions.
For each decision 600 sets are drawn from the model, completed to deals and solved double dummy (DDS 3 of the BEN
image, 41 ms a solve); the key set is chosen on the first third, kept only if it changes the choice on the second
third, and measured on the last third. Run in a container of the BEN image (`export PYTHONPATH=/var/lib/ben/torch-pkgs`):

```
python keycards.py --run /data/runs/v1-brill --data /data/brill_test.npz --tag test        # 11 min, 110 processes
python keycards.py --run /data/runs/v1 --data /data/brill_train.npz --tag train-v1         # 34 min, 120 processes
python keycards.py --tag test --stage report                                               # the report again, seconds
```

Output in `E:\bridge\data\auction-model\keycards\<tag>`: `report.txt`, `decisions.jsonl` (one line per decision:
hand, auction, the two contracts, key set, probabilities, the real deal), `states.npz`, `tricks.npz`.
A decision counts once per auction, at South's last turn where the choice is still available and partner has bid
or doubled, if the model's odds leave it open there. Numbers are weighted by tables.

| | `test`: 384 boards of 6-17 November, `v1-brill` | `train-v1`: 2,172 earlier boards, `v1` |
|---|---|---|
| Boards with a decision at some table (partial/game, game/slam, suit/3NT, any) | 47%, 19%, 38%, 54% | 43%, 19%, 33%, 53% |
| IMPs between the two contracts on the real deal | 6.3, 9.5, 5.0 | 6.3, 9.6, 5.0 |
| Key set found; items in it | 65%, 83%, 57%; 1.6, 2.0, 1.4 | 72%, 83%, 53%; 1.6, 2.0, 1.5 |
| Key set is there, said -> true | 43 -> 43, 34 -> 41, 44 -> 46% | 39 -> 37, 32 -> 27, 41 -> 40% |
| Higher contract makes with the set, said -> true | 62 -> 71, 68 -> 55, 42 -> 56% | 59 -> 60, 62 -> 55, 38 -> 40% |
| Higher contract makes without it, said -> true | 31 -> 31, 27 -> 21, 36 -> 35% | 31 -> 36, 28 -> 21, 33 -> 36% |
| IMPs a decision gained by knowing the set, over choosing by the odds | +0.26 +-0.32, +1.09 +-1.26, +0.11 +-0.34 | +0.26 +-0.18, +0.40 +-0.49, +0.37 +-0.16 |
| The same per board over all boards | +0.26 +-0.22 | +0.22 +-0.10 |
| Ceiling: knowing the double-dummy result, per board over all boards | +1.16 +-0.28 | +1.38 +-0.14 |

(For suit game or 3NT "makes" reads "the suit game scores more than 3NT".)

- The finder is right in direction and its probabilities are close: the set it names is there about as often as
  said, and the higher contract makes 55-71% of the time with it against 21-36% without (partial/game, game/slam).
  For suit or 3NT the split is weak on the larger sample (40% against 36%).
- Slam chances from the sampled deals are too high (said 40%, true 31-34%).
- Knowing for certain whether the key set is there is worth about 0.22 IMPs a board, a sixth of what knowing the
  double-dummy result is worth. A real asking bid gets a less exact answer and uses bidding room, so its gain is
  smaller than that. The baseline is the model choosing by its odds, not BEN's bidding; everything is double dummy.
- Sets are made of partner's suit lengths (33%), counts of aces, kings and top trumps (27%), a named honor with
  partner (20%) or with one opponent (12%), partner's shortness (8%). Tens are not modelled.

### What the entrants did at these decision points

`keycards_entrants.py` (Windows, `bbo-bidding\.venv`, seconds; output `keycards\entrants.txt` in the data folder)
matches every table of `field.jsonl`, and our replay `runs\old.jsonl`, to its row and reads from the final contract
whether the table went to the higher or the lower contract of each decision. Tables that ended elsewhere (another
strain, the other side declaring) are left out: 40% of partial-or-game decisions, 38% of game-or-slam, 24-27% of
suit-or-3NT. Both board sets together, double dummy, IMPs lost to the right one of the two contracts:

| | Ours (`old`, 2,395 boards) | BEN | BBOBen | By the odds | Key set known |
|---|---|---|---|---|---|
| Partial or game: went to game; IMPs lost | 53%; 1.98 | 53%; 2.04 | 55%; 2.03 | 66%; 2.17 | 1.90 |
| Game or slam: went to slam; IMPs lost | 17%; 3.45 | 15%; 3.45 | 16%; 3.73 | 37%; 4.08 | 3.07 |
| Suit game or 3NT: IMPs lost | 0.89 | 0.90 | 0.93 | 0.80 | 0.54 |
| All three, per board over all boards: lost to the odds | -0.07 +-0.10 | -0.00 +-0.10 | -0.03 +-0.12 | | |
| All three, per board: lost to the key set known | +0.11 +-0.10 | +0.11 +-0.11 | +0.11 +-0.13 | | |

("By the odds" and "key set known" are on our tables; they differ a little by entrant.)

- No entrant with 1,000+ boards differs from the by-odds choice by more than its interval, except Pidgin (worse).
  Ours, BEN and BBOBen are level with each other: same board and decision, ours minus BBOBen +0.07 +-0.24 IMPs a
  decision (571), BEN minus BBOBen +0.20 +-0.23 (604), ours minus BEN -0.04 +-0.12 (947). These decisions are not
  where BEN loses to BBOBen.
- Partial or game: the real auctions already do what the key set would. The tables are right more often than the
  odds (69% against 62% for ours) because an invitation lets North decide with its own cards; knowing the key set
  would add 0.08 IMPs a decision to ours.
- Game or slam: the odds bid far too many slams (37% against the tables' 15-17%), the miscalibration noted above.
- Suit game or 3NT: nearly every entrant is a little worse than the odds (ours +0.09 +-0.22, BEN +0.21 +-0.24) and
  the key set would add 0.35 IMPs a decision: the one place with room, about 0.06 IMPs a board.
- The most that knowing the key set for certain could add to our bidding is about 0.1 IMPs a board.

## Hypotheses and asking bids (2026-11-22)

`hypothesis.py` answers the bidder's own question at one position: "slam makes if partner has DA and CK: how likely
is that, and should I bid it, sign off, or ask?" The model draws sets for the hand and auction (all calls seen,
partner's and the opponents'), each set is completed to a deal and solved double dummy for the candidate contracts
(1,000 deals in a few seconds with 16 solver processes; `--solver model` uses the model's trick odds given each set
instead). The docstring has the hypothesis syntax (honors, suit lengths, counts of aces or keycards, points; "or"
between alternatives).

```
python hypothesis.py --run /data/runs/v3-brill --hand AKQJ7.AKQ2.43.A5 --dealer S --vul None --auction "1S p 2S p" \
    --contracts "4S 6S" --if "partner: DA CK" --if "partner: DA" --ask rkcb:S --ask "partner: DA CK @5S"
```

- `--if H`: P(H); each contract's expected score and chance of making on all deals, when H holds and when it does not;
  the best contract each way; for every other contract what it gains each way and the P(H) at which it becomes
  right; what knowing H for certain is worth over choosing by the odds; knowing the whole deal as the ceiling.
- `--ask`: `rkcb:S` (1430), `rkcb0314:S`, `aces:S` (plain 4NT), or a hypothesis as an ideal yes/no question with
  the cheapest contract left after the answer (`"partner: DA CK @5S"`). Per reply: its chance, P(H) after it, the
  best contract still reachable (5 of the suit added as the sign-off), and asking then placing the contract against
  bidding now. `--scoring imps|mp|total`; `--json` writes the results.

The example (dealer South, none vulnerable, partner raised 1S to 2S): P(partner has DA and CK) = 11%; if so 6S
makes 99% and gains 10.8 IMPs over 4S, if not it loses 1.9, so 6S is right above 15% and the odds say 4S; knowing
it for certain is worth +1.2 IMPs. RKCB is worth +1.2 here (the 5C reply makes slam 70%, 5D leaves 5S), while the
ideal "DA and CK?" question with 5S as the floor gains only +0.07: the yes is too rare to pay for the level. After
1S p 3S p the same hand bids 6S by the odds (86%) and no question changes it.

**On the real deals (`keycards_ask.py`, seconds, from keycards.py's saved sets of `v1-brill` / `v1`).** The
game-or-slam decisions of keycards.py in a suit with the auction below 4NT; IMPs a decision against choosing by the
odds now, double dummy, test boards / earlier boards (output `keycards\ask.txt` in the data folder):

| | `test` (81 decisions, 32 boards) | `train-v1` (406 decisions, 204 boards) |
|---|---|---|
| RKCB every time, then 5 or 6 by the odds after the reply | -0.84 +-2.74 | -1.10 +-1.25 |
| the reply known, game still possible | +0.98 +-1.87 | +0.97 +-0.84 |
| 5 instead of game, no reply (cost of the room) | -1.28 +-1.13 | -2.24 +-0.74 |
| RKCB only where the drawn deals say it pays (asked on 23% / 28%) | +0.46 +-0.49 | +0.21 +-0.54 |
| what the drawn deals said that would gain | +0.11 | +0.17 |
| keycards.py's key set known for certain | +0.56 +-1.66 | +0.74 +-0.95 |
| the right contract known (ceiling) | +4.43 +-2.41 | +4.21 +-0.96 |
| replies said -> true: 1 or 4, 0 or 3, 2 no queen, 2 and queen | 35/34/9/22% -> 49/16/2/33% | 45/22/10/23% -> 44/19/9/28% |

- The reply odds are honest on the larger sample (within 5 points each).
- The keycard reply as information is worth about 1 IMP a decision, as much as the best key set, but the room it
  takes costs more when the answer is "stay low": asked every time it loses 1 IMP a decision.
- Asked only where the tool says it pays, it gains +0.2 to +0.5 IMPs a decision (not clear of zero), close to what
  the drawn deals predicted. These decisions come up on about 9% of boards, so over all boards the gain is small.
- Limits: double dummy; one keycard step (no queen ask, no king ask, no cue bids); partner's reply read from the
  real hand, not from a robot; the opponents' calls count as much as the model learned they tell, with no setting to
  trust them less; slam odds from sampled deals ran high on Brill boards (above).

## Calls by purpose (2026-11-18)

The user's point: at every turn a bidder asks whether a call moves toward a contract, interferes with a game or
slam of the other side, invites a sacrifice or pushes them too high, or directs partner's lead, and such a call
is made even when no system prescribes it. BEN cannot: its search only sees calls its network rates above a
threshold, plus EPBot's call. `tactical.py` names such calls from the model's view (definitions and thresholds in
its docstring): `toward`, `interfere`, `sacrifice` (tags `sacrifice` and `push`), `double`, `lead`.
Not wired into BEN; this is a count only.

`tactical_count.py turns` runs BEN's candidate step (no search) and the rules at every South turn of a replay;
`tactical_count.py report` prints the table, the checks on the real deal and examples. Container as for the
replays, with `BEN_ROOT=/ben AUCTION_MODEL_CODE=/code PYTHONPATH=/var/lib/ben/torch-pkgs` and
`src\data\brill-replay` at `/replay`; 24 workers, under a minute a file. Output in
`E:\bridge\data\auction-model\tactical` (`brill-old.jsonl`, `live-old.jsonl`, `dd_brill.json`), reports also in
`src\data\brill-replay\tactical-report`. Config `old`, model `v1-brill`.

| Rule | Live 10,000 deals: boards, calls named, already a BEN candidate | Brill 2,556 boards (every auction South faced in the field) |
|---|---|---|
| `toward` | 5,085 boards, 10,660 calls, 42% | 1,718 boards, 6,856 calls, 45% |
| `interfere` | 543, 987, 24% | 102, 227, 25% |
| `sacrifice` | 531, 697, 58% | 275, 540, 59% |
| `double` | 637, 666, 29% | 290, 393, 44% |
| `lead` | 185, 191, 44% | 53, 59, 54% |

- BEN's network rates 81-93% of the calls it lacks under 1% (by rule), so lowering the threshold would not bring
  them in.
- On the real deal (double dummy) the triggers hold: "they have a game" said 63%, true 73-76%; the interfering
  bid doubled costs no more than their best contract on 80-88%; a named double's contract fails 67% of the time
  on the live deals (said 74%) and 76% on the Brill boards (said 77%); a named bid over their contract, doubled,
  beats defending on 63-73%.
- Where BEN's pass ended its own auction, the named call ending it instead is worth, double dummy, over all
  boards: double +0.040 IMPs a board (live) and +0.047 (Brill), bid over their contract +0.067 and +0.054, push
  +0.026 and +0.020. These overlap (one turn, one call), assume nobody bids again and that partner reads the
  double as penalty, and are double dummy: an upper estimate, not a result.
- `interfere` and `lead` cannot be valued this way: their gain is in what the other side or partner does next.
- `toward` as written is too broad (a third of all turns): it is "bid the contract with the best expected
  score", and most of its new calls break the system in uncontested auctions (1NT for a 1H opening). To be
  narrowed or dropped before any wiring.
- Right-hand opponent's 2C opening in front of South, live deals: 45 times, South with a 6+ suit 12 times, BEN
  passed 7 of them, `interfere` named a call on 6, none of them a BEN candidate.

### Wired into BEN as extra candidates

In `src\ben` (BSalita/ben, on `main` since commit `3038b293`; off unless the keys are set): `auction_model_deals.tactical()`,
two keys read in `Sample.from_conf`, a branch at the end of `get_bid_candidates`, and `do_rollout` no longer
skips the search when such a candidate is present (it did whenever the network's first choice was above
`no_search_threshold`, which is why the par guard's call was never looked at under `old`).

```
[sampling]
auction_model = /var/lib/ben/auction-model/runs/v1-brill
auction_model_deals = False                               # keep BEN's own sampled deals
auction_model_tactical = double,sacrifice,interfere,lead  # rules, in order; at most 3 calls a turn
auction_model_tactical_trust = 0.2                        # the insta_score the calls get; 'top' = just under the best
```

What the trust does: BEN adds `adjust_NN` (60 points, 200 when the other side has not bid) times the
insta_score to a candidate's simulated score, and takes 100 to 400 points off a double whose insta_score is
under 0.5 or 0.1. With 0.2 a named call has to beat the network's choice by about 2 IMPs in the simulation (a
double by about 6); with `top` the simulation alone decides.

Live test on the 10,000 deals of `src\data\brill-replay\live` (P620-5975WX-1, container `ben-live`; code in
`/var/lib/ben/auction-model/code`, PyTorch copied to `/var/lib/ben/torch-pkgs`, configs
`/var/lib/ben/brill/live/conf/tac.conf` (0.2) and `tac-even.conf` (`top`) = `old-brillall-noadj` + the keys):
`tactical_live_report.py` puts each changed board down to the rule that named South's first different call.

Results (2026-11-18; each against `old-brillall-noadj` on the same 10,000 deals, final contracts double dummy,
North, East and West bid by Brill's service; 780 and 1,374 new requests; replays in `live\10k\tac`, `tac-even`,
scores in `live\10k-tac-report`: `compare.txt`, `tac-by-rule.txt`, `tac-even-by-rule.txt`). The baseline with the
new code and no rule made the same auction on 200 of 200 deals.

| | `tac` (trust 0.2) | `tac-even` (trust `top`) |
|---|---|---|
| All deals | +0.038 +-0.022 IMPs a deal, 50.27% +-0.16 | +0.030 +-0.028, 50.30% +-0.20 |
| Auction differs; better, worse | 3.3%; 159, 105 | 5.5%; 237, 176 |
| Deals bid to MP scoring | +0.058 +-0.035 | +0.041 +-0.039 |
| Deals bid to IMP scoring | +0.017 +-0.027 | +0.019 +-0.039 |
| `double`: deals, IMPs each | 109, +2.51 +-1.25 | 197, +1.52 +-0.83 |
| `sacrifice`: deals, IMPs each | 72, +2.29 +-1.35 | 88, +2.10 +-1.22 |
| `push`: deals, IMPs each | 13, +2.31 +-3.46 | 22, +1.27 +-2.45 |
| `interfere`: deals, IMPs each | 109, -0.94 +-1.01 | 201, -1.11 +-0.82 |
| `lead`: deals, IMPs each | 21, 0.00 +-1.89 | 39, -0.21 +-1.31 |

- `double` and `sacrifice` gain under both settings; together with `push` they add about +0.047 (`tac`) and
  +0.051 (`tac-even`) IMPs a deal. That sum is read off the split, not a run of its own.
- `interfere` loses. The robots' reaction is in this test (they bid the rest of every auction), so the loss is
  not a blind spot of double dummy. What goes wrong in the largest swings: the rule's cheapest-level bid replaces
  a better natural one (1S for a 4S overcall on eight spades), a junk opening at the one level that North reads
  as an opening hand (3NT doubled, -1400), and North competing on the strength of the bid. To rework: no
  one-level openings, jumps only, never below a bid BEN already has in the same suit.
- `lead` cannot gain here: double dummy assumes the best lead. It needs Brill's autoplay to be judged.
- Against par the three stand at +0.078 (baseline), +0.101 (`tac`), +0.088 (`tac-even`) IMPs a deal; `old` was -0.024.

### Follow-up runs (2026-11-18/19, same 10,000 deals, trust 0.2)

`tac-ds` = `double,sacrifice` is the confirmed setting; the others are compared with it so that one rule differs.
Run on dev (`/var/lib/ben/brill/live/tacds` in dev's `ben-state` volume, configs `/var/lib/ben/am/conf/tac-*.conf`) and
on P620-5995WX-2 (`tac-d2`); each machine first reproduced the baseline's auctions on 400 of 400 deals. Replays in
`live\10k\<name>`, scores in `live\10k-tac-report` (`compare.txt`, `vs-tac-ds\`, `vs-old\`, `<name>-by-rule.txt`).

| Run | Rules | Against the baseline | Against `tac-ds` |
|---|---|---|---|
| `tac-ds` | `double,sacrifice` | **+0.048 +-0.019 IMPs a deal**, 50.34% +-0.13; 128 better, 61 worse | |
| `tac-i2` | + `interfere2` | +0.045 +-0.020 | -0.003 +-0.006 (9 better, 9 worse) |
| `tac-d2` | `double2,sacrifice` | +0.042 +-0.018 | -0.006 +-0.008 (11 better, 20 worse; IMP deals identical) |
| `tac-s2` | `double,sacrifice2` | +0.031 +-0.016 | -0.017 +-0.010 (21 better, 35 worse) |

- `tac-ds` by rule: `double` 109 deals, +2.51 +-1.25 IMPs each; `sacrifice` 77, +1.90 +-1.34; `push` 15, +2.67 +-3.17.
  Against `old` directly: +0.149 +-0.054 IMPs a deal, 50.96% +-0.35.
- `interfere2` (priced with the robots' measured doubling, one bid a suit at the highest safe level, no one-level
  bid, at most 5% risk of conceding 800): names a bid on 242 deals, the search takes it on 32 (10 better, 9 worse).
  Harmless and idle: at trust 0.2 a new call must beat the network's choice by about 2 IMPs in the simulation.
- `double2` (matchpoints: chance of defeat at least 65%; IMPs: expected IMP gain at least 0.5): same auctions as
  `double` on every IMP deal, a few paying doubles fewer on matchpoint deals. BEN's search already applies the
  scoring once a double is a candidate. Keep `double`.
- `sacrifice2` (priced with `robot_reaction.json`: doubled / they bid on / passed out) names fewer bids than
  `sacrifice` (80 against 125 deals in 2,500) and scores lower. The truer price is the worse nominator: "always
  doubled" plus the push tag lets more bids through to the search, which then prices them itself. Keep `sacrifice`.

### What the Brill robots do to our bids (measured, `robot_doubling.py`, `robot_reaction.py`)

15,145 distinct final contracts and 4,117 auctions of the live replays; the past events agree.

- Our failing contracts end doubled 3.7% (one down), 8.4%, 18.9%, 28.7% (four or more down) of the time; with 17
  points or fewer between our hands, in auctions where both sides bid, 25-47%; with 23 or more, 1-10%.
- After our bid over their game: doubled 48%, they bid on 16%, passed out 36% (failing and 17 points or fewer:
  69% doubled). Over their partscore: doubled 3%, they bid on 47%, passed out 51%.
- No difference between matchpoint and IMP boards (one down 3.2% against 4.3%; over game 53% against 51%).

### The matchpoint double against a real field (`mp_double_field.py`, `mp_double_points.py`)

4,690 tables of the Brill MP events where East-West declare undoubled, real field, tricks really taken.

- Doubling costs 41 matchpoints when the contract makes and gains 30 when it fails: break-even 58%, for games and
  partscores alike. With under 10% of the field in the contract: 19 and 25, break-even 43%; 10-30%: 48%.
  (A first table that averaged "makes" and "one down" over all boards gave 77%; that was wrong.)
- Pushed into game with under 30% of the field there: 73 of 4,690 tables. The field is usually in the same contract
  (the opponents are the same robots at every table): mean 80%.
- Rules for doubling, gain in matchpoints per table, model `v1` (never saw Brill deals): chance of defeat at least
  58%: +1.04 +-1.45; field known exactly: +1.15 +-1.47; field guessed from how usual South's calls were: +1.07;
  at least 75%: +0.98 +-0.68; today's `double` rule: +0.67 +-1.11. Field-awareness adds nothing measurable.
- The model is too sure of defeats at the top (`v1`: said 60-70%, true 50%).

### Real card play (`tactical_autoplay.py`)

Brill's `/autoplay` plays both contracts of the 197 deals on which `tac-ds` and the baseline end in different
contracts (about a minute a contract, one at a time, queued behind the board-table autoplay job). Output
`live\autoplay\tac-ds.jsonl`, log `tac-ds.log`; `tactical_autoplay.py report` compares real play with double dummy.

## Speed and memory notes

- The network is small (8.3M parameters) and limited by memory traffic, not arithmetic: mixed precision made
  it slower, so training runs in full precision. 6.8 steps of 768 boards a second on a quiet RTX 5080
  (about 3 when the machine's processors are busy with other work).
- Batches hold boards of similar auction length; long auctions get smaller batches.
- The sampler caches earlier positions, so drawing a set costs 23 single-token steps. The caches are allocated once
  (`Block.step`); growing them with `torch.cat` took the 16 GB card past its limit.
- Under WSL a process that asks for more than the card holds does not fail: it spills into system memory and runs
  about ten times slower (15.8 GB in `nvidia-smi`, low GPU use). `train.py` caps its share (`--mem-frac`),
  `eval_sets.py` sizes its chunks from free memory. `train.py --resume --resume-frac F` continues a stopped run.

## On prod (2026-11-19)

The confirmed setting (`auction_model_tactical = double,sacrifice`, trust 0.2, on top of the Brill-network
setting) is served by prod's BEN REST API since 2026-11-19 06:57. BEN code: commit `3038b293` on BSalita/ben
`main`. Prod's settings file and what its volume must hold (this folder's `infer.py`, `model.py`, `scoring.py`,
`tactical.py`, `robot_reaction.json`, the run `v1-brill`, and a PyTorch for the processor) are described in
`ops/ben_api_override.auction-model.conf` and `ops/BEN.md`. The files on prod are copies: after a change to
`tactical.py` or a new model, copy them to prod again.

Checks: the commit's tree started through prod's entrypoint in a test container gave the tested run's call on
370 of 370 South positions; prod's public address on 60 of 60, among them all 40 where this setting and the
Brill-network setting differ (0 of 40 before the deploy). A call takes 0.83 s at the median on prod (0.42 s before).

## Explorer page (2026-11-19)

`streamlit_app_auction_explorer.py` shows what the model believes from one seat, call by call: expected tricks
by declarer and strain, the best contracts for each side, the par distribution and how it moved with each call,
where the missing honors and suit lengths probably are, and the calls the purpose rules name. Pick a Brill board
(the real deal is then shown next to every prediction) or enter a hand and an auction.

- Local: `preview_start` name `auction-explorer` (port 8536; `elo/.venv`, PyTorch taken from `acbl-pipeline/.venv`).
- Public: https://auction.7nt.info, a container of its own on prod. Publishing and what prod holds:
  `ops/AUCTION_EXPLORER.md`, `ops/publish_auction_explorer.ps1`. The Brill boards on prod are as of the last publish.
- Tested headlessly with `streamlit.testing.v1.AppTest`: every step of a 14-call auction, all four seats' tabs, own
  deal, and the messages for a bad hand, a bad call and an illegal auction.

With "Anticipated leads" ticked (since 2026-11-24) the page also shows each contract's forecast after the
anticipated lead: "Makes after lead" and "Score after lead" in the two tables of best contracts, and a line
under the contract of the auction shown. The model's forecast is double dummy (the defence finds its best
lead); `lead_forecast.py` draws 100 deals from the model, gives every lead the leader can make the share of
BEN's lead network, solves each double dummy, and the page shifts the model's forecast by the difference to
the best lead on the same deals. On the played tables of the Brill events double dummy is 0.11 tricks too low
for declarer and the after-lead figure 0.17 too high, so the truth lies between the two columns. The helper
runs only where BEN is installed (the `ben` image); elsewhere the page says so.

## More Brill rows: `v1-brill-all` (2026-11-19)

`ft3.sh` repeats the `v1-brill` recipe on more rows. `brill.npz` was rebuilt (10,197 rows to 18 November); the 151
rows of 18 November (32 deals, `brill_new.npz`) are events no earlier model had seen.

South's viewpoint on those 151 rows, before any call -> auction over:

| Model | Trained on | Error of expected DD tricks | Error of expected par |
|---|---|---|---|
| `v1` | BBO only | 1.46 -> 1.08 | 287 -> 182 |
| `v1-brill` | + Brill to 5 November | 1.46 -> 1.00 | 291 -> 158 |
| `v1-brill-old` | + Brill to 17 November | 1.45 -> 1.01 | 290 -> 161 |
| `v1-brill-all` | + Brill to 18 November (these rows too) | not a test: it was trained on them | |

- The first check of the Brill fine-tuning on rows that played no part in choosing its recipe: it holds
  (1.08 -> 1.00 tricks, 182 -> 158 par at the end of the auction), on 32 deals only.
- Adding the 12 days of 6 to 17 November changes nothing that these rows can show.
- On held-out BBO deals the three fine-tuned models are the same (DD error 1.205 against 1.190 for `v1`).
- In BEN, on the 10,000 live deals (doubles and sacrifices, trust 0.2; 28 minutes with 48 workers, 25 new requests to
  Brill's service): `v1-brill-all` against `v1-brill` -0.002 +-0.005 IMPs a deal, the auction differs on 16 deals
  (7 better, 9 worse); against the baseline +0.046 +-0.019 (`v1-brill`: +0.048 +-0.019). The two models bid the same,
  so prod keeps `v1-brill`. Replay `data\brill-replay\live\10k\tac-ds-all`, scores `live\10k-tac-report\vs-tac-ds-all`.
- A retrain is worth repeating when it can be tested: hold the newest week or two out, train on the rest.

## The missing North-dealer boards: `v3`, `v3-brill` (2026-11-19)

Found in the explorer on a Brill board (South KQ732.Q87.A6.KQ3, North deals): after North and East pass, `v1-brill`
gave North-South an expected par of -2 (before any call +331). Cause: `bbo_mldf_augmented.parquet` has no board dealt
by North. The raw downloads (.lin files under `E:\bridge\data\bbo\data\<player>\`) do have boards 1, 5, 9, 13; they
were lost when the first table (`bbo_epdf.parquet`) was built. The model's inputs are relative to the seat, so what
it never saw was "the best hand sits opposite the dealer": with partner dealing, the viewer was never the strong hand.

- `extract_bbo_lin.py scan` reads the .lin files (30% of the 55,098 player folders, 7.4M files, 30 minutes with 56
  processes: Windows allows at most 61 pool workers): 946,865 distinct deal + auction rows on 344,238 North-dealer
  deals. `dd` solves them with endplay (234 deals a second, 25 minutes; tricks and par equal the old file's on 300 of
  300 old rows checked). `build` writes `bbo_north_{train,test}.npz` and `bbo4_{train,test}.npz` = the old rows plus
  these (5,918,594 training rows; by dealer N 926,284, E 1,704,839, S 1,689,788, W 1,597,683).
- `ft4.sh`: `v3` = `v1`'s recipe on `bbo4_train.npz` (60 minutes), `v3-brill` = `v3` + the `v1-brill` fine-tuning.

South's view, error of expected tricks / of expected par at the end of the auction (1-4 calls seen in brackets):

| Deals | `v1-brill` | `v3-brill` |
|---|---|---|
| BBO, North deals, held out (19,840 rows) | 1.15 / 234 (1.49 / 287) | 1.12 / 221 (1.45 / 273) |
| BBO, other dealers, held out (108,763) | 1.15 / 252 (1.51 / 322) | 1.14 / 248 (1.51 / 321) |
| Brill 6-17 November (1,938) | 1.19 / 253 (1.54 / 329) | 1.18 / 246 (1.53 / 323) |
| Brill 18 November (151) | 1.00 / 158 (1.27 / 228) | 0.96 / 163 (1.25 / 224) |

- The board: after the two passes `v3-brill` expects +152 (real par +120); call by call +104 to +228 where `v1-brill`
  went from -10 to +202.
- Still open: dealer and vulnerability go together through the board number, and BBO events seldom pass board 12, so
  four combinations are nearly absent from training (North deals + both vulnerable 1,549 rows, East + nobody 3,204,
  South + North-South only 3,112, West + East-West only 3,025). With North dealing and both vulnerable the same hand
  still falls to -15 after the two passes. Brill's events use boards 1 to 8 only, where this does not arise.
- All BBO deals, the new ones too, are "best hand" deals: South has 15.1-15.2 points on average and the best hand on
  94-95% (Brill events 12.5 and 57%, random deals 10 and 25%). `bias_check.py` measures what that does at the end of
  the auction on deals the model did not train on (South's view, `v1-brill` and `v3-brill` alike): East-West final
  contracts are given 52% to make on Brill deals and make 54% (random deals 63% against 66%), North-South contracts
  60% against 63% (64% against 65%); expected tricks are right within 0.05. So no tilt toward North-South, a slight
  pessimism about every contract.
- In BEN, 10,000 live deals, doubles and sacrifices (29 minutes, 75 new requests): `v3-brill` against `v1-brill`
  -0.000 +-0.009 IMPs a deal, 51 deals changed (27 better, 24 worse); against the baseline +0.048 +-0.019 for both.
  By dealer, and on the four rare combinations, no difference either. Replay `live\10k\tac-ds-v3`, scores
  `live\10k-tac-report\v3`.
- On prod since 2026-11-19 13:12: the robot and the explorer use `v3-brill` (`v1-brill` kept in the volume). Check
  through the public API: on 75 positions, 50 of them where the two models' runs call differently, prod made
  `v3-brill`'s call every time.

### Real card play: result (2026-11-19)

`tactical_autoplay.py` on the 197 deals where doubles-and-sacrifices (`tac-ds`, model `v1-brill`) and its baseline
end in different contracts; Brill's /autoplay played both contracts of 178 of them (20 contracts never came back:
gateway time-outs). New minus baseline, North-South view, IMPs per changed deal:

| Deals | n | Real play | Double dummy |
|---|---|---|---|
| all | 178 | +2.02 +-1.09 (sum +360) | +2.56 +-0.94 (sum +456) |
| a double of the same contract | 75 | +1.01 +-1.80 | +3.15 +-1.41 |
| a different contract (sacrifice, push, ...) | 103 | +2.76 +-1.35 | +2.14 +-1.26 |

Better / same / worse: 107 / 11 / 60 in real play, 115 / 9 / 54 double dummy. Declarers take 0.2 tricks more than
double dummy on average (North-South +0.18, East-West +0.23). So the gain holds in real play at about four fifths of
its double-dummy size (about +0.04 IMPs a deal over all deals); the different contracts carry it, the doubles alone
are not shown to gain in real play.
