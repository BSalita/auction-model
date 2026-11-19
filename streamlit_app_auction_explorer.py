"""Auction model explorer: step through an auction and see what the model believes from one seat.

  streamlit run streamlit_app_auction_explorer.py --server.port 8536

Shows, after any number of calls: expected double-dummy tricks for every declarer and strain, the best contracts
for each side, the par distribution and how it moved call by call, where the missing honors and suit lengths
probably are, and the calls the purpose rules (tactical.py) would name. With a Brill board the real deal is known,
so every prediction is shown next to the truth.

Where things are found (environment variables, each optional):
  AUCTION_MODEL_RUN        run folder holding model.pt (default: the first v1-brill folder that exists)
  AUCTION_MODEL_TORCH      folder holding the torch package if it is not importable
  AUCTION_EXPLORER_FIELD   Brill field file (default ../data/brill-replay/field.jsonl)
  AUCTION_EXPLORER_DD      double-dummy results of those boards (default ../data/brill-replay/board-table/dd.json)
  AUCTION_EXPLORER_BEN_API BEN's REST API, asked what each call shows (default https://ben.7nt.info/gameapi/8085;
                           empty = no meanings in the tooltips)
"""
import concurrent.futures
import json
import os
import pathlib
import sys
import time
import urllib.parse
import urllib.request

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

HERE = pathlib.Path(__file__).resolve().parent
try:
    import torch
except ImportError:
    # appended, not inserted: an older environment's other packages must not shadow this one's
    for extra in (os.environ.get("AUCTION_MODEL_TORCH"), HERE.parent / "acbl-pipeline" / ".venv" / "Lib" / "site-packages",
                  "/var/lib/ben/torch-pkgs"):
        if extra and os.path.isdir(os.path.join(str(extra), "torch")):
            sys.path.append(str(extra))
            break
    import torch
sys.path.insert(0, str(HERE))
import infer  # noqa: E402
import tactical  # noqa: E402

SEATS = "NESW"
SEAT_NAME = {"N": "North", "E": "East", "S": "South", "W": "West"}
STRAINS = infer.STRAINS                              # "SHDCN", the order of the model's trick table
SYMBOL = {"S": "♠", "H": "♥", "D": "♦", "C": "♣", "N": "NT"}
RED = "#d1342f"
GREEN = "#2e9e4f"                                    # a seat that is not vulnerable; RED one that is
BLUE = "42, 120, 214"                                # the one hue for magnitudes; shade = how much
RUNS = [os.environ.get("AUCTION_MODEL_RUN"), r"E:\bridge\data\auction-model\runs\v1-brill",
        "/var/lib/ben/auction-model/runs/v1-brill", "/data/runs/v1-brill"]
FIELD = os.environ.get("AUCTION_EXPLORER_FIELD") or str(HERE.parent / "data" / "brill-replay" / "field.jsonl")
DD = os.environ.get("AUCTION_EXPLORER_DD") or str(HERE.parent / "data" / "brill-replay" / "board-table" / "dd.json")
BEN_API = os.environ.get("AUCTION_EXPLORER_BEN_API", "https://ben.7nt.info/gameapi/8085").rstrip("/")
RULE_NAME = {"toward": "Toward a contract of ours", "interfere": "Interfere with their auction",
             "interfere2": "Interfere (priced with the robots' doubling)", "sacrifice": "Sacrifice or push",
             "sacrifice2": "Sacrifice (priced with the robots' reaction)", "double": "Penalty double",
             "double2": "Double by the scoring", "lead": "Direct partner's lead"}
LIVE_RULES = ("double", "sacrifice")                 # the rules whose calls the robot on prod looks at


# ---- model ----------------------------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading the model ...")
def predictor(run):
    return infer.Predictor(run, device="cpu")


@st.cache_data(show_spinner=False, max_entries=4000)
def glance(run, hand, seat, dealer, vul, calls):
    """Trick probabilities (declarer relative to the seat, strain, tricks) and the par distribution."""
    pos = predictor(run).position(hand, seat, dealer, vul, list(calls))
    mean, top = pos.par_summary(top=len(pos.par))
    return {"dd": pos.dd, "par_mean": mean, "par": top}


@st.cache_data(show_spinner="Sampling the hidden hands ...", max_entries=400)
def detail(run, hand, seat, dealer, vul, calls, matchpoint, to_call):
    """Honor and length probabilities of the hidden hands, and the calls the purpose rules name."""
    torch.manual_seed(0)
    pos = predictor(run).position(hand, seat, dealer, vul, list(calls))
    honors, lengths = pos.marginals(n=2048)
    named, context, problem = [], None, None
    if to_call:
        try:
            named, context = tactical.propose(pos, hand, list(calls), SEATS.index(dealer), SEATS.index(seat), matchpoint=matchpoint)
        except Exception as error:                   # an auction the rules were not written for
            problem = f"{type(error).__name__}: {error}"
    return honors, lengths, named, context, problem


# ---- bridge helpers -------------------------------------------------------------------------------------
def norm_call(text):
    c = text.strip().upper().replace("NT", "N")
    if c in ("P", "PASS", "-", "--"):
        return "PASS"
    if c in ("X", "D", "DBL", "DB"):
        return "X"
    if c in ("XX", "R", "RDBL", "RD"):
        return "XX"
    if len(c) == 2 and c[0] in "1234567" and c[1] in "CDHSN":
        return c
    raise ValueError(f"'{text}' is not a call")


def check_auction(calls):
    """The legal prefix of calls and a message about the first call that is not legal (or None)."""
    last, last_by, dbl, passes = None, None, 0, 0
    for i, c in enumerate(calls):
        if (passes == 3 and last) or (passes == 4):
            return calls[:i], "the auction was already over"
        if c == "PASS":
            passes += 1
            continue
        if c == "X":
            ok = last and dbl == 0 and (i - last_by) % 2 == 1
        elif c == "XX":
            ok = last and dbl == 1 and (i - last_by) % 2 == 0
        else:
            ok = last is None or (int(c[0]), "CDHSN".index(c[1])) > (int(last[0]), "CDHSN".index(last[1]))
        if not ok:
            return calls[:i], f"call {i + 1} ({c}) is not legal there"
        passes = 0
        if c == "X":
            dbl = 1
        elif c == "XX":
            dbl = 2
        else:
            last, last_by, dbl = c, i, 0
    return calls, None


def over(calls):
    return len(calls) >= 4 and all(c == "PASS" for c in calls[-3:]) and (any(c[0].isdigit() for c in calls) or len(calls) == 4)


def contract_of(calls, dealer):
    """(level, strain, doubled 0/1/2, declarer seat) of the last bid so far, or None."""
    bids = [(i, c) for i, c in enumerate(calls) if c[0].isdigit()]
    if not bids:
        return None
    i_last, last = bids[-1]
    side = (SEATS.index(dealer) + i_last) % 2
    first = next(i for i, c in bids if c[1] == last[1] and (SEATS.index(dealer) + i) % 2 == side)
    after = calls[i_last + 1:]
    return int(last[0]), last[1], 2 if "XX" in after else 1 if "X" in after else 0, SEATS[(SEATS.index(dealer) + first) % 4]


def final_contract(text, declarer):
    """'2DX', 'W' -> '2♦x by West'; nothing bid -> 'Passed out'."""
    c = (text or "").upper().replace("NT", "N")
    if len(c) < 2 or c[0] not in "1234567":
        return "Passed out"
    return f"{c[0]}{SYMBOL[c[1]]}{'x' * c[2:].count('X')}" + (f" by {SEAT_NAME[declarer]}" if declarer in SEAT_NAME else "")


def result_of(t):
    """'made', 'made +1', 'down 2' or '' for a table of the field file."""
    c = (t.get("contract") or "").upper()
    if len(c) < 2 or c[0] not in "1234567" or t.get("tricks") is None:
        return ""
    over_ = int(t["tricks"]) - (int(c[0]) + 6)
    return "made" if over_ == 0 else f"made +{over_}" if over_ > 0 else f"down {-over_}"


def side_vul(vul, seat):
    return vul in ("All", "Both") or (vul in ("NS", "N_S") and seat in "NS") or (vul in ("EW", "E_W") and seat in "EW")


def show_call(c):
    if c == "PASS":
        return "Pass"
    if c in ("X", "XX"):
        return c
    color = f' style="color:{RED}"' if c[1] in "HD" else ""
    return f"{c[0]}<span{color}>{SYMBOL[c[1]]}</span>"


def plain_call(c):
    return "Pass" if c == "PASS" else c if c in ("X", "XX") else c[0] + SYMBOL[c[1]]


def hand_html(hand, title, me=False, vulnerable=False):
    rows = "".join(f'<div><span style="color:{RED if s in "HD" else "inherit"}">{SYMBOL[s]}</span> {cards or "—"}</div>'
                   for s, cards in zip("SHDC", hand.split(".")))
    weight = "700" if me else "400"
    return f'<div style="font-family:monospace;line-height:1.35"><div style="font-weight:{weight};color:{RED if vulnerable else GREEN}">{title}</div>{rows}</div>'


def button_call(c, seen):
    """A call as a button label: red suits, gray when not seen yet."""
    if not seen:
        return f":gray[{plain_call(c)}]"
    if c[0].isdigit() and c[1] in "HD":
        return f"{c[0]}:red[{SYMBOL[c[1]]}]"
    return plain_call(c)


def rewind(k):
    st.session_state["step"] = k


@st.cache_resource
def meaning_store():
    """Meanings already fetched, shared by all sessions: {(dealer, vul, calls so far): text}; 'down' = time of the last failure."""
    return {"down": 0.0}


def fetch_meaning(dealer, vul, prefix):
    """What the last call of prefix shows, from BEN's rule book (its /explain service): '5+ ♠; 11-20 HCP'."""
    to_ctx = {"PASS": "--", "X": "Db", "XX": "Rd"}
    by = SEATS[(SEATS.index(dealer) + len(prefix) - 1) % 4]
    query = urllib.parse.urlencode({"seat": by, "dealer": dealer, "vul": vul, "ctx": "".join(to_ctx.get(c, c) for c in prefix)})
    request = urllib.request.Request(f"{BEN_API}/explain?{query}", headers={"User-Agent": "auction-explorer"})
    with urllib.request.urlopen(request, timeout=5) as answer:
        text = json.load(answer).get("explanation") or ""
    name, _, shows = text.partition(" -- ")
    for s in "SHDC":
        name, shows = name.replace("!" + s, SYMBOL[s]), shows.replace("!" + s, SYMBOL[s])
    shows = shows.strip(" ;")
    return ": ".join(x for x in (name.strip(), shows) if x)


def meanings(dealer, vul, calls):
    """The meaning of every call ('' where it is not known). Asks only for what is not stored yet, in parallel;
    after a failure the service is left alone for a minute, so a page never waits on it call after call."""
    store = meaning_store()
    keys = [(dealer, vul, tuple(calls[:i + 1])) for i in range(len(calls))]
    missing = [k for k in keys if k not in store]
    if missing and BEN_API and time.time() - store["down"] > 60:
        try:
            with concurrent.futures.ThreadPoolExecutor(8) as pool:
                for k, text in zip(missing, pool.map(lambda k: fetch_meaning(*k), missing)):
                    store[k] = text
        except Exception:
            store["down"] = time.time()
    return [store.get(k, "") for k in keys]


def auction_grid(calls, step, dealer, seat, vul, tips):
    """The auction as four columns of buttons (West first); a click on a call sets 'Calls seen' to that call.
    tips: the text shown when the pointer rests on each call."""
    st.markdown("""<style>
        .st-key-auction [data-testid="stVerticalBlock"] {gap: 0.1rem}
        .st-key-auction button {min-height: 1.7rem; padding: 0 0.5rem; font-family: monospace}
        .st-key-auction [data-testid="stHorizontalBlock"] {gap: 0.4rem}
        </style>""", unsafe_allow_html=True)
    order = "WNES"
    with st.container(key="auction"):
        cols = st.columns(4, width=420)
        for col, s in zip(cols, order):
            col.markdown(f'<span style="font-family:monospace;font-weight:{700 if s == seat else 400};'
                         f'color:{RED if side_vul(vul, s) else GREEN}">{SEAT_NAME[s]}</span>', unsafe_allow_html=True)
        lead = order.index(dealer)
        for k in range(lead):                        # seats before the dealer in the first row: a blank of a button's height
            cols[k].button(" ", key=f"blank{k}", type="tertiary", disabled=True)
        for i, c in enumerate(calls):
            cols[(lead + i) % 4].button(button_call(c, i < step), key=f"call{i}", on_click=rewind, args=(i + 1,),
                                        type="secondary" if i + 1 == step else "tertiary",
                                        help=tips[i])
        if calls:
            st.button("Before any call", key="call_none", on_click=rewind, args=(0,),
                      type="secondary" if step == 0 else "tertiary")


def shade(lo, hi):
    """Cell style for a magnitude: one hue, more of it for more."""
    def style(v):
        if not isinstance(v, (int, float)) or np.isnan(v):
            return ""
        a = min(max((v - lo) / (hi - lo), 0.0), 1.0) * 0.6
        return f"background-color: rgba({BLUE}, {a:.2f})"
    return style


def contracts_table(dd, seat, vul, side):
    """The best contracts of one side ('NS' or 'EW') by expected double-dummy score, undoubled."""
    rows = []
    for si, strain in enumerate(STRAINS):
        for level in range(1, 8):
            best = None
            for decl in side:
                rel = (SEATS.index(decl) - SEATS.index(seat)) % 4
                p = dd[rel, si]
                score = float((p * infer.SCORE[level - 1, si, 0, int(side_vul(vul, decl))]).sum())
                if best is None or score > best[0]:
                    best = (score, float(p[level + 6:].sum()), decl)
            rows.append({"Contract": f"{level}{SYMBOL[strain]}", "By": SEAT_NAME[best[2]], "Makes": best[1], "Expected score": best[0]})
    return pd.DataFrame(rows).sort_values("Expected score", ascending=False).head(6).reset_index(drop=True)


# ---- Brill boards ---------------------------------------------------------------------------------------
@st.cache_data(show_spinner="Reading the Brill boards ...")
def brill_boards(path, stamp):
    boards = [json.loads(line) for line in open(path, encoding="utf-8")]
    return [b for b in boards if any(t.get("auction") for t in b["tables"])]


@st.cache_data(show_spinner=False)
def dd_results(path, stamp):
    return json.load(open(path, encoding="utf-8")) if os.path.isfile(path) else {}


def stamp(path):
    return os.path.getmtime(path) if os.path.isfile(path) else 0


# ---- page -----------------------------------------------------------------------------------------------
st.set_page_config(page_title="Auction model explorer", layout="wide")
run = next((r for r in RUNS if r and os.path.isfile(os.path.join(r, "model.pt"))), None)
if run is None:
    st.error("No model found. Set AUCTION_MODEL_RUN to the run folder that holds model.pt.")
    st.stop()

st.title("Auction model explorer")
st.caption(f"What the model believes from one seat, call by call. Model: {os.path.basename(os.path.normpath(run))}. "
           "It sees one hand and the calls so far, never the other three hands.")

have_boards = os.path.isfile(FIELD)
with st.sidebar:
    source = st.radio("Deal", ["A Brill board", "My own deal"] if have_boards else ["My own deal"], horizontal=True)
    deal, truth, matchpoint, table, tables = None, None, False, None, []
    if source == "A Brill board":
        boards = brill_boards(FIELD, stamp(FIELD))
        dates = sorted({b["date"] for b in boards}, reverse=True)
        date = st.selectbox("Date", dates)
        events = sorted({(b["event"], b["tournament_id"]) for b in boards if b["date"] == date})
        event = st.selectbox("Event", events, format_func=lambda e: e[0])
        of_event = sorted((b for b in boards if b["tournament_id"] == event[1]), key=lambda b: b["board"])
        board = st.selectbox("Board", of_event, format_func=lambda b: f"Board {b['board']}  ({b['dealer']} deals, {b['vul']} vul)")
        tables = [t for t in board["tables"] if t.get("auction")]
        table = st.selectbox("Auction of", tables, format_func=lambda t: f"{t['name']}: {t.get('contract') or 'passed out'} {t.get('declarer') or ''}")
        first = SEATS.index(board["deal"][0])
        hands = board["deal"][2:].split()
        deal = {SEATS[(first + i) % 4]: hands[i] for i in range(4)}
        dealer, vul, matchpoint = board["dealer"], board["vul"], board.get("scoring") == "MP"
        raw_calls = table["auction"]
        truth = dd_results(DD, stamp(DD)).get(f"{board['deal']}|{board['dealer']}|{board['vul']}")
        seat = st.radio("Seen from", list(SEATS), index=2, horizontal=True, format_func=SEAT_NAME.get)
        hand = deal[seat]
    else:
        seat = st.radio("Seen from", list(SEATS), index=2, horizontal=True, format_func=SEAT_NAME.get)
        hand = st.text_input("Hand (spades.hearts.diamonds.clubs)", "AQJ9652.64.T3.82").strip().upper().replace("10", "T")
        dealer = st.radio("Dealer", list(SEATS), index=0, horizontal=True, format_func=SEAT_NAME.get)
        vul = st.radio("Vulnerable", ["None", "NS", "EW", "All"], horizontal=True)
        matchpoint = st.radio("Scoring", ["IMPs", "Matchpoints"], horizontal=True) == "Matchpoints"
        raw_calls = st.text_input("Auction (calls from the dealer, e.g. 2C P 2D)", "2C").split()
    reveal = st.checkbox("Show the hidden hands and the real results", value=False, disabled=deal is None,
                         help="Only for a Brill board, where the whole deal is known.")

suits = hand.split(".")
if len(suits) != 4 or sum(len(s) for s in suits) != 13 or any(ch not in infer.RANKS for s in suits for ch in s) \
        or any(len(set(s)) != len(s) for s in suits):
    st.error("The hand must be 13 different cards written as spades.hearts.diamonds.clubs, for example AQJ9652.64.T3.82")
    st.stop()
try:
    calls = [norm_call(c) for c in raw_calls]
except ValueError as error:
    st.error(f"{error}. Write calls as 1C, 2N, P, X, XX.")
    st.stop()
calls, problem = check_auction(calls)
if problem:
    st.warning(f"Auction cut short: {problem}.")

n = len(calls)
shown = (hand, seat, dealer, vul, tuple(calls))
if st.session_state.get("shown") != shown or not 0 <= st.session_state.get("step", n) <= n:
    st.session_state["shown"], st.session_state["step"] = shown, n          # another deal or auction: start at its end
step = st.slider("Calls seen", 0, n, key="step", help="Move left to go back in the auction, or click a call in the auction.") if n else 0
seen = tuple(calls[:step])
to_call = SEATS[(SEATS.index(dealer) + step) % 4]
finished = over(list(seen))
now = glance(run, hand, seat, dealer, vul, seen)
sign = 1 if seat in "NS" else -1                     # the files give North-South scores
my_side, their_side = ("NS", "EW") if seat in "NS" else ("EW", "NS")
history = [glance(run, hand, seat, dealer, vul, tuple(calls[:k]))["par_mean"] for k in range(n + 1)]   # expected par, call by call

left, right = st.columns([1, 1.3])
with left:
    if reveal and deal:
        grid = st.columns(3)
        grid[1].markdown(hand_html(deal["N"], "North", seat == "N", side_vul(vul, "N")), unsafe_allow_html=True)
        row = st.columns(3)
        row[0].markdown(hand_html(deal["W"], "West", seat == "W", side_vul(vul, "W")), unsafe_allow_html=True)
        row[2].markdown(hand_html(deal["E"], "East", seat == "E", side_vul(vul, "E")), unsafe_allow_html=True)
        grid = st.columns(3)
        grid[1].markdown(hand_html(deal["S"], "South", seat == "S", side_vul(vul, "S")), unsafe_allow_html=True)
    else:
        st.markdown(hand_html(hand, SEAT_NAME[seat], True, side_vul(vul, seat)), unsafe_allow_html=True)
    st.caption(f"{SEAT_NAME[dealer]} deals, vulnerable: {vul}, {'matchpoints' if matchpoint else 'IMPs'}.")
with right:
    shows = meanings(dealer, vul, calls)
    tips = []
    for i, c in enumerate(calls):
        by = SEATS[(SEATS.index(dealer) + i) % 4]
        lines = [f"**{plain_call(c)} by {SEAT_NAME[by]}**"]
        if shows[i]:
            lines.append(f"Shows: {shows[i]}")
        lines.append(f"Expected par for {my_side[0]}-{my_side[1]}: {history[i]:+.0f} before, {history[i + 1]:+.0f} after")
        tips.append("  \n".join(lines))
    auction_grid(calls, step, dealer, seat, vul, tips)
    tail = (" Click a call to see the auction up to it. Rest the pointer on a call for what it shows in the 2/1 rule book "
            "BEN uses (the player may have meant something else) and how the model's par moved.") if n else ""
    if finished:
        st.caption("The auction is over." + tail)
    elif step < n:
        st.caption(f"{SEAT_NAME[to_call]} to call. Gray calls have not been seen yet." + tail)
    else:
        st.caption(f"{SEAT_NAME[to_call]} to call." + tail)

current = contract_of(list(seen), dealer)
top = st.columns(4)
top[0].metric(f"Expected par for {my_side[0]}-{my_side[1]}", f"{now['par_mean']:+.0f}")
if current:
    level, strain, dbl, decl = current
    rel = (SEATS.index(decl) - SEATS.index(seat)) % 4
    p = now["dd"][rel, STRAINS.index(strain)]
    score = float((p * infer.SCORE[level - 1, STRAINS.index(strain), dbl, int(side_vul(vul, decl))]).sum())
    name = f"{level}{SYMBOL[strain]}{'x' * dbl} by {SEAT_NAME[decl]}"
    top[1].metric("Contract so far", name)
    top[2].metric("Chance it makes", f"{p[level + 6:].sum():.0%}")
    top[3].metric(f"Its expected score for {my_side[0]}-{my_side[1]}", f"{(score if decl in my_side else -score):+.0f}")
if reveal and table:
    st.markdown("**What really happened at this table**")
    real = st.columns(4)
    real[0].metric("Final contract", final_contract(table.get("contract"), table.get("declarer")))
    real[1].metric("Result", f"{table['tricks']} tricks, {result_of(table)}" if result_of(table) else "no play")
    if table.get("ns_score") is not None:
        real[2].metric(f"Score for {my_side[0]}-{my_side[1]}", f"{sign * table['ns_score']:+d}")
    if truth:
        real[3].metric(f"Real par for {my_side[0]}-{my_side[1]}", f"{sign * truth['par_score']:+d}")
if reveal and truth:
    line = f"Real par for {my_side[0]}-{my_side[1]}: {sign * truth['par_score']:+d} ({truth.get('par', '')})."
    if current:
        took = truth["tricks"][current[1] + current[3]]
        line += f" Double dummy, {name} takes {took} tricks: {'makes' if took >= current[0] + 6 else f'down {current[0] + 6 - took}'}."
    st.caption(line)

names = ["Tricks and contracts", "Par", "Hidden hands", "Calls by purpose"] + (["Results at the tables"] if table else [])
tabs = st.tabs(names)
tab_tricks, tab_par, tab_hidden, tab_calls = tabs[:4]

with tab_tricks:
    expected = (now["dd"] * np.arange(14)).sum(-1)
    order = [(SEATS.index(d) - SEATS.index(seat)) % 4 for d in SEATS]
    table_model = pd.DataFrame(expected[order], index=[SEAT_NAME[d] for d in SEATS], columns=[SYMBOL[s] for s in STRAINS])
    cols = st.columns(2)
    cols[0].markdown("**Expected tricks, by declarer**")
    cols[0].dataframe(table_model.style.format("{:.1f}").map(shade(4, 11)), width="stretch")
    if reveal and truth:
        real = pd.DataFrame([[truth["tricks"][s + d] for s in STRAINS] for d in SEATS],
                            index=[SEAT_NAME[d] for d in SEATS], columns=[SYMBOL[s] for s in STRAINS])
        cols[1].markdown("**Real double-dummy tricks**")
        cols[1].dataframe(real.style.map(shade(4, 11)), width="stretch")
    else:
        cols[1].caption("Darker = more tricks. The model gives a probability for every number of tricks; this table shows "
                        "the average. Tick the box in the sidebar to compare with the real deal.")
    cols = st.columns(2)
    for col, side in zip(cols, (my_side, their_side)):
        col.markdown(f"**Best contracts for {side[0]}-{side[1]}** (undoubled, by expected score)")
        col.dataframe(contracts_table(now["dd"], seat, vul, side).style.format({"Makes": "{:.0%}", "Expected score": "{:+.0f}"})
                      .map(shade(0, 1), subset=["Makes"]), width="stretch", hide_index=True)

with tab_par:
    trend = pd.DataFrame({"Calls seen": range(n + 1), "Expected par": history,
                          "After": ["before any call"] + [plain_call(c) for c in calls]})
    cols = st.columns(2)
    with cols[0]:
        st.markdown(f"**Expected par for {my_side[0]}-{my_side[1]}, call by call**")
        base = alt.Chart(trend).encode(x=alt.X("Calls seen:Q", axis=alt.Axis(tickMinStep=1, grid=False)),
                                       y=alt.Y("Expected par:Q", scale=alt.Scale(zero=False), title="Expected par"))
        layers = [base.mark_line(color=f"rgb({BLUE})", strokeWidth=2),
                  base.mark_point(color=f"rgb({BLUE})", filled=True, size=70, opacity=1)
                      .encode(tooltip=["Calls seen", "After", alt.Tooltip("Expected par:Q", format="+.0f")]),
                  alt.Chart(trend[trend["Calls seen"] == step]).mark_point(color=f"rgb({BLUE})", size=260, strokeWidth=2)
                      .encode(x="Calls seen:Q", y="Expected par:Q")]
        if reveal and truth:
            real_par = pd.DataFrame({"y": [sign * truth["par_score"]], "label": [f"real par {sign * truth['par_score']:+d}"]})
            layers.append(alt.Chart(real_par).mark_rule(color="gray", strokeDash=[5, 4]).encode(y="y:Q"))
            layers.append(alt.Chart(real_par).mark_text(align="left", dx=4, dy=-7, color="gray").encode(x=alt.value(0), y="y:Q", text="label"))
        st.altair_chart(alt.layer(*layers).properties(height=300), width="stretch")
        st.caption("The ring marks the step chosen above.")
    with cols[1]:
        st.markdown(f"**Most likely par scores now, for {my_side[0]}-{my_side[1]}**")
        likely = pd.DataFrame(now["par"][:10], columns=["Par", "Probability"]).sort_values("Par")
        likely["Par score"] = likely["Par"].map(lambda v: f"{v:+d}")
        bars = alt.Chart(likely).mark_bar(color=f"rgb({BLUE})", cornerRadiusTopLeft=4, cornerRadiusTopRight=4).encode(
            x=alt.X("Par score:N", sort=list(likely["Par score"]), axis=alt.Axis(labelAngle=0), title="Par score"),
            y=alt.Y("Probability:Q", axis=alt.Axis(format="%")),
            tooltip=["Par score", alt.Tooltip("Probability:Q", format=".1%")])
        st.altair_chart(bars.properties(height=300), width="stretch")
        if reveal and truth:
            given = next((p for v, p in now["par"] if v == sign * truth["par_score"]), 0.0)
            st.caption(f"The real par, {sign * truth['par_score']:+d}, was given {given:.0%}.")

with tab_hidden:
    my_turn = to_call == seat and not finished
    honors, lengths, named, context, rule_problem = detail(run, hand, seat, dealer, vul, seen, matchpoint, my_turn)
    who = {"LHO": SEATS[(SEATS.index(seat) + 1) % 4], "partner": SEATS[(SEATS.index(seat) + 2) % 4], "RHO": SEATS[(SEATS.index(seat) + 3) % 4]}
    label = {k: f"{SEAT_NAME[v]} ({'partner' if k == 'partner' else k})" for k, v in who.items()}
    cols = st.columns([1.2, 1])
    with cols[0]:
        st.markdown("**Where the missing honors are**")
        rows = []
        for card, p in honors.items():
            row = {"Card": SYMBOL[card[0]] + card[1], **{label[k]: p[k] for k in ("LHO", "partner", "RHO")}}
            if reveal and deal:
                row["Really"] = next(SEAT_NAME[s] for s in SEATS if card[1] in deal[s].split(".")["SHDC".index(card[0])])
            rows.append(row)
        frame = pd.DataFrame(rows)
        st.dataframe(frame.style.format({v: "{:.0%}" for v in label.values()}).map(shade(0, 1), subset=list(label.values())),
                     width="stretch", hide_index=True, height=min(38 * (len(frame) + 1), 620))
    with cols[1]:
        st.markdown("**Expected suit lengths**")
        frame = pd.DataFrame([[lengths[k]["SHDC"[i]] for i in range(4)] for k in ("LHO", "partner", "RHO")],
                             index=[label[k] for k in ("LHO", "partner", "RHO")], columns=[SYMBOL[s] for s in "SHDC"])
        st.dataframe(frame.style.format("{:.1f}").map(shade(0, 7)), width="stretch")
        if reveal and deal:
            st.markdown("**Real suit lengths**")
            real = pd.DataFrame([[len(x) for x in deal[who[k]].split(".")] for k in ("LHO", "partner", "RHO")],
                                index=[label[k] for k in ("LHO", "partner", "RHO")], columns=[SYMBOL[s] for s in "SHDC"])
            st.dataframe(real.style.map(shade(0, 7)), width="stretch")
        st.caption("From 2,048 layouts drawn from the model. Darker = more likely, or longer.")

with tab_calls:
    if finished:
        st.info("The auction is over: no call to make.")
    elif to_call != seat:
        mine = [k for k in range(n + 1) if SEATS[(SEATS.index(dealer) + k) % 4] == seat and not over(calls[:k])]
        st.info(f"It is {SEAT_NAME[to_call]}'s turn. The rules speak when {SEAT_NAME[seat]} is to call: "
                f"set 'Calls seen' to {', '.join(map(str, mine)) or 'a later step'}.")
    else:
        if rule_problem:
            st.warning(f"The rules could not read this auction ({rule_problem}).")
        if context:
            c = st.columns(4)
            c[0].metric("They have a game", f"{context['p_game']:.0%}")
            c[1].metric("They have a slam", f"{context['p_slam']:.0%}")
            c[2].metric("Their best contract", f"{context['their_best'][0]}{SYMBOL[context['their_best'][1]]} by {SEAT_NAME[context['their_best'][2]]}")
            c[3].metric("Its score for us", f"{context['their_best_score']:+d}")
        if named:
            rows = [{"Purpose": RULE_NAME.get(p["rule"], p["rule"]), "Kind": p["tag"], "Call": plain_call(p["bid"]),
                     "Looked at by the robot": "yes" if p["rule"] in LIVE_RULES else "no",
                     "The numbers behind it": ", ".join(f"{k} {v}" for k, v in p.items() if k not in ("rule", "tag", "bid"))}
                    for p in named]
            st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
            st.caption("A named call is a suggestion, not a decision: the robot adds the calls marked 'yes' to its candidates "
                       "and its search then decides. Scores are for our side; 'their_fail' and 'their_make' are chances.")
        elif not rule_problem:
            st.write("The rules name no call here.")
        if step < n:
            st.caption(f"The call actually made here: {plain_call(calls[step])}.")

if table:
    with tabs[4]:
        if not reveal:
            st.info("Tick 'Show the hidden hands and the real results' in the sidebar to see what every table did with this board.")
        else:
            rows = [{"Player": t["name"], "Kind": t.get("type", ""),
                     "Auction": " ".join(plain_call(norm_call(c)) for c in t["auction"]),
                     "Contract": final_contract(t.get("contract"), t.get("declarer")),
                     "Tricks": t.get("tricks") if result_of(t) else None, "Result": result_of(t),
                     f"Score for {my_side[0]}-{my_side[1]}": None if t.get("ns_score") is None else sign * t["ns_score"],
                     "_chosen": (t.get("player_id"), t["auction"]) == (table.get("player_id"), table["auction"])}
                    for t in tables]
            score = f"Score for {my_side[0]}-{my_side[1]}"
            frame = pd.DataFrame(rows).sort_values(score, ascending=False, na_position="last").reset_index(drop=True)
            chosen = frame.pop("_chosen").tolist()
            st.markdown(f"**All {len(frame)} tables on this board**, best score for {my_side[0]}-{my_side[1]} first. "
                        f"The colored row is the table chosen in the sidebar ({table['name']}).")
            st.dataframe(frame.style.format({score: "{:+.0f}", "Tricks": "{:.0f}"}, na_rep="")
                         .apply(lambda row: [f"background-color: rgba({BLUE}, 0.35)" if chosen[row.name] else ""] * len(row), axis=1),
                         width="stretch", hide_index=True,
                         height=min(38 * (len(frame) + 1), 700))
            if truth:
                st.caption(f"Real par for {my_side[0]}-{my_side[1]}: {sign * truth['par_score']:+d} ({truth.get('par', '')}).")
