"""The explorer's "Why this call?" panel and the feedback bar under it.

For the call selected in the auction grid, when the visitor may see the hand that made it: Morty's own reasoning
at that point (BEN's /bid with details=true: the candidates, the deals its search imagined, each candidate's score
on them and the features that decide), written by morty-book's search_explain.py. Under it, a bar to mark the
explanation helpful, unclear or wrong, or to give the call the visitor would make, sent to the feedback store
(morty-book feedback.py) through the gateway's public /feedback.

  AUCTION_EXPLORER_FEEDBACK  the public feedback address (default https://mortybridgebot.7nt.info/feedback;
                             empty = no bar)

The record names the position as the explain service does (the caller as South: the auction from the dealer,
the hand, vulnerability turned to the caller's side), so the triage job can run Morty's search on it again.
"""
import hashlib
import json
import os
import sys
import time
import urllib.parse
import urllib.request
import uuid

import streamlit as st

HERE = os.path.dirname(os.path.abspath(__file__))
for folder in (HERE, os.path.join(HERE, '..', 'morty-book')):      # prod: a copy beside this file; dev: the sibling
    if os.path.exists(os.path.join(folder, 'search_explain.py')) and folder not in sys.path:
        sys.path.append(folder)
        break
import search_explain  # noqa: E402

FEEDBACK = os.environ.get('AUCTION_EXPLORER_FEEDBACK', 'https://mortybridgebot.7nt.info/feedback').strip()
VERSION = 'auction-explorer why_call 2026-11-26'
SEATS = 'NESW'
VERDICTS = {'Helpful': 'helpful', 'Unclear': 'unclear', 'Wrong': 'wrong', "I'd bid something else": 'disagree'}
TAGS = ['inference from the auction', 'system or agreement', 'hand evaluation', 'vulnerability or scoring', 'other']
PER_MINUTE = 5                     # records a minute from one visitor (the store also limits the explorer as a whole)


@st.cache_data(show_spinner="Asking Morty why ...", max_entries=500, ttl=3600)
def bid_details(ben_api, hand, seat, dealer, vul, before, system, matchpoint):
    asked = {'hand': hand, 'seat': seat, 'dealer': dealer, 'vul': vul, 'ctx': '-'.join(before) if before else '',
             'details': 'true', 'matchtype': 'MP' if matchpoint else 'IMP', 'ourSystem': system, 'theirSystem': system}
    request = urllib.request.Request(f"{ben_api}/bid?{urllib.parse.urlencode(asked)}",
                                     headers={'User-Agent': 'auction-explorer'})
    with urllib.request.urlopen(request, timeout=60) as answer:
        return json.load(answer)


def south_vul(vul, seat):
    """Vulnerability seen with the caller turned to South."""
    if seat in 'NS':
        return vul
    return {'NS': 'EW', 'EW': 'NS'}.get(vul, vul)


def visitor_id():
    if 'visitor' not in st.session_state:
        st.session_state['visitor'] = uuid.uuid4().hex[:12]
    return st.session_state['visitor']


def too_many():
    now = time.time()
    sent = [t for t in st.session_state.get('feedback_times', []) if now - t < 60]
    st.session_state['feedback_times'] = sent
    return len(sent) >= PER_MINUTE


def send(record):
    data = json.dumps(record).encode('utf-8')
    request = urllib.request.Request(FEEDBACK, data=data, method='POST',
                                     headers={'Content-Type': 'application/json', 'User-Agent': 'auction-explorer'})
    try:
        with urllib.request.urlopen(request, timeout=20) as answer:
            return answer.status, json.load(answer)
    except urllib.error.HTTPError as error:
        try:
            return error.code, json.load(error)
        except ValueError:
            return error.code, {'error': str(error)}


def panel(ben_api, seen, dealer, vul, hands, may_see, system, matchpoint, played):
    """Under the auction grid. seen: the calls up to the selected one; hands: {seat: hand} the page knows;
    may_see(seat): whether the visitor may see that seat's hand; played: the call shown at the table."""
    if not seen or not ben_api:
        return
    by = SEATS[(SEATS.index(dealer) + len(seen) - 1) % 4]
    hand = hands.get(by)
    if not hand or not may_see(by):
        return
    before = list(seen[:-1])
    key = f"why|{dealer}|{vul}|{system}|{matchpoint}|{by}|{hand}|{' '.join(before)}"
    if st.button(f"Why would Morty bid that? ({ {'N': 'North', 'E': 'East', 'S': 'South', 'W': 'West'}[by] }'s hand)", key='why_button',
                 help="Morty's reasoning at this point: the calls it weighed, the deals it imagined, and what decides."):
        st.session_state['why_open'] = key
    if st.session_state.get('why_open') != key:
        return
    try:
        result = bid_details(ben_api, hand, by, dealer, vul, tuple(before), system, matchpoint)
    except Exception as error:
        st.warning(f"Morty could not be asked now ({type(error).__name__}). Try again shortly.")
        return
    scoring = 'MP' if matchpoint else 'IMP'
    title, md, facts = search_explain.explain(result, hand, before, scoring)
    morty = facts.get('call')
    with st.container(border=True):
        if morty and played and morty != played:
            st.info(f"At this table the call was {search_explain.show(played)}; Morty would bid "
                    f"{search_explain.show(morty)}. The reasoning below is Morty's.")
        st.markdown(md.split('\n', 1)[1] if md.startswith('# ') else md)       # the page has its own headings
        if FEEDBACK:
            bar(md, facts, hand, by, before, vul, system, scoring)


def bar(md, facts, hand, by, before, vul, system, scoring):
    st.markdown('**Was this explanation useful?**')
    verdict = st.radio('Your view', list(VERDICTS), horizontal=True, index=None, key='fb_verdict',
                       label_visibility='collapsed')
    if verdict is None:
        return
    v = VERDICTS[verdict]
    preferred = statement = None
    if v == 'disagree':
        preferred = st.text_input('Your call (for example 3H, 2N, P, X)', key='fb_call').strip().upper() or None
    if v == 'wrong':
        statement = st.text_input('Which statement is wrong?', key='fb_statement', max_chars=500).strip() or None
    reason = st.text_area('Why? (optional, at most 1,000 characters)', key='fb_reason', max_chars=1000).strip() or None
    tags = st.multiselect('What it is about (optional)', TAGS, key='fb_tags') if v in ('disagree', 'wrong') else []
    st.caption('Your note goes to the people improving Morty. Nothing identifies you: no name, email or address is kept.')
    if not st.button('Send', key='fb_send'):
        return
    if too_many():
        st.warning('That is a lot of notes in a minute; please wait a moment.')
        return
    record = {'kind': 'call', 'verdict': v, 'source': 'auction-explorer', 'route': '/search',
              'params': {'auction': ' '.join(list(before) + ['?']), 'hand': hand, 'vul': south_vul(vul, by),
                         'scoring': scoring, 'system': system},
              'shown': {'version': VERSION, 'digest': hashlib.sha1(md.encode('utf-8')).hexdigest()[:12],
                        'choice': facts.get('call')},
              'preferred': preferred, 'statement': statement, 'reason': reason, 'tags': tags or None,
              'who': visitor_id()}
    record = {k: val for k, val in record.items() if val not in (None, [], '')}
    status, answer = send(record)
    if status == 200:
        st.session_state.setdefault('feedback_times', []).append(time.time())
        st.success(f"Thank you. Recorded as {answer.get('id')}.")
    elif status == 429:
        st.warning('The feedback store is busy; please try again in a minute.')
    else:
        st.error(f"Not recorded: {answer.get('error', status)}")
