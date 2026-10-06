"""
EARLY V alerts (final version) + overnight SPY.

Morning (run with no flags, start any time before 9:30 ET):
  Universe : from the stock list below, names over $20 trading $500M+ a day (20-day avg),
             then the 60 with the widest average daily range over the prior 20 sessions.
  Knife    : stock trades at least 1.5% below its 9:30 open.
  V        : bounce, pullback of 30%+ of the bounce without a new low, then a 5-min close above
             the bounce high. Signal bar starts 9:45 at the latest (entry 9:50 at the latest).
  Filters  : stock above VWAP; SPY and QQQ above VWAP; QQQ at least 0.5% above its morning low.
  Ranking  : skip gap-ups > 1%, lagging theme first, one per theme, quieter opening volume. Top 3.
  HIGH CONVICTION tag: market volatile/down (SPY 20d vol > 18% or below 200-day EMA) and the
             stock gapped down more than 2% (62% win, about +2%/trade in the test; consider 1.5x size).
  Plan     : 1/3 of the account per pick. Noon: at/below entry -> sell; green -> hold to the close.

Afternoon (--overnight, runs at 3:50 ET):
  If Early V had no signal today AND SPY closed above its 200-day EMA -> email BUY SPY at the close,
  sell at tomorrow's open.

Large-cap realistic backtest (Nov 2021-Oct 2026, 0.1% costs): Early V about +0.4%/trade, ~46% win,
about 11%/yr; with overnight SPY about 17-18%/yr, max drawdown about -19%. Expect less live.
Research tool only, not financial advice. Nothing here places orders.

  python early_v.py                 # live morning run (emails)
  python early_v.py --overnight     # 3:50 PM overnight-SPY check
  python early_v.py --test-email
  python early_v.py --replay 2026-10-02   # what would have fired (last ~30 days), no email
"""
import os, sys, time, smtplib, argparse, datetime as dt
from email.message import EmailMessage
from zoneinfo import ZoneInfo
import pandas as pd
import yfinance as yf

# ---------------- settings ----------------
EMAIL_FROM = os.environ.get("EV_EMAIL_FROM", "you@gmail.com")
EMAIL_APP_PASSWORD = os.environ.get("EV_EMAIL_PASSWORD", "xxxx xxxx xxxx xxxx")
EMAIL_TO = os.environ.get("EV_EMAIL_TO", "you@gmail.com")
SMTP_HOST, SMTP_PORT = "smtp.gmail.com", 465
SEND_NOON_CHECK = True          # email a HOLD/SELL note per position at noon
KNIFE = 0.015                   # 1.5% below the open
PULLBACK = 0.30                 # pullback of at least 30% of the bounce
LAST_SIGNAL_BAR = dt.time(9, 45)  # signal bar start time; entry is the next bar (9:50 at the latest)
ET = ZoneInfo("America/New_York")

UNIV = """NVDA AMD TSLA META AMZN GOOGL MSFT AAPL NFLX AVGO MU SMCI PLTR COIN MSTR HOOD SHOP CRWD SNOW DDOG NET PANW ZS MDB UBER ABNB DASH RBLX AFRM SOFI UPST ROKU TTD ARM MRVL LRCX AMAT KLAC ON ANET DELL HPE CELH ENPH FSLR RIVN LCID NIO BABA PDD JD XPEV MARA RIOT CLSK APP RDDT IONQ RGTI SOUN OKLO VST CEG LLY NVO UNH BA CAT DE GS JPM XOM OXY DKNG CVNA W CHWY ETSY PINS SNAP U PATH S TEAM INTC QCOM TXN ADBE CRM ORCL NOW INTU
ACHR JOBY ASTS LUNR RKLB HIMS NU DUOL CAVA ELF ONON DECK LULU NKE SBUX CMG MCD WMT COST TGT HD LOW DIS WBD SPOT ZM DOCU OKTA TWLO ESTC FTNT ALAB CRDO COHR VRT ETN GEV SMR NNE CCJ LEU MRNA BNTX REGN VRTX ISRG DXCM PODD TMDX NTRA V MA PYPL XYZ AXP C BAC WFC MS SCHW IBKR FCX NEM AA CLF SLB HAL DVN EOG F GM STLA LI ZETA BBAI TEM CRWV NBIS CIEN""".split()


THEMES = {'AI/semis':'NVDA AMD AVGO MU SMCI ARM MRVL LRCX AMAT KLAC ON ANET INTC QCOM TXN ALAB CRDO COHR CRWV NBIS DELL HPE VRT CIEN',
 'Crypto':'COIN MSTR MARA RIOT CLSK HOOD','Quantum':'IONQ RGTI SOUN BBAI','Space/drones':'RKLB ASTS LUNR ACHR JOBY',
 'Nuclear/power':'OKLO SMR NNE LEU CCJ VST CEG GEV ETN',
 'Software':'PLTR SHOP CRWD SNOW DDOG NET PANW ZS MDB PATH S TEAM ADBE CRM ORCL NOW INTU OKTA TWLO ESTC FTNT DOCU ZM U APP TTD RDDT ZETA TEM',
 'Fintech':'SOFI AFRM UPST PYPL XYZ NU','EV/auto':'TSLA RIVN LCID NIO XPEV LI F GM STLA','China':'BABA PDD JD',
 'Consumer/internet':'META AMZN GOOGL MSFT AAPL NFLX UBER ABNB DASH RBLX ROKU DKNG CVNA W CHWY ETSY PINS SNAP SPOT DUOL CAVA ELF ONON DECK LULU NKE SBUX CMG MCD WMT COST TGT HD LOW DIS WBD CELH',
 'Health':'LLY NVO UNH HIMS MRNA BNTX REGN VRTX ISRG DXCM PODD TMDX NTRA','Energy/materials':'XOM OXY SLB HAL DVN EOG FCX NEM AA CLF FSLR ENPH',
 'Finance/industrial':'BA CAT DE GS JPM V MA AXP C BAC WFC MS SCHW IBKR'}
REGIME_VOLATILE = False
THEME_OF = {s: k for k, v in THEMES.items() for s in v.split()}

# ---------------- helpers ----------------
def log(*a): print(dt.datetime.now(ET).strftime("%H:%M:%S"), *a, flush=True)

def send_email(subject, body, dry=False):
    if dry:
        log("EMAIL (dry run) |", subject); print(body); return
    msg = EmailMessage(); msg["Subject"] = subject; msg["From"] = EMAIL_FROM; msg["To"] = EMAIL_TO
    msg.set_content(body)
    for attempt in range(3):
        try:
            with smtplib.SMTP_SSL(SMTP_HOST, SMTP_PORT, timeout=30) as s:
                s.login(EMAIL_FROM, EMAIL_APP_PASSWORD); s.send_message(msg)
            log("emailed:", subject); return
        except Exception as e:
            log("email failed:", e); time.sleep(5)

def vwap(x):
    tp = (x.High + x.Low + x.Close) / 3
    return (tp * x.Volume).cumsum() / x.Volume.cumsum()

def build_universe(day):
    """Large caps from UNIV (> $20, $500M+/day), top 60 by 20-day average range, plus context."""
    d = yf.download(UNIV + ["SPY"], period="1y", interval="1d", group_by="ticker", progress=False, threads=True, auto_adjust=False)
    d.index = pd.to_datetime(d.index).date
    rng, prev_close = {}, {}
    # theme strength: avg 5-day return of the theme's stocks minus SPY's, using closes before today
    global THEME_5D
    def r5(t):
        x = d[t].Close.dropna(); x = x[x.index < day]
        return float(x.iloc[-1] / x.iloc[-6] - 1) if len(x) > 6 else None
    spy5 = r5("SPY") or 0.0; THEME_5D = {}
    for k, v in THEMES.items():
        vals = [r for r in (r5(t) for t in v.split() if t in d.columns.get_level_values(0)) if r is not None]
        if vals: THEME_5D[k] = (sum(vals) / len(vals) - spy5) * 100
    global REGIME_VOLATILE
    spy = d["SPY"].Close.dropna(); spy = spy[spy.index < day]
    vol20 = spy.pct_change().tail(20).std() * (252 ** 0.5) * 100
    REGIME_VOLATILE = bool(vol20 > 18 or spy.iloc[-1] < spy.ewm(span=200).mean().iloc[-1])
    for t in UNIV:
        try:
            x = d[t].dropna(); x = x[x.index < day]
            r = ((x.High - x.Low) / x.Close.shift()).dropna().tail(20)
            dv = (x.Close * x.Volume).tail(20).mean()
            if len(r) >= 15 and x.Close.iloc[-1] > 20 and dv > 5e8:
                rng[t] = r.mean(); prev_close[t] = float(x.Close.iloc[-1])
        except Exception: pass
    s = pd.Series(rng).sort_values(ascending=False)
    top = list(s.index[:60])
    return top, {t: float(rng[t]) for t in top}, prev_close

def opening_volume_baseline(tickers, day, until=dt.time(9, 45)):
    """Average volume from 9:30 up to `until` over recent sessions (first 15 min by default; 30 min for Opening Drive)."""
    d = yf.download(tickers, period="30d", interval="5m", group_by="ticker", progress=False, threads=True)
    d.index = d.index.tz_convert(ET); out = {}
    for t in tickers:
        try:
            x = d[t].dropna(); x = x[(x.index.date < day) & (x.index.time < until)]
            v = x.Volume.groupby(x.index.date).sum().tail(20)
            if len(v) >= 5: out[t] = float(v.mean())
        except Exception: pass
    return out

def fetch_today(tickers, day, now=None):
    d = yf.download(tickers + ["SPY", "QQQ"], period="5d" if now is None else "30d", interval="5m",
                    group_by="ticker", progress=False, threads=True)
    d.index = d.index.tz_convert(ET); bars = {}
    for t in tickers + ["SPY", "QQQ"]:
        try:
            x = d[t].dropna(); x = x[x.index.date == day]
            x = x[x.index.time >= dt.time(9, 30)]
            if now is not None: x = x[x.index + pd.Timedelta(minutes=5) <= now]   # completed bars only
            if len(x): bars[t] = x
        except Exception: pass
    return bars

def find_signals(bars, tickers):
    spy, qqq = bars.get("SPY"), bars.get("QQQ")
    if spy is None or qqq is None or not len(spy) or not len(qqq): return []
    sv, qv = vwap(spy), vwap(qqq); qlow = qqq.Low.cummin(); out = []
    for t in tickers:
        g = bars.get(t)
        if g is None or len(g) < 2: continue
        o = float(g.Open.iloc[0]); gv = vwap(g); lo = float("inf"); hi = None; H = None
        for ts, b in g.iterrows():
            if ts.time() > LAST_SIGNAL_BAR: break
            if b.Low < lo:
                lo = float(b.Low); hi = lo; H = None; continue          # new low: restart the V
            if hi is None: continue
            hi = max(hi, float(b.High))
            if H is None:
                if hi > lo and (hi - b.Low) >= PULLBACK * (hi - lo): H = hi   # higher low forming
            elif b.Close > H:                                               # higher high
                ok = (lo / o - 1 <= -KNIFE and ts in sv.index and ts in qv.index and b.Close > gv[ts]
                      and spy.Close[ts] > sv[ts] and qqq.Close[ts] > qv[ts] and qqq.Close[ts] >= qlow[ts] * 1.005)
                if ok:
                    out.append(dict(ticker=t, signal_bar=ts, close=float(b.Close), low=lo, open=o,
                                    knife=(lo / o - 1) * 100, bounce_high=float(H)))
                break                                                       # one attempt per stock per day
    return out

def describe(s, n, atr_pct, prev_close, vol_base, bars):
    t = s["ticker"]; g = bars[t]
    gap = (s["open"] / prev_close[t] - 1) * 100 if t in prev_close else None
    first3 = float(g[g.index.time < dt.time(9, 45)].Volume.sum())
    rv = first3 / vol_base[t] if vol_base.get(t) else None
    entry_bar = (s["signal_bar"] + pd.Timedelta(minutes=5)).strftime("%H:%M")
    risk = (s["close"] - s["low"]) / s["close"] * 100
    th = THEME_OF.get(t, "Other"); tstr = THEME_5D.get(th)
    s["theme"], s["theme5"], s["gap"], s["rv"] = th, tstr, gap, rv
    notes = []
    if gap is not None and gap > 1.0: notes.append(f"gapped UP {gap:+.1f}% before flushing (these tested weaker — consider skipping)")
    if rv is not None and rv > 2: notes.append(f"heavy opening volume ({rv:.1f}x normal) — may be news-driven (tested weaker)")
    s["hc"] = bool(REGIME_VOLATILE and gap is not None and gap < -2)
    if s["hc"]: notes.append(f"HIGH CONVICTION: volatile/down market and gapped down {gap:.1f}% (best-tested setup)")
    if not notes: notes.append("no warning flags")
    subject = f"BUY {t} @ ~${s['close']:.2f}"  # rank tag added in run()
    body = f"""Early V signal #{n} today: {t}

Signal bar: {s['signal_bar'].strftime('%H:%M')} ET (closed above the bounce high ${s['bounce_high']:.2f})
Entry: buy at the {entry_bar} bar open, around ${s['close']:.2f} (signal bar close)

Setup
  Open ${s['open']:.2f}  ->  morning low ${s['low']:.2f}  (knife {s['knife']:.1f}%)
  Distance to the morning low from here: {risk:.1f}%
  Stock's average daily range (20d): {atr_pct*100:.1f}%
  Gap vs yesterday's close: {('%+.1f%%' % gap) if gap is not None else 'n/a'}
  Opening volume vs normal: {('%.1fx' % rv) if rv is not None else 'n/a'}
  Theme: {th} — {('%+.1f%% vs SPY over the past 5 days' % tstr) if tstr is not None else 'n/a'} {'(lagging: tested best)' if (tstr is not None and tstr < -1) else '(hot: tested weaker)' if (tstr is not None and tstr > 1) else ''}
  Flags: {'; '.join(notes)}

Plan
  1. Buy near ${s['close']:.2f}.
  2. Optional emergency stop: below the morning low ${s['low']:.2f}.
  3. Noon check (12:00 ET): at or below your entry -> sell. Green -> hold.
  4. Sell anything still open at the close (3:55-4:00 ET).
  Sizing that tested best: split across the first 3 signals of the day.

Backtest context (large caps, 0.1% costs): about +0.4% per trade, ~46% win; winners avg +3.7%, losers -2.3%.
Market regime today: {'VOLATILE/DOWN (Early V tested ~3x better)' if REGIME_VOLATILE else 'calm'}
Research tool only, not financial advice.
"""
    return subject, body

def noon_check(alerts, day, dry, now=None):
    if not alerts: return
    tick = [a["ticker"] for a in alerts]
    bars = fetch_today(tick, day, now)
    for a in alerts:
        g = bars.get(a["ticker"])
        if g is None: continue
        g = g[g.index.time <= dt.time(11, 55)]
        if not len(g): continue
        px = float(g.Close.iloc[-1]); vw = float(vwap(bars[a["ticker"]]).loc[g.index[-1]])
        entry = a["entry"] if a.get("entry") else a["close"]
        ch = (px / entry - 1) * 100
        action = "HOLD to close" if px > entry else "SELL (not green at noon)"
        subj = f"NOON {a['ticker']}: {action.split(' (')[0]} ({ch:+.1f}%)"
        body = (f"{a['ticker']} at noon: ${px:.2f}, {ch:+.1f}% vs entry ${entry:.2f}. "
                f"{'Above' if px > vw else 'Below'} VWAP (${vw:.2f}).\n\nRule: {action}.\n"
                "If holding, sell at the close (3:55-4:00 ET).")
        send_email(subj, body, dry)


# ---------------- main ----------------
def run(day, dry, replay=False):
    log(f"Early V alerts for {day} ({'replay' if replay else 'live'})")
    top, rng, prev_close = build_universe(day)
    log(f"universe: {len(top)} most volatile names")
    vol_base = opening_volume_baseline(top, day)
    sent, alerts, n = set(), [], 0
    checks = [dt.time(9, 35), dt.time(9, 40), dt.time(9, 45), dt.time(9, 50)]   # bar closes for 9:30..9:45 bars
    for chk in checks:
        target = dt.datetime.combine(day, chk, ET) + dt.timedelta(seconds=20)
        if not replay:
            while dt.datetime.now(ET) < target: time.sleep(5)
        now = pd.Timestamp(target) if replay else pd.Timestamp(dt.datetime.now(ET))
        bars = {}
        for attempt in range(3):
            bars = fetch_today(top, day, now)
            if "SPY" in bars and bars["SPY"].index[-1] + pd.Timedelta(minutes=5) >= now - pd.Timedelta(minutes=1): break
            time.sleep(10 if not replay else 0)
        if "SPY" not in bars:
            log("no market data for today (holiday or data problem) — exiting"); return
        new = [s for s in find_signals(bars, top) if s["ticker"] not in sent]
        made = []
        for s in new:
            n += 1; sent.add(s["ticker"])
            made.append((s,) + describe(s, n, rng[s["ticker"]], prev_close, vol_base, bars))
        # rank everything so far: skip-flags last, lagging themes first, one per theme, quieter volume first
        allsig = alerts + [m[0] for m in made]
        def key(x): return ((x["gap"] or 0) > 1.0, x["theme5"] if x["theme5"] is not None else 0, x["rv"] or 1)
        picks, used = [], set()
        for x in sorted(allsig, key=key):
            if len(picks) < 3 and x["theme"] not in used and not ((x["gap"] or 0) > 1.0):
                picks.append(x["ticker"]); used.add(x["theme"])
        for x in sorted(allsig, key=key):
            if len(picks) < 3 and x["ticker"] not in picks: picks.append(x["ticker"])
        for s, subj, body in sorted(made, key=lambda m: key(m[0])):
            tag = (" | TOP 3 PICK" if s["ticker"] in picks else " | backup") + (" | HIGH CONVICTION" if s.get("hc") else "")
            body += f"\nRanking so far today (lagging theme first, one per theme, gap-ups last): {', '.join(picks)} are the top 3.\n"
            send_email(subj + tag, body, dry); alerts.append(s)
        log(f"checked {chk.strftime('%H:%M')} bar close: {n} signal(s) so far")
    if not alerts:
        send_email("Early V: no signals today", "No Early V signals fired between 9:35 and 9:50 ET.\nThe 3:50 PM check will say whether to hold SPY overnight.", dry)
    if SEND_NOON_CHECK and alerts:
        target = dt.datetime.combine(day, dt.time(12, 0, 30), ET)
        if not replay:
            log("waiting for the noon check…")
            while dt.datetime.now(ET) < target: time.sleep(15)
        noon_check(alerts, day, dry, pd.Timestamp(target) if replay else None)
    log("done for today")

def overnight(day, dry):
    """3:50 PM: buy SPY at the close if Early V had no signal today and SPY > 200-day EMA."""
    target = dt.datetime.combine(day, dt.time(15, 50), ET)
    while dt.datetime.now(ET) < target: time.sleep(15)
    top, rng, prev_close = build_universe(day)
    bars = fetch_today(top, day, pd.Timestamp(dt.datetime.combine(day, dt.time(9, 50, 20), ET)))
    if "SPY" not in bars: log("no market data today — exiting"); return
    had = find_signals(bars, top)
    d = yf.download("SPY", period="2y", interval="1d", progress=False, auto_adjust=False)
    c = d["Close"].squeeze().dropna(); last = float(yf.download("SPY", period="1d", interval="1m", progress=False)["Close"].squeeze().dropna().iloc[-1])
    hist = c[pd.to_datetime(c.index).date < day]; ema = float(pd.concat([hist, pd.Series([last])]).ewm(span=200).mean().iloc[-1])
    if had:
        log("Early V had signals today — no overnight trade"); return
    if last > ema:
        send_email(f"BUY SPY @ ~${last:.2f} | OVERNIGHT", f"""No Early V signal today and SPY (${last:.2f}) is above its 200-day EMA (${ema:.2f}).

Plan: buy SPY at the close (market-on-close or 3:55-4:00 ET), sell at tomorrow's open (9:30 ET).
Size: full Early V allocation.
Backtest: about +0.1% per night on average; adds roughly 6%/yr to Early V.
Research tool only, not financial advice.""", dry)
    else:
        send_email("Overnight: NO SPY (below 200-day EMA)", f"SPY ${last:.2f} is below its 200-day EMA ${ema:.2f}. Stay in cash tonight.", dry)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--replay", help="YYYY-MM-DD: show what would have fired (no email)")
    ap.add_argument("--test-email", action="store_true")
    ap.add_argument("--dry", action="store_true", help="print emails instead of sending")
    ap.add_argument("--overnight", action="store_true", help="3:50 PM overnight SPY check")
    a = ap.parse_args()
    if a.test_email:
        send_email("BUY TEST @ $1.23", "Test email from early_v_alerts.py — if you got this, alerts will reach you."); sys.exit()
    if a.replay:
        run(dt.date.fromisoformat(a.replay), dry=True, replay=True)
    else:
        today = dt.datetime.now(ET).date()
        if today.weekday() >= 5: log("weekend — nothing to do"); sys.exit()
        now_t = dt.datetime.now(ET).time()
        # backup schedules start this script more than once; a copy that starts too late (or after another finished) exits quietly
        if a.overnight:
            if now_t > dt.time(15, 57): log("too late for the overnight check — exiting"); sys.exit()
            overnight(today, dry=a.dry)
        else:
            if now_t > dt.time(9, 50): log("started after 9:50 ET (late or duplicate run) — exiting"); sys.exit()
            run(today, dry=a.dry)
