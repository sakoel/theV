"""
Short-term setup alerts (runs next to Early V; same email settings/secrets).

Setups (5-min / 15-min / 1-hour data from Yahoo):
  NBIS_945   9:45: NBIS above its 9:30 open -> LONG, below -> SHORT. Exit 10:15.
  MU_945     9:45: MU above its 9:30 open -> LONG, below -> SHORT. Exit 15:00.
  MARA_1100  11:00: MARA above its 9:30 open -> LONG, below -> SHORT. Exit 14:00.
  NBIS_SPIKE NBIS 15-min bar with volume > 3x its 20-bar average: green -> LONG, red -> SHORT. Exit 30 min later.
  MU_1H      MU 1-hour bar closes above its upper Bollinger Band (20,2) or RSI(14) crosses above 70 -> LONG.
             Hold 14 hourly bars (about 2 trading days). Long only (the short side lost money).

Cooldown (checked automatically from each setup's own recent signals, whether you traded them or not):
  HALF SIZE : the last 3 signals were all losers (normal bad luck happens; don't quit, shrink).
  COOLDOWN  : last 15 signals average below zero, OR the setup is down more than 2x its worst backtest drop.
              Alerts still go out tagged PAPER ONLY so you can watch it.
  Back to ACTIVE: last 10 signals average above zero with 55%+ winners.

  python setups.py --am       # 9:45-12:00 ET events (GitHub job #1)
  python setups.py --pm       # 12:00-16:00 ET events (GitHub job #2)
  python setups.py --status   # print the cooldown table, no email
  add --dry to print emails instead of sending.
Research tool only, not financial advice. Nothing here places orders.
"""
import sys, time, argparse, datetime as dt
import numpy as np, pandas as pd, yfinance as yf
from early_v import send_email, log, ET

SEND_SHORTS = False     # long only. Set True to also get SHORT/PUTS alerts
COST = 0.05             # % per trade
LATE_OK = dt.timedelta(minutes=5)   # skip an alert if the job is more than 5 min late for it
SIZE = {"NBIS_945": "20-25%", "MU_945": "~50%", "MARA_1100": "~50%", "NBIS_SPIKE": "20-25%", "MU_1H": "35-50%"}
WORST_DD = {("NBIS_945", 1): 7.2, ("NBIS_945", -1): 3.6, ("MU_945", 1): 3.5, ("MU_945", -1): 4.3,
            ("MARA_1100", 1): 1.7, ("MARA_1100", -1): 4.5, ("NBIS_SPIKE", 1): 5.8, ("NBIS_SPIKE", -1): 4.8,
            ("MU_1H", 1): 17.1}
NAME = {"NBIS_945": "NBIS 9:45 open-direction (exit 10:15)", "MU_945": "MU 9:45 open-direction (exit 15:00)",
        "MARA_1100": "MARA 11:00 open-direction (exit 14:00)", "NBIS_SPIKE": "NBIS 15-min volume spike (exit +30 min)",
        "MU_1H": "MU 1-hour breakout (hold ~2 days)"}
WINDOWS = {"NBIS_945": ("NBIS", "09:45", "10:15"), "MU_945": ("MU", "09:45", "15:00"), "MARA_1100": ("MARA", "11:00", "14:00")}

# ---------------- data ----------------
def dl(t, period, interval):
    for a in range(4):
        try:
            x = yf.download(t, period=period, interval=interval, progress=False, auto_adjust=True)
            if isinstance(x.columns, pd.MultiIndex): x.columns = x.columns.get_level_values(0)
            x = x.dropna()
            if len(x):
                if x.index.tz is not None: x.index = x.index.tz_convert(ET)
                return x
        except Exception as e:
            log("download error", t, interval, e)
        time.sleep(15 * (a + 1))
    return pd.DataFrame()

def to15(x5):
    g = x5.resample("15min", label="left", closed="left").agg(
        {"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"})
    return g.dropna()

# ---------------- setup rules (same code for history and live) ----------------
def window_trades(x5, T, X, before):
    """Open-direction trades from 5-min bars on days before `before`."""
    out = []
    x = x5[x5.index.date < before]
    for d, g in x.groupby(x.index.date):
        g = g.set_axis(g.index.strftime("%H:%M"))
        if "09:30" not in g.index or T not in g.index: continue
        side = np.sign(g.Open[T] - g.Open["09:30"])
        pX = g.Close.iloc[-1] if X == "16:00" else g.Open.get(X, np.nan)
        if side == 0 or np.isnan(pX): continue
        out.append((pd.Timestamp(d), int(side), side * (pX / g.Open[T] - 1) * 100 - COST))
    return out

def spike_trades(x15, hold=2):
    """Returns (taken entries [(ts, side, ret or None)])."""
    v = x15.Volume; vr = v / v.rolling(20).mean()
    c = x15.Close.values; o = x15.Open.values; idx = x15.index; N = len(x15)
    last_of_day = pd.Series(np.arange(N), index=idx).groupby(idx.date).transform("max").values
    out, nxt = [], -1
    for i in np.nonzero((vr.values > 3) & (c != o))[0]:
        if i < 20 or i < nxt: continue
        side = 1 if c[i] > o[i] else -1
        live = idx[i].date() == idx[-1].date()          # today's session may not be finished
        e = i + hold if live else min(i + hold, last_of_day[i])
        if e <= i: continue                             # spike on the day's last bar: no trade
        ret = side * (c[e] / c[i] - 1) * 100 - COST if e < N else None
        out.append((idx[i], side, ret)); nxt = e
    return out

def mu1h_trades(h, hold=14):
    c = h.Close
    d = c.diff(); up = d.clip(lower=0).ewm(alpha=1/14).mean(); dn = (-d.clip(upper=0)).ewm(alpha=1/14).mean()
    rsi = 100 - 100 / (1 + up / dn)
    bbu = c.rolling(20).mean() + 2 * c.rolling(20).std()
    sig = ((c > bbu) & (c.shift() <= bbu.shift())) | ((rsi > 70) & (rsi.shift() <= 70))
    cv = c.values; out, nxt = [], -1
    for i in np.nonzero(sig.values)[0]:
        if i < 50 or i < nxt: continue
        e = i + hold
        out.append((h.index[i], 1, (cv[e] / cv[i] - 1) * 100 - COST if e < len(cv) else None)); nxt = e
    return out

# ---------------- cooldown ----------------
def status(trades, key, side):
    r = np.array([t[2] for t in trades if t[1] == side and t[2] is not None])
    if len(r) < 10: return "ACTIVE", f"only {len(r)} past signals"
    eq = np.cumprod(1 + r / 100); dd = (eq[-1] / eq.max() - 1) * 100
    m15, m10, w10 = r[-15:].mean(), r[-10:].mean(), (r[-10:] > 0).mean()
    streak = 0
    for x in r[::-1]:
        if x > 0: break
        streak += 1
    limit = 2 * max(WORST_DD.get((key, side), 5), 5)
    info = (f"last 10: {w10*100:.0f}% win, avg {m10:+.2f}% | last 15 avg {m15:+.2f}% | "
            f"losing streak {streak} | down {dd:.1f}% from peak (cooldown at -{limit:.0f}%)")
    if (m15 < 0 or dd < -limit) and not (m10 > 0 and w10 >= 0.55): return "COOLDOWN", info
    if streak >= 3: return "HALF SIZE", info
    return "ACTIVE", info

def history(today):
    H = {}
    for k, (t, T, X) in WINDOWS.items():
        H[k] = window_trades(dl(t, "60d", "5m"), T, X, today)
    n5 = dl("NBIS", "60d", "5m"); n5 = n5[n5.index.date < today]
    H["NBIS_SPIKE"] = [x for x in spike_trades(to15(n5)) if x[2] is not None]
    h = dl("MU", "730d", "1h"); h = h[h.index.date < today]
    H["MU_1H"] = [x for x in mu1h_trades(h) if x[2] is not None]
    S = {}
    for k, tr in H.items():
        for side in ((1, -1) if k != "MU_1H" else (1,)):
            S[(k, side)] = status(tr, k, side)
    return S

def status_table(S):
    lines = []
    for (k, side), (lab, info) in S.items():
        if side == -1 and not SEND_SHORTS: continue
        lines.append(f"{lab:9s} {NAME[k]} {'LONG' if side == 1 else 'SHORT'}\n          {info}")
    return "\n".join(lines)

# ---------------- alerts ----------------
def tag_size(S, k, side):
    lab = S.get((k, side), ("ACTIVE", ""))[0]
    size = SIZE[k]
    if lab == "HALF SIZE": size = f"half of {SIZE[k]}"
    if lab == "COOLDOWN": size = "0% (PAPER ONLY)"
    return lab, size

def alert(S, k, side, price, entry_note, exit_note, dry, extra=""):
    if side == -1 and not SEND_SHORTS: return
    lab, size = tag_size(S, k, side)
    word = "LONG" if side == 1 else "SHORT/PUTS"
    t = WINDOWS.get(k, ("NBIS" if k == "NBIS_SPIKE" else "MU",))[0]
    subj = f"{word} {t} @ ~${price:.2f} | {NAME[k].split(' (')[0]} | {lab}"
    body = (f"{NAME[k]}\n\n{entry_note}\nExit: {exit_note}\nSize: {size} of the account\n"
            f"Setup status: {lab} — {S.get((k, side), ('', ''))[1]}\n{extra}\n"
            "Research tool only, not financial advice.")
    send_email(subj, body, dry)

def wait_until(day, hhmm, sec=20, replay=False):
    target = dt.datetime.combine(day, dt.time(*map(int, hhmm.split(":"))), ET) + dt.timedelta(seconds=sec)
    if replay: return True
    now = dt.datetime.now(ET)
    if now > target + LATE_OK:
        log(f"too late for {hhmm} event — skipped"); return False
    while dt.datetime.now(ET) < target: time.sleep(5)
    return True

def price_now(t, day, upto):
    """open at 9:30 and last price at time `upto` (1-min data)."""
    x = dl(t, "5d", "1m"); x = x[x.index.date == day]
    if not len(x): return None, None
    o = float(x[x.index.time >= dt.time(9, 30)].Open.iloc[0])
    y = x[x.index + pd.Timedelta(minutes=1) <= upto]
    return o, (float(y.Close.iloc[-1]) if len(y) else None)

def direction_event(S, k, day, dry, replay):
    t, T, X = WINDOWS[k]
    if not wait_until(day, T, replay=replay): return
    upto = pd.Timestamp(dt.datetime.combine(day, dt.time(*map(int, T.split(":"))), ET))
    o, p = price_now(t, day, upto)
    if p is None: log(f"no data for {t}"); return
    side = 1 if p > o else -1 if p < o else 0
    if side == 0: log(f"{t} flat vs open — no trade"); return
    alert(S, k, side, p, f"{t} is {(p/o-1)*100:+.2f}% vs its 9:30 open (${o:.2f}). Enter now near ${p:.2f}.",
          f"{X} ET (you'll get an EXIT email)", dry)

def exit_event(S, k, day, dry, replay):
    t, T, X = WINDOWS[k]
    if not wait_until(day, X, sec=0, replay=replay): return
    o, pT = price_now(t, day, pd.Timestamp(dt.datetime.combine(day, dt.time(*map(int, T.split(":"))), ET)))
    _, pX = price_now(t, day, pd.Timestamp(dt.datetime.combine(day, dt.time(*map(int, X.split(":"))), ET)))
    if pT is None or pX is None or pT == o: return
    side = 1 if pT > o else -1
    if side == -1 and not SEND_SHORTS: return
    r = side * (pX / pT - 1) * 100
    send_email(f"EXIT {t} now (~${pX:.2f}) | {NAME[k].split(' (')[0]} {r:+.2f}%",
               f"Close the {'LONG' if side == 1 else 'SHORT/PUTS'} {t} from {T} (entry ~${pT:.2f}). Result about {r:+.2f}%.", dry)

def spike_event(S, day, close_hhmm, dry, replay):
    if not wait_until(day, close_hhmm, sec=40, replay=replay): return
    now = pd.Timestamp(dt.datetime.combine(day, dt.time(*map(int, close_hhmm.split(":"))), ET))
    x5 = dl("NBIS", "10d", "5m"); x5 = x5[x5.index + pd.Timedelta(minutes=5) <= now]
    x15 = to15(x5)
    if not len(x15) or x15.index[-1] + pd.Timedelta(minutes=15) != now: log("15-min bar not ready"); return
    tr = spike_trades(x15)
    if tr and tr[-1][0] == x15.index[-1]:
        _, side, _ = tr[-1]; b = x15.iloc[-1]
        vr = b.Volume / x15.Volume.tail(20).mean()
        ex = (now + pd.Timedelta(minutes=30)).strftime("%H:%M")
        alert(S, "NBIS_SPIKE", side, float(b.Close),
              f"15-min bar {x15.index[-1].strftime('%H:%M')} had {vr:.1f}x normal volume and closed "
              f"{'green' if side == 1 else 'red'} (${b.Open:.2f} -> ${b.Close:.2f}). Enter now.",
              f"{ex} ET (you'll get an EXIT email)", dry,
              "NOTE: this is the opening 15-min bar, so it usually matches the NBIS 9:45 alert — "
              "if both say the same direction it's ONE trade, don't double the size." if close_hhmm == "09:45" else "")
        return (ex, side, float(b.Close))

def spike_exit(day, pending, dry, replay):
    ex, side, entry = pending
    if not wait_until(day, ex, sec=0, replay=replay): return
    _, p = price_now("NBIS", day, pd.Timestamp(dt.datetime.combine(day, dt.time(*map(int, ex.split(":"))), ET)))
    if p is None or (side == -1 and not SEND_SHORTS): return
    send_email(f"EXIT NBIS now (~${p:.2f}) | volume spike {side*(p/entry-1)*100:+.2f}%",
               f"Close the NBIS volume-spike {'LONG' if side == 1 else 'SHORT/PUTS'} (entry ~${entry:.2f}).", dry)

def mu1h_event(S, day, close_hhmm, dry, replay):
    if not wait_until(day, close_hhmm, sec=60, replay=replay): return
    now = pd.Timestamp(dt.datetime.combine(day, dt.time(*map(int, close_hhmm.split(":"))), ET))
    h = dl("MU", "730d", "1h")
    h = h[(h.index + pd.Timedelta(hours=1) <= now) | ((h.index.date == day) & (h.index.time == dt.time(15, 30)) & (now.time() >= dt.time(16, 0)))]
    if not len(h) or h.index[-1].date() != day: log("MU 1h bar not ready"); return
    tr = mu1h_trades(h)
    if tr and tr[-1][0] == h.index[-1]:
        p = float(h.Close.iloc[-1])
        ex_day = (pd.Timestamp(day) + pd.offsets.BDay(2)).date()
        alert(S, "MU_1H", 1, p,
              f"MU 1-hour bar {h.index[-1].strftime('%H:%M')} closed at ${p:.2f}: above its upper Bollinger Band "
              f"or RSI crossed above 70. Enter now.",
              f"about {close_hhmm} ET on {ex_day:%a %b %d} (14 hourly bars; no exit email for this one — set a reminder)", dry)

def run(session, day, dry, replay=False):
    log(f"setups {session} for {day}")
    S = history(day)
    if session == "am":
        send_email(f"Setups status {day:%b %d}", "Today's setup status (cooldown rules in setups.py):\n\n" + status_table(S), dry)
    ev = []
    if session == "am":
        ev += [("09:45", lambda: direction_event(S, "NBIS_945", day, dry, replay)),
               ("09:45", lambda: direction_event(S, "MU_945", day, dry, replay)),
               ("10:15", lambda: exit_event(S, "NBIS_945", day, dry, replay)),
               ("11:00", lambda: direction_event(S, "MARA_1100", day, dry, replay))]
        spikes = ["09:45", "10:00", "10:15", "10:30", "10:45", "11:00", "11:15", "11:30", "11:45"]
        hours = ["10:30", "11:30"]
    else:
        ev += [("14:00", lambda: exit_event(S, "MARA_1100", day, dry, replay)),
               ("15:00", lambda: exit_event(S, "MU_945", day, dry, replay))]
        spikes = ["12:00", "12:15", "12:30", "12:45", "13:00", "13:15", "13:30", "13:45", "14:00", "14:15",
                  "14:30", "14:45", "15:00", "15:15", "15:30"]   # 15:45 bar would exit after the close
        hours = ["12:30", "13:30", "14:30", "15:30", "16:00"]
    pending = []
    def sp(hm):
        r = spike_event(S, day, hm, dry, replay)
        if r: pending.append(r)
    ev += [(hm, (lambda hm=hm: sp(hm))) for hm in spikes]
    ev += [(hm, (lambda hm=hm: mu1h_event(S, day, hm, dry, replay))) for hm in hours]
    done_exits = set()
    for hm, f in sorted(ev, key=lambda e: e[0]):
        for p in [p for p in pending if p[0] <= hm and p[0] not in done_exits]:
            spike_exit(day, p, dry, replay); done_exits.add(p[0])
        try: f()
        except Exception as e: log(f"event {hm} failed:", e)
    for p in [p for p in pending if p[0] not in done_exits]:
        spike_exit(day, p, dry, replay)
    log("done")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--am", action="store_true"); ap.add_argument("--pm", action="store_true")
    ap.add_argument("--status", action="store_true"); ap.add_argument("--dry", action="store_true")
    ap.add_argument("--replay", help="YYYY-MM-DD (last ~5 days): print what would have fired, no email")
    a = ap.parse_args()
    today = dt.datetime.now(ET).date()
    if a.status:
        print(status_table(history(today))); sys.exit()
    if a.replay:
        d = dt.date.fromisoformat(a.replay); run("am", d, True, True); run("pm", d, True, True); sys.exit()
    if today.weekday() >= 5: log("weekend"); sys.exit()
    now_t = dt.datetime.now(ET).time()
    try:
        if a.am:
            if now_t > dt.time(11, 50): log("started too late (or duplicate run) — exiting"); sys.exit()
            run("am", today, a.dry)
        elif a.pm:
            if now_t > dt.time(15, 55): log("started too late (or duplicate run) — exiting"); sys.exit()
            run("pm", today, a.dry)
    except Exception as e:
        send_email("Setups script crashed", f"{type(e).__name__}: {e}", a.dry); raise
