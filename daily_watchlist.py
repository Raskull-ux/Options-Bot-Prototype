# Daily watchlist -> Discord messages.
#
#   Top bullish / top bearish names (scored 0-9 on Taz's rules), each with
#   resistance/support levels (day H/L, 20d H/L, 8/10/20/50/200 SMA, 60-day
#   fibs), an "above X calls / below Y puts" idea with targets and
#   invalidation, and a real option contract quote 7-14 days out.
#   Core watchlist with levels. RSI exhaustion watch (80 / 85 / 90+ tiers).
#   Every ranked idea is logged and graded over the next 5 sessions; a weekly
#   scorecard reports how the ideas actually did.
import re
from datetime import date, timedelta

import numpy as np
import pandas as pd

import config
from lib import technicals as ta
from lib.earnings_data import load_earnings
from lib.symbol_filter import load_excluded_symbols

OCC = re.compile(r"^(?P<root>.+?)(?P<exp>\d{6})(?P<cp>[CP])(?P<strike>\d{8})$")  # parse from the fixed-width tail


def fmt(p: float) -> str:
    return f"{p:,.2f}"


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def load_frames(bars: pd.DataFrame) -> dict[str, pd.DataFrame]:
    cols = [c for c in ("open", "high", "low", "close", "volume") if c in bars.columns]
    return {s: g.set_index("date")[cols].sort_index() for s, g in bars.groupby("symbol")}


def universe_symbols() -> set:
    uni = set(pd.read_csv(config.UNIVERSE_MEMBERSHIP_FILE)["symbol"])
    excluded, have = load_excluded_symbols()
    return uni - excluded if have else uni


def upcoming_earnings(today: pd.Timestamp) -> dict[str, str]:
    e = load_earnings()
    if e.empty:
        return {}
    e = e.assign(d=pd.to_datetime(e["earnings_date"], errors="coerce"))
    e = e[(e["d"] > today) & (e["d"] <= today + timedelta(days=config.WL_EARNINGS_WARN_DAYS))]
    return {r.symbol: f"{r.d:%a %m/%d}{' ' + r.hour if isinstance(r.hour, str) and r.hour else ''}" for r in e.itertuples()}


# ---------------------------------------------------------------------------
# Option contract suggestion (real indicative quotes from Alpaca)
# ---------------------------------------------------------------------------
def pick_contract(client, symbol: str, side: str, strike_near: float, today: date) -> tuple[str | None, str]:
    """Nearest-strike contract 7-14 days out whose bid/ask spread is at most
    WL_MAX_SPREAD of its mid. Returns (text, status)."""
    try:
        snaps = client.get_option_snapshots(symbol, strike_gte=round(strike_near * 0.93, 2),
                                            strike_lte=round(strike_near * 1.07, 2))
    except Exception:
        return None, "no_quotes"
    want = "C" if side == "bull" else "P"
    lo, hi = today + timedelta(days=config.WL_EXPIRY_MIN_DAYS), today + timedelta(days=config.WL_EXPIRY_MAX_DAYS)
    cands = []
    for sym, snap in snaps.items():
        m = OCC.match(sym)
        if not m or m["cp"] != want:
            continue
        exp = date(2000 + int(m["exp"][:2]), int(m["exp"][2:4]), int(m["exp"][4:]))
        if not lo <= exp <= hi:
            continue
        q = snap.get("latestQuote") or {}
        bid, ask = float(q.get("bp") or 0), float(q.get("ap") or 0)
        if bid <= 0 or ask <= bid:
            continue
        mid = (bid + ask) / 2
        if (ask - bid) / mid > config.WL_MAX_SPREAD:
            continue
        k = int(m["strike"]) / 1000
        if abs(k / strike_near - 1) > config.WL_MAX_STRIKE_DIST:
            continue
        cands.append((abs(k - strike_near), abs((exp - today).days - 10), exp, k, bid, ask))
    if not cands:
        return None, "too_wide"
    _, _, exp, k, bid, ask = min(cands)
    return f"{exp:%m/%d} {k:g}{want} ~${(bid + ask) / 2:.2f} ({bid:.2f}/{ask:.2f})", "ok"


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------
def lvl_txt(xs: list, n: int = 3) -> str:
    return " · ".join(f"{fmt(p)} ({lab})" for p, lab in xs[:n]) or "none nearby"


def merged_gaps(m: dict) -> list[dict]:
    out: dict = {}
    for g in m.get("gaps", []):
        key = (round(g["lo"], 2), round(g["hi"], 2))
        if key in out:
            e = out[key]
            e["tf"] = "+".join(sorted(set(e["tf"].split("+")) | {g["tf"]}))
            e["skipped"] = e["skipped"] or g["skipped"]
        else:
            out[key] = dict(g)
    return list(out.values())


def gap_txt(m: dict, n: int = 2, below_only: bool = False) -> str:
    c, gs = m["close"], merged_gaps(m)
    above = [] if below_only else sorted([g for g in gs if g["lo"] > c], key=lambda g: g["lo"])[:n]
    below = sorted([g for g in gs if g["hi"] < c], key=lambda g: -g["hi"])[:n]
    inside = [] if below_only else [g for g in gs if g["lo"] <= c <= g["hi"]]
    f = lambda g, a: f"{a}{fmt(g['lo'])}–{fmt(g['hi'])} {g['tf']}{' ⭐' if g['skipped'] else ''}"
    parts = [f(g, "in ") for g in inside[:1]] + [f(g, "↑") for g in above] + [f(g, "↓") for g in below]
    return ("Gaps below: " if below_only else "Open gaps: ") + (" · ".join(parts) if parts else "none")


def lw_txt(m: dict) -> str:
    """Last week's close shown as a LEVEL. The touch-trigger version failed as a
    signal in testing (about 50/50), so the bot no longer says puts/calls on it."""
    lw = m.get("lw_close", np.nan)
    if np.isnan(lw):
        return ""
    return f"Last wk close {fmt(lw)} ({'below' if m['close'] > lw else 'above'} price)"


def core_line(sym: str, m: dict, res: list, sup: list) -> str:
    b, _ = ta.score(m, "bull")
    br, _ = ta.score(m, "bear")
    bias = "bullish" if b - br >= 3 else "bearish" if br - b >= 3 else "neutral"
    up = res[0] if res else (m["close"] + m["atr"], "+1 ATR")
    dn = sup[0] if sup else (m["close"] - m["atr"], "-1 ATR")
    return (f"**{sym} {fmt(m['close'])}** ({m['close'] / m['prev_close'] - 1:+.1%}) {bias} (bull {b}/9, bear {br}/9) · RSI {m['rsi']:.0f} · {ta.td_text(m['td'])}\n"
            f"  Above {fmt(up[0])} ({up[1]}) → calls · Below {fmt(dn[0])} ({dn[1]}) → puts · "
            f"10/20/50 SMA {fmt(m['sma10'])}/{fmt(m['sma20'])}/{fmt(m['sma50'])}\n"
            f"  {gap_txt(m, 1)}" + (f" · {lw_txt(m)}" if lw_txt(m) else ""))


def chunk(header: str, blocks: list[str], limit: int = 1900, sep: str = "\n\n") -> list[str]:
    msgs, cur = [], header
    for b in blocks:
        if len(cur) + len(b) + len(sep) > limit:
            msgs.append(cur)
            cur = b
        else:
            cur += sep + b
    msgs.append(cur)
    return msgs


# ---------------------------------------------------------------------------
# Idea log + grading
# ---------------------------------------------------------------------------
IDEA_COLS = ["date", "symbol", "side", "score", "close", "trigger", "t1", "t2", "invalid", "contract",
             "status", "triggered_date", "resolved_date"]


def grade_ideas(ideas: pd.DataFrame, frames: dict, max_sessions: int) -> pd.DataFrame:
    """Walk each open idea forward on daily bars for up to max_sessions.
    Outcomes: t2 | t1 (hit T1, then stopped or expired: half sold at T1 per the
    framework) | stopped (invalidated before T1) | triggered_no_target | no_trigger.
    Conservative: if invalidation and a target print on the same day, the stop
    is assumed to have come first."""
    ideas = ideas.copy()
    for i, r in ideas[ideas["status"] == "open"].iterrows():
        f = frames.get(r["symbol"])
        if f is None:
            continue
        after = f[f.index > pd.Timestamp(r["date"])].iloc[:max_sessions]
        if after.empty:
            continue
        bull = r["side"] == "bull"
        trig_day, hit_t1, status = None, False, None
        for d, b in after.iterrows():
            if trig_day is None:
                if not ((b["high"] >= r["trigger"]) if bull else (b["low"] <= r["trigger"])):
                    continue
                trig_day = d
            stop = (b["low"] <= r["invalid"]) if bull else (b["close"] >= r["invalid"])  # puts: daily close above
            if stop:
                status = "t1" if hit_t1 else "stopped"
                break
            if (b["high"] >= r["t2"]) if bull else (b["low"] <= r["t2"]):
                status = "t2"
                break
            if (b["high"] >= r["t1"]) if bull else (b["low"] <= r["t1"]):
                hit_t1 = True
        ideas.at[i, "triggered_date"] = trig_day.strftime("%Y-%m-%d") if trig_day is not None else None
        if status is None:
            if len(after) < max_sessions:
                continue  # still inside its tracking window
            status = "t1" if hit_t1 else ("triggered_no_target" if trig_day is not None else "no_trigger")
        ideas.at[i, "status"] = status
        ideas.at[i, "resolved_date"] = after.index[-1].strftime("%Y-%m-%d") if status in (
            "t1", "triggered_no_target", "no_trigger") and len(after) >= max_sessions else d.strftime("%Y-%m-%d")
    return ideas


def scorecard(ideas: pd.DataFrame, since: pd.Timestamp | None = None) -> str:
    done = ideas[ideas["status"] != "open"]
    if since is not None:
        done = done[pd.to_datetime(done["resolved_date"]) >= since]
    if done.empty:
        return "📊 **Scorecard:** no ideas resolved yet (each idea is tracked for 5 sessions)."
    lines = ["📊 **Watchlist scorecard** (each idea tracked 5 sessions; same-day stop+target counts as stopped)"]
    for lab, g in [("All", done), ("Bullish", done[done["side"] == "bull"]), ("Bearish", done[done["side"] == "bear"])]:
        if g.empty:
            continue
        trig = g[g["status"] != "no_trigger"]
        if trig.empty:
            lines.append(f"**{lab}:** {len(g)} ideas, none triggered")
            continue
        n = len(trig)
        t1 = trig["status"].isin(["t1", "t2"]).sum()
        lines.append(f"**{lab}:** {len(g)} ideas · {n} triggered ({n / len(g):.0%}) · "
                     f"T1+ {t1 / n:.0%} · T2 {(trig['status'] == 't2').sum() / n:.0%} · "
                     f"stopped {(trig['status'] == 'stopped').sum() / n:.0%} · no target {(trig['status'] == 'triggered_no_target').sum() / n:.0%}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------
def put_targets(m: dict, trigger: float) -> list[tuple[float, str]]:
    """Taz's targets: the next levels below -- prior day's low, gap fills, last
    week's close, daily 8 SMA, then 20/50 SMA and the 20-day low. Each target
    must sit at least WL_MIN_TARGET_STEP below the one before it."""
    c = m["close"]
    lv = [(m["prev_low"], "prior day low"), (m["lw_close"], "last wk close"),
          (m["sma8"], "8 SMA"), (m["sma20"], "20 SMA"), (m["sma50"], "50 SMA"), (m["lo20"], "20d low")]
    for g in merged_gaps(m):
        if g["hi"] < c:
            star = " ⭐" if g["skipped"] else ""
            lv += [(g["hi"], f"{g['tf']} gap top{star}"), (g["lo"], f"{g['tf']} gap fill{star}")]
    lv = sorted([(p, lab) for p, lab in lv if p == p and p < trigger], reverse=True)
    out, ref = [], trigger
    for p, lab in lv:
        if p <= ref * (1 - config.WL_MIN_TARGET_STEP):
            out.append((p, lab))
            ref = p
        if len(out) == 2:
            break
    return out


def put_features(m: dict) -> tuple[dict, list[str], tuple | None]:
    """Every put confirmation as a yes/no flag, plus display labels. Shared by
    the live scanner and backtest_research.py so both judge setups identically."""
    rej = ta.rejection(m)
    gaps_below = [g for g in merged_gaps(m) if g["hi"] < m["close"]]
    ex = ta.exhaustion(dict(m, rsi=m["rsi_max5"]))
    td_side, td_n = m["td"]
    vol_x = m["volume"] / m["vol20"] if m["vol20"] > 0 else 0
    f = {
        "rsi80": ex is not None,
        "stretched": m["ext_atr_max5"] >= config.WL_STRETCH_ATR,
        "rsi_div": bool(m["rsi_div"]),
        "macd_div": bool(m["macd_div"]),
        "below10": m["close"] < m["sma10"],
        "lower_high": bool(m["lower_high"]),
        "trend": m["sma10"] < m["sma20"] or m["sma10"] < m["sma50"],
        "rej": rej is not None,
        "rej_key": rej is not None and any(k in rej[1] for k in ("gap", "50 SMA", "200 SMA")),
        "failed_gap": bool(m.get("failed_gap_up")),
        "red_vol": bool(m["red"] and vol_x >= 1.5),
        "gap_below": bool(gaps_below),
        "td9": td_side == "sell" and 9 <= td_n <= 14,
        "td_buy9": td_side == "buy" and td_n >= 9,
        "at_top": m["close"] >= 0.90 * m["hi60"] and m["rsi_max10"] >= 70,
        "near_high": m["close"] >= 0.95 * m["hi60"],
    }
    labels = [
        (f["rsi80"], f"RSI {m['rsi_max5']:.0f} {ex['tier'] if ex else ''} (5d)".strip()),
        (f["stretched"], f"stretched {m['pct_above_sma20_max5']:+.0%} over 20 SMA (5d peak)"),
        (f["rsi_div"], "RSI bear div"),
        (f["macd_div"], "MACD bear div"),
        (f["below10"], "close < 10 SMA"),
        (f["lower_high"], "lower high"),
        (f["trend"], "trend breaker" if m["sma10"] < m["sma50"] else "10<20"),
        (f["rej"], f"rejected at {fmt(rej[0])} ({rej[1]})" if rej else ""),
        (f["failed_gap"], "failed gap-up"),
        (f["red_vol"], f"red on {vol_x:.1f}x vol"),
        (f["gap_below"], "gap below ⭐" if any(g["skipped"] for g in gaps_below) else "gap below"),
        (f["td9"], "TD sell 9 ✓"),
    ]
    hits = [lab for ok, lab in labels if ok]
    f["score12"] = len(hits)
    f["core4"] = int(f["rsi_div"]) + int(f["macd_div"]) + int(f["rsi80"]) + int(f["td9"])
    f["door"] = ("exhaustion" if f["at_top"] else None) or ("resistance" if f["rej"] and f["gap_below"] else None) \
        or ("failed gap" if f["failed_gap"] else None)
    return f, hits, rej


def _exh_core(f: dict) -> bool:
    return (f["rsi_div"] or f["macd_div"]) and (f["rsi80"] or f["td9"])


# Candidate put rules. Defined ONCE here; the live scanner uses config.WL_PUT_RULE
# and backtest_research.py scores all of them on the same data.
PUT_RULES = {
    "v1_live":           lambda f: f["door"] is not None and f["score12"] >= config.WL_MIN_PUT_SCORE,
    "exh_core":          _exh_core,
    "exh_core_rolled":   lambda f: _exh_core(f) and f["below10"],
    "div_both":          lambda f: f["rsi_div"] and f["macd_div"],
    "div_rsi80":         lambda f: (f["rsi_div"] or f["macd_div"]) and f["rsi80"],
    "div_near_high":     lambda f: (f["rsi_div"] or f["macd_div"]) and f["near_high"],
    "exh_core_keylevel": lambda f: _exh_core(f) and f["rej_key"],
    "exh_core_no_tdbuy": lambda f: _exh_core(f) and not f["td_buy9"],
}


def put_geometry(m: dict, rej) -> dict:
    """Trigger = rejection low (day low); invalid = the rejected level, the gap
    (failed gap-up), or the day high; targets = Taz's next levels below."""
    trigger = m["day_low"]
    if rej is not None:
        invalid, inv_lab = rej[0], rej[1]
    elif m.get("failed_gap_up"):
        invalid, inv_lab = m["prev_body_hi"], "back above the gap"
    else:
        invalid, inv_lab = m["day_high"], "day high"
    targets = put_targets(m, trigger)
    rr = (trigger - targets[0][0]) / (invalid - trigger) if targets and invalid > trigger else np.nan
    return {"trigger": trigger, "invalid": invalid, "inv_lab": inv_lab, "targets": targets, "rr": rr}


def put_setup(m: dict, rule: str | None = None) -> dict | None:
    rule = rule or config.WL_PUT_RULE
    f, hits, rej = put_features(m)
    if not PUT_RULES[rule](f):
        return None
    rolling = f["below10"] or f["lower_high"] or f["failed_gap"] or (f["rej"] and m["red"])
    score = f["score12"] if rule == "v1_live" else f["core4"]
    return {"door": f["door"] or "exhaustion core", "score": score, "score_of": 12 if rule == "v1_live" else 4,
            "hits": hits, "features": f, "stage": "Rolling over" if rolling else "At the top — not broken yet",
            "td_buy9": f["td_buy9"], "rej": rej, **put_geometry(m, rej)}


def put_lw_txt(m: dict) -> str:
    lw = m.get("lw_close", np.nan)
    if lw != lw:
        return ""
    return f"Last wk close {fmt(lw)} overhead (resistance)" if lw > m["close"] else f"Last wk close {fmt(lw)} below (target)"


def put_block(sym: str, m: dict, st: dict, contract: str | None, earn: str | None) -> str:
    t = st["targets"]
    tgt = ", ".join(f"T{i + 1} {fmt(p)} ({lab})" for i, (p, lab) in enumerate(t)) or "no clean level below — trail it"
    lines = [f"🔻 **{sym} {fmt(m['close'])}** ({m['close'] / m['prev_close'] - 1:+.1%}) — **{st['stage']}** · {st['score']}/{st['score_of']}{' exhaustion' if st['score_of'] == 4 else ''} · {ta.td_text(m['td'])}",
             " · ".join(h for h in st["hits"] if h != "failed gap-up"),
             gap_txt(m, 2, below_only=True) + (f" | {put_lw_txt(m)}" if put_lw_txt(m) else ""),
             f"**Below {fmt(st['trigger'])} (rejection low) → puts**" + (f" ({contract})" if contract else "")
             + f". {tgt}. Invalid: daily close above {fmt(st['invalid'])} ({st['inv_lab']})."
             + (f" Reward/risk to T1: {st['rr']:.1f}" if st["rr"] == st["rr"] else "")]
    if st["td_buy9"]:
        lines.append("⚠ TD buy 9 in — bounce risk")
    if earn:
        lines.append(f"⚠ earnings {earn}")
    return "\n".join(lines)


def scan_universe(frames: dict, spy) -> list[str]:
    """Core watchlist + the WL_UNIVERSE_TOP most-traded names (20-day dollar volume)."""
    uni = universe_symbols()
    dv = []
    for s in uni:
        f = frames.get(s)
        if f is None or len(f) < 25:
            continue
        x = f.iloc[-20:]
        if x["close"].iloc[-1] < config.WL_MIN_PRICE:
            continue
        dv.append(((x["close"] * x["volume"]).mean(), s))
    top = [s for _, s in sorted(dv, reverse=True)[:config.WL_UNIVERSE_TOP]]
    return list(dict.fromkeys(config.CORE_WATCHLIST + top))


def build_messages(client, today: pd.Timestamp) -> list[str]:
    bars = pd.read_csv(config.BARS_FILE, parse_dates=["date"])
    core_missing = [s for s in config.CORE_WATCHLIST if s not in set(bars["symbol"])]
    if core_missing and client is not None:
        raw = client.get_daily_bars(core_missing, lookback_days=300, feed=config.STOCK_BARS_FEED)
        extra = pd.DataFrame([{"symbol": s, "date": pd.Timestamp(b["t"][:10]), "open": b["o"], "high": b["h"], "low": b["l"],
                               "close": b["c"], "volume": b["v"]} for s, bl in raw.items() for b in bl])
        bars = pd.concat([bars, extra], ignore_index=True)
    frames = load_frames(bars)
    spy = frames["SPY"]["close"] if "SPY" in frames else None
    earn = upcoming_earnings(today)
    nxt = today + pd.offsets.BDay(1)
    names = scan_universe(frames, spy)
    mets = {}
    for s in names:
        f = frames.get(s)
        m = ta.metrics(f, spy) if f is not None else None
        if m is not None and m["date"] == today:
            mets[s] = m

    # ---- put setups
    cands = []
    for s, m in mets.items():
        if s in ("SPY", "QQQ"):
            continue  # index ETFs live in the core watchlist
        st = put_setup(m)
        if st is None or not st["targets"]:
            continue
        if not (st["rr"] == st["rr"] and st["rr"] >= config.WL_MIN_RR):
            continue  # too little room to T1 for the distance to invalidation
        if s in earn:
            continue  # reports inside the option's life: IV crush (our backtest: -23% to -48% median)
        rr = st["rr"] if st["rr"] == st["rr"] else -1
        cands.append((st["score"], st["stage"] == "Rolling over", rr, s, m, st))
    cands.sort(key=lambda x: (x[0], x[1], x[2]), reverse=True)
    blocks, rows, skipped_wide = [], [], []
    for sc, _, _, s, m, st in cands:
        if len(blocks) == config.WL_TOP_N:
            break
        contract = None
        if client is not None:
            contract, status = pick_contract(client, s, "bear", st["trigger"], nxt.date())
            if contract is None:
                skipped_wide.append(s)
                continue
        blocks.append(put_block(s, m, st, contract, None))
        t = st["targets"] + [(np.nan, "")] * 2
        rows.append({"date": today.strftime("%Y-%m-%d"), "symbol": s, "side": "bear", "score": sc, "close": m["close"],
                     "trigger": st["trigger"], "t1": t[0][0], "t2": t[1][0] if t[1][0] == t[1][0] else t[0][0],
                     "invalid": st["invalid"], "contract": contract, "status": "open",
                     "triggered_date": None, "resolved_date": None})
    head = f"🔻 **Top put setups** — game plan for {nxt:%a %b %d} ({len(cands)} names qualified out of {len(mets)} scanned)"
    if skipped_wide:
        blocks.append(f"_Skipped, no tight contract near the trigger: {', '.join(skipped_wide[:8])}_")
    msgs = chunk(head, blocks) if blocks else [head + "\nNo put setups passed today."]

    # ---- core watchlist (both directions)
    core = []
    for s in config.CORE_WATCHLIST:
        m = mets.get(s)
        if m is None:
            core.append(f"**{s}**: no data today")
            continue
        res, sup = ta.split_levels(m, ta.levels(m))
        core.append(core_line(s, m, res, sup))
    msgs += chunk("📋 **Core watchlist**", core)

    # ---- exhaustion watch
    ex = [(m["rsi"], s, m, ta.exhaustion(m)) for s, m in mets.items() if ta.exhaustion(m)]
    ex.sort(key=lambda x: x[0], reverse=True)
    if ex:
        lines = [f"**{s} {fmt(m['close'])}** RSI {r:.1f} **{e['tier']}** · {m['pct_above_sma20']:+.0%} vs 20 SMA · {ta.td_text(m['td'])}"
                 + (" · RSI div" if m["rsi_div"] else "") + (" · MACD div" if m["macd_div"] else "")
                 for r, s, m, e in ex[:config.WL_EXHAUSTION_N]]
        msgs += chunk("🔥 **Exhaustion watch** (RSI 80 early · 85 elevated · 90+ extreme)", lines, sep="\n")

    # (Failed gap-up section removed: failed testing -- stocks went UP on average.)

    # ---- log + grade + weekly scorecard
    try:
        ideas = pd.read_csv(config.WL_IDEAS_FILE)
    except FileNotFoundError:
        ideas = pd.DataFrame(columns=IDEA_COLS)
    ideas = ideas[ideas["date"] != today.strftime("%Y-%m-%d")]
    ideas = grade_ideas(ideas, frames, config.WL_TRACK_SESSIONS)
    ideas = pd.concat([ideas, pd.DataFrame(rows, columns=IDEA_COLS)], ignore_index=True)
    ideas.to_csv(config.WL_IDEAS_FILE, index=False)
    if today.dayofweek == config.WL_SCORECARD_WEEKDAY:
        msgs.append(scorecard(ideas))
    return msgs
