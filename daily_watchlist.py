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
    return {r.symbol: f"{r.d:%a}{' ' + r.hour if isinstance(r.hour, str) and r.hour else ''}" for r in e.itertuples()}


# ---------------------------------------------------------------------------
# Option contract suggestion (real indicative quotes from Alpaca)
# ---------------------------------------------------------------------------
def pick_contract(client, symbol: str, side: str, strike_near: float, today: date) -> str | None:
    try:
        snaps = client.get_option_snapshots(symbol, strike_gte=round(strike_near * 0.95, 2),
                                            strike_lte=round(strike_near * 1.05, 2))
    except Exception:
        return None
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
        if ask <= 0:
            continue
        k = int(m["strike"]) / 1000
        cands.append((abs(k - strike_near), abs((exp - today).days - 10), exp, k, bid, ask))
    if not cands:
        return None
    _, _, exp, k, bid, ask = min(cands)
    ks = f"{k:g}"
    px = f"~${(bid + ask) / 2:.2f} (bid {bid:.2f} / ask {ask:.2f})" if bid > 0 else f"ask ${ask:.2f}"
    return f"{exp:%m/%d} {ks}{want} {px}"


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------
def lvl_txt(xs: list, n: int = 3) -> str:
    return " · ".join(f"{fmt(p)} ({lab})" for p, lab in xs[:n]) or "none nearby"


def gap_txt(m: dict, n: int = 2) -> str:
    c, gs = m["close"], m.get("gaps", [])
    above = sorted([g for g in gs if g["lo"] > c], key=lambda g: g["lo"])[:n]
    below = sorted([g for g in gs if g["hi"] < c], key=lambda g: -g["hi"])[:n]
    inside = [g for g in gs if g["lo"] <= c <= g["hi"]]
    f = lambda g, a: f"{a}{fmt(g['lo'])}–{fmt(g['hi'])} {g['tf']}{' ⭐' if g['skipped'] else ''}"
    parts = [f(g, "in ") for g in inside[:1]] + [f(g, "↑") for g in above] + [f(g, "↓") for g in below]
    return "Open gaps: " + (" · ".join(parts) if parts else "none")


def lw_txt(m: dict) -> str:
    lw = m.get("lw_close", np.nan)
    if np.isnan(lw):
        return ""
    return (f"Last wk close {fmt(lw)}: touch from above → puts" if m["close"] > lw
            else f"Last wk close {fmt(lw)}: touch from below → calls")


def ticker_block(sym: str, m: dict, side: str, sc: int, reasons: list, idea: dict, res: list, sup: list,
                 contract: str | None, earn: str | None) -> str:
    icon, word = ("🟢", "bullish") if side == "bull" else ("🔴", "bearish")
    day = m["close"] / m["prev_close"] - 1
    t, t1, t2, inv, fl = idea["trigger"], idea["t1"], idea["t2"], idea["invalid"], idea["flip_target"]
    go, flip = ("Above", "calls") if side == "bull" else ("Below", "puts")
    other_go, other = ("Below", "puts") if side == "bull" else ("Above", "calls")
    lines = [f"{icon} **{sym} {fmt(m['close'])}** ({day:+.1%}) — {word} {sc}/9" + (f"  ⚠ earnings {earn}" if earn else ""),
             " · ".join(reasons + [ta.td_text(m["td"])]),
             gap_txt(m) + (f" | {lw_txt(m)}" if lw_txt(m) else ""),
             f"R: {lvl_txt(res)}",
             f"S: {lvl_txt(sup)}",
             f"**{go} {fmt(t[0])} → {flip}**" + (f" ({contract})" if contract else "")
             + f". T1 {fmt(t1[0])} ({t1[1]}), T2 {fmt(t2[0])} ({t2[1]}). Invalid {('below' if side == 'bull' else 'above')} {fmt(inv[0])}.",
             f"{other_go} {fmt(inv[0])} → {other} toward {fmt(fl[0])} ({fl[1]})."]
    return "\n".join(lines)


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
            stop = (b["low"] <= r["invalid"]) if bull else (b["high"] >= r["invalid"])
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

    # rank the universe
    uni = universe_symbols()
    scored = []
    for s in uni:
        f = frames.get(s)
        if f is None:
            continue
        m = ta.metrics(f, spy)
        if m is None or m["date"] != today or m["close"] < config.WL_MIN_PRICE or m["dollar_vol20"] < config.WL_MIN_DOLLAR_VOL:
            continue
        scored.append((s, m))
    msgs = []
    rows = []
    for side, title in (("bull", "🟢 **Top bullish**"), ("bear", "🔴 **Top bearish**")):
        ranked = []
        for s, m in scored:
            sc, why = ta.score(m, side)
            if sc >= config.WL_MIN_SCORE:
                rs = abs(m["rs20"]) if not np.isnan(m["rs20"]) else 0
                volx = m["volume"] / m["vol20"] if m["vol20"] > 0 else 0
                ranked.append((sc, rs, volx, s, m, why))
        ranked.sort(key=lambda x: (x[0], x[1], x[2]), reverse=True)
        blocks = []
        for sc, _, _, s, m, why in ranked[:config.WL_TOP_N]:
            res, sup = ta.split_levels(m, ta.levels(m))
            idea = ta.idea(m, side, res, sup)
            contract = pick_contract(client, s, side, idea["trigger"][0], nxt.date()) if client is not None else None
            blocks.append(ticker_block(s, m, side, sc, why, idea, res, sup, contract, earn.get(s)))
            rows.append({"date": today.strftime("%Y-%m-%d"), "symbol": s, "side": side, "score": sc, "close": m["close"],
                         "trigger": idea["trigger"][0], "t1": idea["t1"][0], "t2": idea["t2"][0],
                         "invalid": idea["invalid"][0], "contract": contract, "status": "open",
                         "triggered_date": None, "resolved_date": None})
        if blocks:
            msgs += chunk(f"{title} — game plan for {nxt:%a %b %d} ({len(ranked)} names scored {config.WL_MIN_SCORE}+/9)", blocks)
        else:
            msgs.append(f"{title}: no names scored {config.WL_MIN_SCORE}+/9 today.")

    # core watchlist
    core = []
    for s in config.CORE_WATCHLIST:
        f = frames.get(s)
        m = ta.metrics(f, spy) if f is not None else None
        if m is None:
            core.append(f"**{s}**: not enough data")
            continue
        res, sup = ta.split_levels(m, ta.levels(m))
        core.append(core_line(s, m, res, sup))
    msgs += chunk("📋 **Core watchlist**", core)

    # exhaustion watch (universe + core), RSI tiers 80 / 85 / 90+
    ex = []
    seen = set()
    for s, m in scored + [(s, ta.metrics(frames[s], spy)) for s in config.CORE_WATCHLIST if s in frames]:
        if m is None or s in seen:
            continue
        seen.add(s)
        e = ta.exhaustion(m)
        if e:
            ex.append((m["rsi"], s, m, e))
    ex.sort(reverse=True, key=lambda x: x[0])
    if ex:
        lines = [f"**{s} {fmt(m['close'])}** RSI {r:.1f} **{e['tier']}** · {e['extension_atr']:+.1f} ATR above 20 SMA · {ta.td_text(m['td'])}"
                 + (" · RSI divergence" if e["divergence"] else "") for r, s, m, e in ex[:config.WL_EXHAUSTION_N]]
        msgs += chunk("🔥 **Exhaustion watch** (RSI 80 early · 85 elevated · 90+ extreme)", lines, sep="\n")

    failed = [(s, m) for s, m in scored + [(s, ta.metrics(frames[s], spy)) for s in config.CORE_WATCHLIST if s in frames]
              if m is not None and m.get("failed_gap_up")]
    seen_f, flines = set(), []
    for s, m in failed:
        if s in seen_f:
            continue
        seen_f.add(s)
        flines.append(f"**{s} {fmt(m['close'])}** gapped above its 20d high and filled back down → **put trigger** · "
                      f"invalid above {fmt(m['day_high'])} (day high) · {ta.td_text(m['td'])}")
    if flines:
        msgs += chunk("⚠️ **Failed gap-ups** (gap above resistance that filled back down)", flines[:8], sep="\n")

    # log + grade
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
