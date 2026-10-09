"""
Check Taz's QQQ examples against the real daily data (free, ~1 minute).
For each date: QQQ close vs the 10/20/50/200-day SMAs from 3 days before to 10 days after,
whether the rule "closes below the 10, 20 AND 50, and two days earlier was NOT below all three"
fires, how bunched the three averages are, and what happened over the next 10 / 20 days.
Out: results/examples/report.md
"""
import os
import pandas as pd
import setup_backtest as sb

DATES = [("2021-09-17", "dumped 10+20 together, closed under 50, rallied 3-4 days, retest of 10 = trigger"),
         ("2022-01-04", "started the slide; MAs far above the 200"),
         ("2023-03-09", "broke them but into a golden cross -> rallied (should be skipped)"),
         ("2023-10-18", "touched the 50 and 10 from below, closed red, next day under the 20"),
         ("2024-04-04", "broke all three in one day, bounced and held (precursor)"),
         ("2024-04-15", "broke all three in one day, flushed after"),
         ("2024-08-30", "green hammer at the 10, next day broke the 20 and 50"),
         ("2025-02-21", "broke them, got wrecked")]


def main():
    os.makedirs("results/examples", exist_ok=True)
    d = sb.fetch(["QQQ"], "1Day", "2020-01-01", (pd.Timestamp.now(tz="UTC") - pd.Timedelta(minutes=20)).isoformat())
    d = d.assign(day=d.t.dt.tz_localize(None).dt.normalize()).set_index("day")[["o", "h", "l", "c"]]
    for n in (10, 20, 50, 200):
        d[f"s{n}"] = d.c.rolling(n).mean()
    below3 = (d.c < d.s10) & (d.c < d.s20) & (d.c < d.s50)
    d["rule"] = below3 & ~below3.shift(2, fill_value=False)
    d["bunch_pct"] = (d[["s10", "s20", "s50"]].max(axis=1) - d[["s10", "s20", "s50"]].min(axis=1)) / d.c * 100
    d["s50_vs_200_pct"] = (d.s50 / d.s200 - 1) * 100
    L = ["# Your QQQ examples, checked against the data\n",
         "Below = closed under that average. RULE = closed under the 10, 20 and 50 today, and was not under all three two days earlier.\n"]
    for ds, note in DATES:
        t = pd.Timestamp(ds)
        if t not in d.index:
            t = d.index[d.index.searchsorted(t)]
        k = d.index.get_loc(t)
        w = d.iloc[max(0, k - 3):k + 11]
        fut = d.c.iloc[k + 1:k + 21]
        L += [f"\n## {ds} — {note}\n",
              f"- 50 SMA vs 200 SMA: {d.s50_vs_200_pct.iloc[k]:+.1f}%  |  10/20/50 bunched within {d.bunch_pct.iloc[k]:.1f}% of price",
              f"- Next 10 days: {(d.c.iloc[min(k+10,len(d)-1)]/d.c.iloc[k]-1)*100:+.1f}%  |  next 20 days: {(d.c.iloc[min(k+20,len(d)-1)]/d.c.iloc[k]-1)*100:+.1f}%  |  "
              f"lowest close within 20 days: {(fut.min()/d.c.iloc[k]-1)*100:+.1f}%\n",
              "| Day | Close | 10 SMA | 20 SMA | 50 SMA | 200 SMA | Below 10/20/50 | RULE |",
              "|---|---|---|---|---|---|---|---|"]
        for day, r in w.iterrows():
            mark = "**→**" if day == t else ""
            b = "".join(x if r.c < r[f"s{n}"] else "·" for x, n in (("10 ", 10), ("20 ", 20), ("50", 50)))
            L.append(f"| {mark}{day:%a %b %d %Y} | {r.c:.2f} | {r.s10:.2f} | {r.s20:.2f} | {r.s50:.2f} | {r.s200:.2f} | {b} | {'FIRES' if r.rule else ''} |")
    open("results/examples/report.md", "w").write("\n".join(L) + "\n"); print("\n".join(L))


if __name__ == "__main__":
    main()
