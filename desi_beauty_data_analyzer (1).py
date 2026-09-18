"""
============================================================================
  DESI BEAUTY DATA ANALYZER  ·  v2 (News + Sentiment)
  A real-time trend dashboard for an Indian beauty / makeup / skincare
  Instagram page.
----------------------------------------------------------------------------
  Sources : Google Trends (pytrends, GEO=IN)
            Google News RSS (current articles) + VADER sentiment
  Output  : Pastel, Instagram-ready 1080x1080 charts you can download & post
  UI      : A local / cloud Streamlit dashboard in your browser
============================================================================

WHAT CHANGED FROM v1
- Reddit is gone. Reddit blocked anonymous data access in 2026, so it kept
  failing on cloud servers. It's replaced by Google News RSS, which is free,
  needs no keys, and works reliably from the cloud.
- New: sentiment analysis. Every headline is scored positive / neutral /
  negative so you can see how people feel about a brand or trend.

SETUP (only if running locally; on Streamlit Cloud requirements.txt handles it)
    pip install streamlit pandas matplotlib pytrends requests feedparser vaderSentiment
    streamlit run desi_beauty_data_analyzer.py
============================================================================
"""

import io
import re
import time
import random
import html
import datetime as dt
from collections import Counter
from urllib.parse import quote_plus

import requests
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm

import streamlit as st

# --- optional third-party libs (import defensively) ---
try:
    from pytrends.request import TrendReq
    PYTRENDS_AVAILABLE = True
except Exception:
    PYTRENDS_AVAILABLE = False

try:
    import feedparser
    FEEDPARSER_AVAILABLE = True
except Exception:
    FEEDPARSER_AVAILABLE = False

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    VADER_AVAILABLE = True
except Exception:
    VADER_AVAILABLE = False


# ---------------------------------------------------------------------------
# 1. CONFIG
# ---------------------------------------------------------------------------

KEYWORD_GROUPS = {
    "Lip Products":       ["kajal", "tinted lip balm", "lip tint", "matte lipstick"],
    "Skincare":           ["clear sunscreen", "kojic acid", "saffron serum"],
    "Cultural Catalysts": ["glass skin", "wedding makeup"],
}

# What we ask Google News for (India-scoped). Edit freely.
NEWS_QUERIES = [
    "Indian skincare trends",
    "Indian makeup trends",
    "glass skin skincare India",
    "sunscreen India beauty",
    "lipstick launch India",
    "beauty brand India launch",
]

BRANDS = [
    "Nykaa", "Tira", "Sugar", "Minimalist", "L'Oreal", "Maybelline", "Lakme",
    "Plum", "Dot & Key", "The Ordinary", "Cetaphil", "Deconstruct", "Foxtale",
    "Renee", "Mamaearth", "Sunscoop", "Aqualogica", "Cosrx", "Innisfree",
]


# ---------------------------------------------------------------------------
# 2. AESTHETIC
# ---------------------------------------------------------------------------

PALETTE = {"bg": "#FFFDF9", "grid": "#EAE3DA", "ink": "#5A5150", "muted": "#A79B97"}
SERIES_COLORS = ["#E19AAE", "#A9C3A0", "#B7A6D6", "#F0B79A", "#9FC0D4",
                 "#CBA0C4", "#E7C98B", "#8FB8AE"]
POS, NEU, NEG = "#A9C3A0", "#B7A6D6", "#E19AAE"   # sage / lavender / rose
SQUARE_IN, SQUARE_DPI = 10.8, 100                 # -> 1080x1080 px


def _pick_font():
    preferred = ["Poppins", "Montserrat", "Nunito Sans", "Segoe UI",
                 "Helvetica Neue", "Arial"]
    installed = {f.name for f in fm.fontManager.ttflist}
    for name in preferred:
        if name in installed:
            return name
    return "DejaVu Sans"


def set_aesthetic_style():
    font = _pick_font()
    plt.rcParams.update({
        "figure.facecolor": PALETTE["bg"], "axes.facecolor": PALETTE["bg"],
        "savefig.facecolor": PALETTE["bg"], "font.family": font,
        "text.color": PALETTE["ink"], "axes.edgecolor": PALETTE["grid"],
        "axes.labelcolor": PALETTE["ink"], "axes.titlecolor": PALETTE["ink"],
        "xtick.color": PALETTE["muted"], "ytick.color": PALETTE["muted"],
        "axes.linewidth": 0.8, "grid.color": PALETTE["grid"],
        "grid.linewidth": 0.6, "figure.dpi": SQUARE_DPI,
    })


def _new_square_fig():
    fig, ax = plt.subplots(figsize=(SQUARE_IN, SQUARE_IN), dpi=SQUARE_DPI)
    fig.subplots_adjust(left=0.14, right=0.94, top=0.84, bottom=0.16)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(True, alpha=0.6)
    ax.set_axisbelow(True)
    return fig, ax


def _title(fig, ax, title, subtitle):
    fig.text(0.14, 0.93, title, fontsize=26, fontweight="bold",
             color=PALETTE["ink"], ha="left")
    fig.text(0.14, 0.885, subtitle, fontsize=13, color=PALETTE["muted"], ha="left")
    fig.text(0.94, 0.05, "@ your.beauty.page", fontsize=11,
             color=PALETTE["muted"], ha="right", style="italic")


def _png(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=SQUARE_DPI, facecolor=PALETTE["bg"])
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# 3. HELPERS
# ---------------------------------------------------------------------------

def _with_retry(fn, tries=3, base_delay=2.0):
    last = None
    for i in range(tries):
        try:
            return fn()
        except Exception as e:
            last = e
            time.sleep(base_delay * (2 ** i) + random.random())
    raise last


def _clean(text):
    text = re.sub(r"<[^>]+>", " ", text or "")
    text = html.unescape(text)
    return re.sub(r"\s+", " ", text).strip()


# ---------------------------------------------------------------------------
# 4. GOOGLE TRENDS (retry + fallback)
# ---------------------------------------------------------------------------

def _fallback_trends(groups, start, end):
    idx = pd.date_range(start=start, end=end, freq="D")
    data = {}
    for kws in groups.values():
        for kw in kws:
            rng = random.Random(sum(ord(c) for c in kw))
            level = rng.randint(25, 70)
            vals = []
            for _ in idx:
                level = max(3, min(100, level + rng.randint(-8, 9)))
                vals.append(level)
            data[kw] = vals
    df = pd.DataFrame(data, index=idx)
    df.index.name = "date"
    return df


@st.cache_data(ttl=3600, show_spinner=False)
def get_trends_data(groups, geo, start, end):
    if not PYTRENDS_AVAILABLE:
        return _fallback_trends(groups, start, end), False
    timeframe = f"{start} {end}"
    frames = []
    try:
        pytrends = TrendReq(hl="en-IN", tz=330, timeout=(10, 25))
        for kws in groups.values():
            def _fetch(kws=kws):
                pytrends.build_payload(kws, cat=0, timeframe=timeframe, geo=geo)
                return pytrends.interest_over_time()
            df = _with_retry(_fetch)
            if df is None or df.empty:
                continue
            df = df.drop(columns=[c for c in ("isPartial",) if c in df.columns])
            frames.append(df)
            time.sleep(1.5)
        if not frames:
            return _fallback_trends(groups, start, end), False
        merged = pd.concat(frames, axis=1)
        merged = merged.loc[:, ~merged.columns.duplicated()]
        merged.index.name = "date"
        return merged, True
    except Exception:
        return _fallback_trends(groups, start, end), False


# ---------------------------------------------------------------------------
# 5. GOOGLE NEWS RSS (free, no key) + fallback
# ---------------------------------------------------------------------------

_FALLBACK_NEWS = [
    ("Nykaa reports strong demand for glass-skin serums this festive season",
     "Business Today", "positive"),
    ("Minimalist launches new SPF 50 clear sunscreen, sells out in days",
     "YourStory", "positive"),
    ("Consumers complain about greasy finish on some tinted lip balms",
     "The Print", "negative"),
    ("Tira expands offline stores as beauty retail booms in India",
     "Economic Times", "positive"),
    ("Sugar Cosmetics matte lipstick range wins praise for shade inclusivity",
     "Vogue India", "positive"),
    ("Dermatologists warn against overusing kojic acid for pigmentation",
     "Indian Express", "negative"),
    ("Wedding makeup trends 2026: dewy skin and bold kajal dominate",
     "Cosmopolitan India", "positive"),
    ("Aqualogica sunscreen faces backlash over white-cast claims",
     "The Quint", "negative"),
    ("Mamaearth parent Honasa posts steady quarter amid beauty slowdown",
     "Mint", "neutral"),
    ("Foxtale vitamin C serum gains traction with Gen Z shoppers",
     "Hindustan Times", "positive"),
    ("Lakme rolls out saffron-infused serum for the Indian market",
     "Financial Express", "neutral"),
    ("Deconstruct and Minimalist lead the budget skincare shake-up",
     "Forbes India", "positive"),
]


def _fallback_articles():
    arts = []
    now = dt.datetime.utcnow()
    for i, (title, source, _lab) in enumerate(_FALLBACK_NEWS):
        arts.append({"title": title, "source": source,
                     "published": (now - dt.timedelta(days=i)).strftime("%d %b %Y"),
                     "link": "", "text": title, "query": "sample"})
    return arts


@st.cache_data(ttl=1800, show_spinner=False)
def get_news(queries, per_query=12):
    """Fetch current Indian beauty articles via Google News RSS.
    Returns (list_of_articles, is_live)."""
    if not FEEDPARSER_AVAILABLE:
        return _fallback_articles(), False

    ua = "Mozilla/5.0 (desi-beauty-data-analyzer)"
    seen, articles = set(), []
    for q in queries:
        url = ("https://news.google.com/rss/search?q="
               + quote_plus(q) + "&hl=en-IN&gl=IN&ceid=IN:en")
        try:
            def _fetch(url=url):
                return feedparser.parse(url, agent=ua)
            feed = _with_retry(_fetch, tries=2, base_delay=2.0)
            for e in feed.entries[:per_query]:
                title = _clean(getattr(e, "title", ""))
                if not title or title in seen:
                    continue
                seen.add(title)
                source = ""
                if getattr(e, "source", None) and getattr(e.source, "title", None):
                    source = e.source.title
                articles.append({
                    "title": title,
                    "source": source,
                    "published": getattr(e, "published", ""),
                    "link": getattr(e, "link", ""),
                    "text": title + ". " + _clean(getattr(e, "summary", "")),
                    "query": q,
                })
            time.sleep(0.6)
        except Exception:
            continue

    if not articles:
        return _fallback_articles(), False
    return articles, True


# ---------------------------------------------------------------------------
# 6. SENTIMENT + AGGREGATION
# ---------------------------------------------------------------------------

def score_articles(articles):
    """Attach a compound sentiment score + label to each article."""
    if VADER_AVAILABLE:
        analyzer = SentimentIntensityAnalyzer()
        for a in articles:
            a["score"] = analyzer.polarity_scores(a["text"])["compound"]
    else:
        # crude keyword fallback if VADER isn't installed
        pos = {"love", "best", "glow", "praise", "wins", "strong", "booms", "gain"}
        neg = {"warn", "backlash", "complain", "greasy", "slowdown", "worst", "avoid"}
        for a in articles:
            t = a["text"].lower()
            a["score"] = 0.4 * sum(w in t for w in pos) - 0.4 * sum(w in t for w in neg)
    for a in articles:
        s = a["score"]
        a["label"] = "positive" if s >= 0.05 else "negative" if s <= -0.05 else "neutral"
    return articles


def brand_stats(articles, brands, min_articles=1):
    """Per-brand article volume + mean sentiment."""
    rows = []
    for b in brands:
        bl = b.lower()
        hits = [a for a in articles if bl in a["text"].lower()]
        if len(hits) >= min_articles:
            mean = sum(a["score"] for a in hits) / len(hits)
            rows.append({"brand": b, "volume": len(hits), "sentiment": mean})
    return sorted(rows, key=lambda r: r["volume"], reverse=True)


def overall_mix(articles):
    c = Counter(a["label"] for a in articles)
    return c.get("positive", 0), c.get("neutral", 0), c.get("negative", 0)


# ---------------------------------------------------------------------------
# 7. CHARTS
# ---------------------------------------------------------------------------

def chart_interest_over_time(df, group, kws):
    set_aesthetic_style()
    fig, ax = _new_square_fig()
    for i, kw in enumerate(kws):
        if kw in df.columns:
            ax.plot(df.index, df[kw], label=kw, linewidth=2.6,
                    color=SERIES_COLORS[i % len(SERIES_COLORS)],
                    solid_capstyle="round")
    ax.set_ylabel("Search interest", fontsize=12)
    ax.set_ylim(0, 105)
    fig.autofmt_xdate(rotation=25)
    leg = ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14),
                    ncol=min(len(kws), 3), frameon=False, fontsize=11)
    for t in leg.get_texts():
        t.set_color(PALETTE["ink"])
    _title(fig, ax, group, "Google Trends · India · search interest over time")
    return _png(fig)


def chart_trending_now(df, top_n=10):
    set_aesthetic_style()
    latest = df.iloc[-1].sort_values(ascending=True).tail(top_n)
    fig, ax = _new_square_fig()
    ax.grid(True, axis="x", alpha=0.6); ax.grid(False, axis="y")
    colors = [SERIES_COLORS[i % len(SERIES_COLORS)] for i in range(len(latest))]
    bars = ax.barh(latest.index, latest.values, color=colors, height=0.62)
    for bar, v in zip(bars, latest.values):
        ax.text(bar.get_width() + 1.5, bar.get_y() + bar.get_height() / 2,
                f"{int(v)}", va="center", fontsize=11, color=PALETTE["muted"])
    ax.set_xlim(0, 108)
    ax.set_xlabel("Search interest (latest)", fontsize=12)
    _title(fig, ax, "Trending Right Now",
           "Google Trends · India · latest search interest")
    return _png(fig)


def chart_news_volume(rows, top_n=10):
    set_aesthetic_style()
    fig, ax = _new_square_fig()
    ax.grid(True, axis="x", alpha=0.6); ax.grid(False, axis="y")
    rows = rows[:top_n][::-1]
    if not rows:
        ax.text(0.5, 0.5, "No brand mentions in news today",
                transform=ax.transAxes, ha="center", va="center",
                fontsize=16, color=PALETTE["muted"])
    else:
        labels = [r["brand"] for r in rows]
        vals = [r["volume"] for r in rows]
        colors = [SERIES_COLORS[i % len(SERIES_COLORS)] for i in range(len(rows))]
        bars = ax.barh(labels, vals, color=colors, height=0.62)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_width() + max(vals) * 0.02,
                    bar.get_y() + bar.get_height() / 2, str(v),
                    va="center", fontsize=11, color=PALETTE["muted"])
        ax.margins(x=0.12)
    ax.set_xlabel("Articles mentioning brand", fontsize=12)
    _title(fig, ax, "Who's In The News",
           "Google News · India · brand share of coverage")
    return _png(fig)


def chart_news_sentiment(rows, top_n=10):
    set_aesthetic_style()
    fig, ax = _new_square_fig()
    ax.grid(True, axis="x", alpha=0.6); ax.grid(False, axis="y")
    rows = rows[:top_n][::-1]
    if not rows:
        ax.text(0.5, 0.5, "Not enough coverage to score",
                transform=ax.transAxes, ha="center", va="center",
                fontsize=16, color=PALETTE["muted"])
    else:
        labels = [r["brand"] for r in rows]
        vals = [r["sentiment"] for r in rows]
        colors = [POS if v >= 0.05 else NEG if v <= -0.05 else NEU for v in vals]
        ax.barh(labels, vals, color=colors, height=0.62)
        ax.axvline(0, color=PALETTE["grid"], linewidth=1.2)
        ax.set_xlim(-1, 1)
    ax.set_xlabel("Average sentiment  (left = negative · right = positive)", fontsize=12)
    _title(fig, ax, "How People Feel",
           "Google News · India · avg sentiment per brand")
    return _png(fig)


# ---------------------------------------------------------------------------
# 8. STREAMLIT DASHBOARD
# ---------------------------------------------------------------------------

def _dl(label, png, name):
    st.download_button(label, data=png, file_name=name, mime="image/png",
                       use_container_width=True)


def main():
    st.set_page_config(page_title="Desi Beauty Data Analyzer",
                       page_icon="💄", layout="wide")
    st.markdown("""<style>
        .stApp{background-color:#FFFDF9;}
        h1,h2,h3,h4{color:#5A5150;}
        section[data-testid="stSidebar"]{background-color:#FCE9EC;}
        </style>""", unsafe_allow_html=True)

    st.title("💄 Desi Beauty Data Analyzer")
    st.caption("Real-time Indian beauty · makeup · skincare trends → Instagram-ready charts")

    with st.sidebar:
        st.header("Controls")
        today = dt.date.today()
        default_start = today - dt.timedelta(days=90)
        dr = st.date_input("Trends date range",
                           value=(default_start, today), max_value=today)
        start_date, end_date = dr if isinstance(dr, tuple) and len(dr) == 2 \
            else (default_start, today)
        st.caption("News is always the latest available (last few days).")
        if st.button("🔄 Refresh real-time data", use_container_width=True):
            st.cache_data.clear()
        if not FEEDPARSER_AVAILABLE:
            st.warning("feedparser not installed — News shows SAMPLE data. "
                       "Add `feedparser` to requirements.txt")
        if not VADER_AVAILABLE:
            st.warning("vaderSentiment not installed — using a rough sentiment "
                       "fallback. Add `vaderSentiment` to requirements.txt")

    start_s, end_s = start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d")

    with st.spinner("Pulling Google Trends (India)…"):
        trends_df, trends_live = get_trends_data(KEYWORD_GROUPS, "IN", start_s, end_s)
    with st.spinner("Pulling current news + scoring sentiment…"):
        articles, news_live = get_news(NEWS_QUERIES)
        articles = score_articles(articles)

    c1, c2 = st.columns(2)
    with c1:
        (st.success if trends_live else st.info)(
            "Google Trends: LIVE ✅" if trends_live
            else "Google Trends: SAMPLE data (throttled) ⚠️")
    with c2:
        (st.success if news_live else st.info)(
            f"News: LIVE ✅ ({len(articles)} articles)" if news_live
            else "News: SAMPLE data ⚠️")

    st.divider()
    t_trends, t_news, t_data = st.tabs(
        ["📈 Google Trends", "📰 News & Sentiment", "🧾 Raw data"])

    with t_trends:
        group = st.selectbox("Keyword group", list(KEYWORD_GROUPS.keys()))
        kws = KEYWORD_GROUPS[group]
        l, r = st.columns(2)
        with l:
            p = chart_interest_over_time(trends_df, group, kws)
            st.image(p, use_container_width=True)
            _dl("⬇️ Download (1080×1080)", p,
                f"trends_{group.lower().replace(' ', '_')}.png")
        with r:
            p = chart_trending_now(trends_df)
            st.image(p, use_container_width=True)
            _dl("⬇️ Download 'Trending Now' (1080×1080)", p, "trending_now.png")

    with t_news:
        pos, neu, neg = overall_mix(articles)
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Articles", len(articles))
        m2.metric("Positive", pos)
        m3.metric("Neutral", neu)
        m4.metric("Negative", neg)

        rows = brand_stats(articles, BRANDS)
        l, r = st.columns(2)
        with l:
            p = chart_news_volume(rows)
            st.image(p, use_container_width=True)
            _dl("⬇️ Download 'In The News' (1080×1080)", p, "news_volume.png")
        with r:
            p = chart_news_sentiment(rows)
            st.image(p, use_container_width=True)
            _dl("⬇️ Download 'How People Feel' (1080×1080)", p, "news_sentiment.png")

        st.subheader("Headlines driving the numbers")
        emoji = {"positive": "🟢", "neutral": "🟣", "negative": "🔴"}
        for a in sorted(articles, key=lambda x: x["score"], reverse=True):
            src = f" · {a['source']}" if a["source"] else ""
            line = f"{emoji[a['label']]} **{a['title']}**{src}"
            if a["link"]:
                line += f"  [↗]({a['link']})"
            st.markdown(line)

    with t_data:
        st.subheader("Google Trends — interest over time")
        st.dataframe(trends_df, use_container_width=True)
        st.download_button("⬇️ Trends CSV", data=trends_df.to_csv().encode("utf-8"),
                           file_name="google_trends_india.csv", mime="text/csv")
        st.subheader("News articles + sentiment")
        ndf = pd.DataFrame([{"title": a["title"], "source": a["source"],
                             "sentiment": round(a["score"], 3),
                             "label": a["label"], "link": a["link"]}
                            for a in articles])
        st.dataframe(ndf, use_container_width=True)
        st.download_button("⬇️ News CSV", data=ndf.to_csv(index=False).encode("utf-8"),
                           file_name="beauty_news_sentiment.csv", mime="text/csv")


if __name__ == "__main__":
    main()
