"""
============================================================================
  DESI BEAUTY DATA ANALYZER
  A local, real-time trend dashboard for an Indian beauty / makeup /
  skincare Instagram page.
----------------------------------------------------------------------------
  Data:   Google Trends (pytrends, GEO=IN)  +  Indian beauty subreddits
  Output: Pastel, Instagram-ready 1080x1080 charts you can download & post
  UI:     A local Streamlit dashboard in your browser
============================================================================

QUICK START (run these in your terminal, one time):

    python -m venv venv
    # Mac/Linux:
    source venv/bin/activate
    # Windows:
    venv\\Scripts\\activate

    pip install streamlit pandas matplotlib pytrends requests

Then, from the folder containing this file, run:

    streamlit run desi_beauty_data_analyzer.py

Your browser opens automatically at http://localhost:8501

NOTES
- Reddit works with NO credentials (uses the public JSON feed). If Reddit
  ever throttles you, add free API keys in the sidebar (optional).
- Google Trends has aggressive rate limits. This app retries with backoff
  and, if the live call still fails, falls back to clearly-labelled SAMPLE
  data so your dashboard is never blank while prepping content.
============================================================================
"""

import io
import re
import time
import random
import datetime as dt
from collections import Counter

import requests
import pandas as pd

import matplotlib
matplotlib.use("Agg")            # headless backend, safe inside Streamlit
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm

import streamlit as st

# pytrends is optional at import-time so a missing install can't crash the app
try:
    from pytrends.request import TrendReq
    PYTRENDS_AVAILABLE = True
except Exception:
    PYTRENDS_AVAILABLE = False


# ---------------------------------------------------------------------------
# 1. CONFIG — edit these lists to change what the dashboard tracks
# ---------------------------------------------------------------------------

KEYWORD_GROUPS = {
    "Lip Products":       ["kajal", "tinted lip balm", "lip tint", "matte lipstick"],
    "Skincare":           ["clear sunscreen", "kojic acid", "saffron serum"],
    "Cultural Catalysts": ["glass skin", "wedding makeup"],
}

DEFAULT_SUBREDDITS = ["IndianMakeupAddicts", "IndianSkincareAddicts", "IndianBeautyDeals"]

# Brands to count in Reddit chatter (add/remove freely)
BRANDS = [
    "Nykaa", "Tira", "Sugar", "Minimalist", "L'Oreal", "Maybelline", "Lakme",
    "Plum", "Dot & Key", "The Ordinary", "Cetaphil", "Deconstruct", "Foxtale",
    "Renee", "Mamaearth", "Sunscoop", "Aqualogica", "Cosrx", "Innisfree",
]

# Product-type buzzwords to count in Reddit chatter
PRODUCT_TYPES = [
    "sunscreen", "serum", "moisturizer", "lip balm", "lipstick", "kajal",
    "foundation", "concealer", "toner", "cleanser", "retinol", "niacinamide",
    "vitamin c", "spf", "glass skin", "kojic acid",
]


# ---------------------------------------------------------------------------
# 2. AESTHETIC — pastel, chic, minimalist styling
# ---------------------------------------------------------------------------

PALETTE = {
    "bg":    "#FFFDF9",   # warm cream canvas
    "panel": "#FCE9EC",   # pale blush plot area
    "grid":  "#EAE3DA",   # whisper-thin gridlines
    "ink":   "#5A5150",   # soft charcoal for text
    "muted": "#A79B97",   # muted secondary text
}

# Trendy pastel series colours — no default bright blues/reds
SERIES_COLORS = [
    "#E19AAE",  # dusty rose
    "#A9C3A0",  # sage
    "#B7A6D6",  # lavender
    "#F0B79A",  # peach
    "#9FC0D4",  # powder blue
    "#CBA0C4",  # mauve
    "#E7C98B",  # butter
    "#8FB8AE",  # eucalyptus
]

SQUARE_IN = 10.8   # inches; at dpi=100 -> exactly 1080 px
SQUARE_DPI = 100


def _pick_font():
    """Prefer a clean modern font if the machine has one; else DejaVu Sans."""
    preferred = ["Poppins", "Montserrat", "Nunito Sans", "Quicksand",
                 "Segoe UI", "Helvetica Neue", "Arial"]
    installed = {f.name for f in fm.fontManager.ttflist}
    for name in preferred:
        if name in installed:
            return name
    return "DejaVu Sans"


def set_aesthetic_style():
    """Apply the global pastel look to matplotlib."""
    font = _pick_font()
    plt.rcParams.update({
        "figure.facecolor":  PALETTE["bg"],
        "axes.facecolor":    PALETTE["bg"],
        "savefig.facecolor": PALETTE["bg"],
        "font.family":       font,
        "text.color":        PALETTE["ink"],
        "axes.edgecolor":    PALETTE["grid"],
        "axes.labelcolor":   PALETTE["ink"],
        "axes.titlecolor":   PALETTE["ink"],
        "xtick.color":       PALETTE["muted"],
        "ytick.color":       PALETTE["muted"],
        "axes.linewidth":    0.8,
        "grid.color":        PALETTE["grid"],
        "grid.linewidth":    0.6,
        "figure.dpi":        SQUARE_DPI,
    })


def _new_square_fig():
    """A locked 1:1 figure. We intentionally avoid bbox_inches='tight' so the
    saved PNG is exactly 1080x1080."""
    fig, ax = plt.subplots(figsize=(SQUARE_IN, SQUARE_IN), dpi=SQUARE_DPI)
    fig.subplots_adjust(left=0.12, right=0.94, top=0.84, bottom=0.18)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.grid(True, axis="both", alpha=0.6)
    ax.set_axisbelow(True)
    return fig, ax


def _title(fig, ax, title, subtitle):
    fig.text(0.12, 0.93, title, fontsize=26, fontweight="bold",
             color=PALETTE["ink"], ha="left")
    fig.text(0.12, 0.885, subtitle, fontsize=13, color=PALETTE["muted"], ha="left")
    fig.text(0.94, 0.05, "@ your.beauty.page", fontsize=11,
             color=PALETTE["muted"], ha="right", style="italic")


def _fig_to_png_bytes(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=SQUARE_DPI, facecolor=PALETTE["bg"])
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# 3. DATA COLLECTION — Google Trends (with retry + graceful fallback)
# ---------------------------------------------------------------------------

def _with_retry(fn, tries=3, base_delay=2.0):
    """Run fn() with exponential backoff. Raises the last error if all fail."""
    last = None
    for i in range(tries):
        try:
            return fn()
        except Exception as e:
            last = e
            time.sleep(base_delay * (2 ** i) + random.random())
    raise last


def _fallback_trends(groups, start, end):
    """Deterministic, plausible sample interest-over-time so the UI stays alive
    when Google Trends throttles. Clearly flagged as sample in the UI."""
    idx = pd.date_range(start=start, end=end, freq="D")
    data = {}
    for _, kws in groups.items():
        for kw in kws:
            seed = sum(ord(c) for c in kw)
            rng = random.Random(seed)
            level = rng.randint(25, 70)
            series = []
            for _ in idx:
                level = max(3, min(100, level + rng.randint(-8, 9)))
                series.append(level)
            data[kw] = series
    df = pd.DataFrame(data, index=idx)
    df.index.name = "date"
    return df


@st.cache_data(ttl=3600, show_spinner=False)
def get_trends_data(groups, geo, start, end):
    """Fetch interest-over-time for every keyword, one group at a time
    (Google caps payloads at 5 keywords). Returns (DataFrame, is_live)."""
    if not PYTRENDS_AVAILABLE:
        return _fallback_trends(groups, start, end), False

    timeframe = f"{start} {end}"
    frames = []
    try:
        pytrends = TrendReq(hl="en-IN", tz=330, timeout=(10, 25))
        for _, kws in groups.items():
            def _fetch(kws=kws):
                pytrends.build_payload(kws, cat=0, timeframe=timeframe, geo=geo)
                return pytrends.interest_over_time()

            df = _with_retry(_fetch)
            if df is None or df.empty:
                continue
            df = df.drop(columns=[c for c in ("isPartial",) if c in df.columns])
            frames.append(df)
            time.sleep(1.5)   # be polite between payloads to dodge 429s

        if not frames:
            return _fallback_trends(groups, start, end), False

        merged = pd.concat(frames, axis=1)
        merged = merged.loc[:, ~merged.columns.duplicated()]
        merged.index.name = "date"
        return merged, True

    except Exception:
        return _fallback_trends(groups, start, end), False


# ---------------------------------------------------------------------------
# 4. DATA COLLECTION — Reddit (public JSON, no keys needed; PRAW optional)
# ---------------------------------------------------------------------------

_FALLBACK_TITLES = [
    "Nykaa vs Tira — where are you actually shopping now?",
    "Minimalist niacinamide serum finally back in stock",
    "Glass skin routine that survived a Delhi summer",
    "Sugar matte lipstick shade recommendations for medium skin",
    "Is the Aqualogica sunscreen worth the hype? SPF review",
    "Dot & Key lip balm dupe for chapped winter lips",
    "The Ordinary retinol vs Minimalist retinol — HG pick",
    "Wedding makeup trial: Lakme vs L'Oreal foundation",
    "Foxtale vitamin c serum results after 4 weeks",
    "Kojic acid soap for pigmentation — did it work for you?",
    "Cetaphil cleanser holy grail for sensitive skin",
    "Renee kajal that actually does not smudge in humidity",
    "Plum green tea toner honest review",
    "Best clear sunscreen no white cast for Indian skin",
    "Mamaearth vs Deconstruct — budget skincare showdown",
]


@st.cache_data(ttl=1800, show_spinner=False)
def get_reddit_titles(subreddits, limit=40, client_id="", client_secret=""):
    """Pull recent hot post titles. Returns (titles, is_live).
    Path 1: PRAW if credentials given. Path 2: public JSON feed. Path 3: sample."""
    # --- Path 1: authenticated PRAW ---
    if client_id and client_secret:
        try:
            import praw
            reddit = praw.Reddit(
                client_id=client_id,
                client_secret=client_secret,
                user_agent="desi-beauty-data-analyzer/1.0",
            )
            titles = []
            for sub in subreddits:
                for post in reddit.subreddit(sub).hot(limit=limit):
                    titles.append(post.title)
            if titles:
                return titles, True
        except Exception:
            pass  # fall through to public JSON

    # --- Path 2: public JSON feed (no auth) ---
    titles = []
    headers = {"User-Agent": "desi-beauty-data-analyzer/1.0 (personal use)"}
    for sub in subreddits:
        url = f"https://www.reddit.com/r/{sub}/hot.json?limit={limit}"
        try:
            def _fetch(url=url):
                r = requests.get(url, headers=headers, timeout=15)
                r.raise_for_status()
                return r.json()

            data = _with_retry(_fetch, tries=2, base_delay=2.0)
            for child in data.get("data", {}).get("children", []):
                t = child.get("data", {}).get("title")
                if t:
                    titles.append(t)
            time.sleep(1.0)
        except Exception:
            continue

    if titles:
        return titles, True

    # --- Path 3: sample ---
    return list(_FALLBACK_TITLES), False


# ---------------------------------------------------------------------------
# 5. PROCESSING — keyword / brand counters
# ---------------------------------------------------------------------------

def count_mentions(titles, terms):
    """Case-insensitive mention count of each term across all titles."""
    blob = " ".join(titles).lower()
    counts = Counter()
    for term in terms:
        t = term.lower()
        if re.search(r"[^a-z0-9]", t):       # multi-word / punctuated -> substring
            n = blob.count(t)
        else:                                # single token -> word boundary
            n = len(re.findall(rf"\b{re.escape(t)}\b", blob))
        if n:
            counts[term] = n
    return counts


# ---------------------------------------------------------------------------
# 6. CHARTS — all 1080x1080, all on-brand
# ---------------------------------------------------------------------------

def chart_interest_over_time(df, group_name, keywords):
    set_aesthetic_style()
    fig, ax = _new_square_fig()
    for i, kw in enumerate(keywords):
        if kw in df.columns:
            ax.plot(df.index, df[kw], label=kw, linewidth=2.6,
                    color=SERIES_COLORS[i % len(SERIES_COLORS)],
                    solid_capstyle="round")
    ax.set_ylabel("Search interest", fontsize=12)
    ax.set_ylim(0, 105)
    ax.tick_params(labelsize=10)
    fig.autofmt_xdate(rotation=25)
    leg = ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14),
                    ncol=min(len(keywords), 3), frameon=False, fontsize=11)
    for txt in leg.get_texts():
        txt.set_color(PALETTE["ink"])
    _title(fig, ax, group_name, "Google Trends · India · search interest over time")
    return _fig_to_png_bytes(fig)


def chart_trending_now(df, top_n=10):
    """Latest-day interest, ranked — a clean 'what's hot right now' bar."""
    set_aesthetic_style()
    latest = df.iloc[-1].sort_values(ascending=True).tail(top_n)
    fig, ax = _new_square_fig()
    ax.grid(True, axis="x", alpha=0.6)
    ax.grid(False, axis="y")
    colors = [SERIES_COLORS[i % len(SERIES_COLORS)] for i in range(len(latest))]
    bars = ax.barh(latest.index, latest.values, color=colors, height=0.62)
    for bar, val in zip(bars, latest.values):
        ax.text(bar.get_width() + 1.5, bar.get_y() + bar.get_height() / 2,
                f"{int(val)}", va="center", fontsize=11, color=PALETTE["muted"])
    ax.set_xlim(0, 108)
    ax.set_xlabel("Search interest (latest)", fontsize=12)
    ax.tick_params(labelsize=11)
    _title(fig, ax, "Trending Right Now",
           "Google Trends · India · latest search interest")
    return _fig_to_png_bytes(fig)


def chart_mentions(counts, title, subtitle, top_n=10):
    set_aesthetic_style()
    fig, ax = _new_square_fig()
    ax.grid(True, axis="x", alpha=0.6)
    ax.grid(False, axis="y")
    if not counts:
        ax.text(0.5, 0.5, "No mentions found today", transform=ax.transAxes,
                ha="center", va="center", fontsize=16, color=PALETTE["muted"])
        _title(fig, ax, title, subtitle)
        return _fig_to_png_bytes(fig)

    items = counts.most_common(top_n)[::-1]
    labels = [k for k, _ in items]
    values = [v for _, v in items]
    colors = [SERIES_COLORS[i % len(SERIES_COLORS)] for i in range(len(items))]
    bars = ax.barh(labels, values, color=colors, height=0.62)
    for bar, val in zip(bars, values):
        ax.text(bar.get_width() + max(values) * 0.02,
                bar.get_y() + bar.get_height() / 2,
                str(val), va="center", fontsize=11, color=PALETTE["muted"])
    ax.set_xlabel("Mentions", fontsize=12)
    ax.tick_params(labelsize=11)
    ax.margins(x=0.12)
    _title(fig, ax, title, subtitle)
    return _fig_to_png_bytes(fig)


# ---------------------------------------------------------------------------
# 7. STREAMLIT DASHBOARD
# ---------------------------------------------------------------------------

def _download(label, png, filename):
    st.download_button(label, data=png, file_name=filename,
                       mime="image/png", use_container_width=True)


def main():
    st.set_page_config(page_title="Desi Beauty Data Analyzer",
                       page_icon="💄", layout="wide")

    # Light pastel skin for the Streamlit chrome itself
    st.markdown(
        """
        <style>
          .stApp { background-color: #FFFDF9; }
          h1, h2, h3, h4 { color: #5A5150; }
          section[data-testid="stSidebar"] { background-color: #FCE9EC; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.title("💄 Desi Beauty Data Analyzer")
    st.caption("Real-time Indian beauty · makeup · skincare trends → Instagram-ready charts")

    # ---- Sidebar controls ----
    with st.sidebar:
        st.header("Controls")

        today = dt.date.today()
        default_start = today - dt.timedelta(days=90)
        date_range = st.date_input(
            "Date range",
            value=(default_start, today),
            max_value=today,
        )
        if isinstance(date_range, tuple) and len(date_range) == 2:
            start_date, end_date = date_range
        else:
            start_date, end_date = default_start, today

        subs = st.multiselect("Subreddits", DEFAULT_SUBREDDITS,
                              default=DEFAULT_SUBREDDITS)

        with st.expander("Reddit API keys (optional)"):
            st.caption("Leave blank to use the public feed. Keys help if you "
                       "hit rate limits — create them at reddit.com/prefs/apps")
            r_id = st.text_input("client_id", type="password")
            r_secret = st.text_input("client_secret", type="password")

        refresh = st.button("🔄 Refresh real-time data", use_container_width=True)
        if refresh:
            st.cache_data.clear()

        if not PYTRENDS_AVAILABLE:
            st.warning("pytrends not installed — Trends will show SAMPLE data. "
                       "Run: pip install pytrends")

    start_s = start_date.strftime("%Y-%m-%d")
    end_s = end_date.strftime("%Y-%m-%d")

    # ---- Fetch ----
    with st.spinner("Pulling Google Trends (India)…"):
        trends_df, trends_live = get_trends_data(KEYWORD_GROUPS, "IN", start_s, end_s)
    with st.spinner("Pulling Reddit chatter…"):
        titles, reddit_live = get_reddit_titles(
            subs or DEFAULT_SUBREDDITS, limit=40,
            client_id=r_id, client_secret=r_secret,
        )

    # ---- Status banners ----
    c1, c2 = st.columns(2)
    with c1:
        (st.success if trends_live else st.info)(
            "Google Trends: LIVE ✅" if trends_live
            else "Google Trends: SAMPLE data (live fetch throttled) ⚠️")
    with c2:
        (st.success if reddit_live else st.info)(
            f"Reddit: LIVE ✅ ({len(titles)} posts)" if reddit_live
            else "Reddit: SAMPLE data (live fetch failed) ⚠️")

    st.divider()

    tab_trends, tab_reddit, tab_data = st.tabs(
        ["📈 Google Trends", "🗣️ Reddit Buzz", "🧾 Raw data"])

    # ---- Trends tab ----
    with tab_trends:
        group = st.selectbox("Keyword group", list(KEYWORD_GROUPS.keys()))
        kws = KEYWORD_GROUPS[group]

        left, right = st.columns(2)
        with left:
            png_line = chart_interest_over_time(trends_df, group, kws)
            st.image(png_line, use_container_width=True)
            _download("⬇️ Download this chart (1080×1080)",
                      png_line, f"trends_{group.lower().replace(' ', '_')}.png")
        with right:
            png_now = chart_trending_now(trends_df, top_n=10)
            st.image(png_now, use_container_width=True)
            _download("⬇️ Download 'Trending Now' (1080×1080)",
                      png_now, "trending_now.png")

    # ---- Reddit tab ----
    with tab_reddit:
        brand_counts = count_mentions(titles, BRANDS)
        type_counts = count_mentions(titles, PRODUCT_TYPES)

        left, right = st.columns(2)
        with left:
            png_brands = chart_mentions(
                brand_counts, "Most-Talked-About Brands",
                f"Indian beauty subreddits · {len(titles)} recent posts")
            st.image(png_brands, use_container_width=True)
            _download("⬇️ Download brands chart (1080×1080)",
                      png_brands, "reddit_brands.png")
        with right:
            png_types = chart_mentions(
                type_counts, "Trending Product Types",
                f"Indian beauty subreddits · {len(titles)} recent posts")
            st.image(png_types, use_container_width=True)
            _download("⬇️ Download product-types chart (1080×1080)",
                      png_types, "reddit_product_types.png")

        with st.expander("See the post titles being analysed"):
            for t in titles:
                st.write("•", t)

    # ---- Raw data tab ----
    with tab_data:
        st.subheader("Google Trends — interest over time")
        st.dataframe(trends_df, use_container_width=True)
        st.download_button(
            "⬇️ Download trends as CSV",
            data=trends_df.to_csv().encode("utf-8"),
            file_name="google_trends_india.csv",
            mime="text/csv",
        )


if __name__ == "__main__":
    main()
