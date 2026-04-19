import streamlit as st
import pandas as pd
import requests
import os
import ast
import numpy as np
import streamlit.components.v1 as components
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
TMDB_API_KEY = os.getenv("63c39d2a4b6cbbe2c300411d8980ade1", "")
IMAGE_BASE   = "https://image.tmdb.org/t/p/w500"

st.set_page_config(layout="wide", page_title="CineAI", page_icon="🎬")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=DM+Sans:wght@300;400;600&display=swap');
*, *::before, *::after { box-sizing: border-box; }
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background: #0a0a0f;
    color: #e8e8f0;
}
h1 { font-family:'Bebas Neue',sans-serif; font-size:2.8rem; letter-spacing:5px; color:#ff2d55; margin-bottom:0; }
h2,h3 { font-family:'Bebas Neue',sans-serif; letter-spacing:2px; color:#ff2d55; }
.stButton>button {
    background:#ff2d55; color:#fff; border:none; border-radius:6px;
    font-weight:600; font-size:0.85rem; padding:6px 14px;
    transition:all .15s;
}
.stButton>button:hover { background:#cc1f3f; transform:scale(1.04); }
[data-testid="stSidebar"] { background:#0f0f18 !important; border-right:1px solid #1e1e2e; }
.stTabs [data-baseweb="tab-list"] { background:#0f0f18; border-bottom:1px solid #1e1e2e; gap:4px; }
.stTabs [data-baseweb="tab"] { color:#888; font-weight:600; }
.stTabs [aria-selected="true"] { color:#ff2d55 !important; border-bottom:2px solid #ff2d55; }
.stSelectbox>div>div, .stTextInput>div>div>input {
    background:#1a1a2e !important; color:#e8e8f0 !important;
    border:1px solid #2a2a3e !important; border-radius:8px;
}
div[data-testid="stExpander"] { background:#13131f; border:1px solid #1e1e2e; border-radius:10px; }
.tag {
    display:inline-block; background:#1e1e2e; color:#aaa;
    padding:2px 10px; border-radius:20px; font-size:0.72rem; margin:2px;
}
.lbl { color:#666; font-size:0.72rem; text-transform:uppercase; letter-spacing:1px; }
.val { color:#fff; font-size:0.95rem; font-weight:600; }
.badge { background:#ff2d55; color:#fff; padding:2px 8px; border-radius:10px; font-size:0.75rem; font-weight:700; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────
def _init(key, val):
    if key not in st.session_state:
        st.session_state[key] = val

_init("watchlist",      {})    # {title: {rating, movie_idx}}
_init("poster_cache",   {})    # cache key -> url string
_init("detail_movie",   None)  # title string when detail panel open
_init("sim_results",    None)  # (base_title, [(idx, score), ...])
_init("filter_results", None)  # DataFrame

# ─────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────
@st.cache_data
def load_data():
    movies = pd.read_csv("tmdb_5000_movies.csv")
    keep   = ['id','title','overview','genres','vote_average',
              'release_date','runtime','budget','revenue','vote_count','popularity']
    movies = movies[[c for c in keep if c in movies.columns]].dropna(subset=['overview'])

    def parse_genres(g):
        try:    return [x['name'] for x in ast.literal_eval(g)]
        except: return []

    movies['genre_list'] = movies['genres'].apply(parse_genres)
    movies['genre_str']  = movies['genre_list'].apply(lambda x: ' '.join(x))
    movies['year']       = pd.to_datetime(movies.get('release_date', ''), errors='coerce').dt.year

    try:
        credits = pd.read_csv("tmdb_5000_credits.csv")
        id_col  = 'movie_id' if 'movie_id' in credits.columns else 'id'
        credits = credits.rename(columns={id_col: 'id'})[['id','cast','crew']]
        movies  = movies.merge(credits, on='id', how='left')

        def p_cast(c):
            try:    return [x['name'] for x in ast.literal_eval(c)[:5]]
            except: return []

        def p_dir(c):
            try:
                for x in ast.literal_eval(c):
                    if x.get('job') == 'Director': return x['name']
            except: pass
            return "N/A"

        movies['cast_list'] = movies['cast'].apply(p_cast) if 'cast' in movies.columns else [[] for _ in range(len(movies))]
        movies['director']  = movies['crew'].apply(p_dir)  if 'crew' in movies.columns else ['N/A'] * len(movies)
    except Exception:
        movies['cast_list'] = [[] for _ in range(len(movies))]
        movies['director']  = 'N/A'

    return movies.reset_index(drop=True)

movies = load_data()

@st.cache_resource
def build_similarity(_df):
    df       = _df.copy()
    df['tags'] = df['overview'].fillna('') + ' ' + df['genre_str'].fillna('')
    cv       = CountVectorizer(max_features=5000, stop_words='english')
    vecs     = cv.fit_transform(df['tags']).toarray()
    return cosine_similarity(vecs)

similarity = build_similarity(movies)

# ─────────────────────────────────────────────
# MOOD → GENRE KEYWORDS
# ─────────────────────────────────────────────
MOOD_GENRES = {
    "Happy":     ["Comedy","Animation","Family","Music"],
    "Sad":       ["Drama","Romance"],
    "Thrilling": ["Action","Thriller","Crime","Mystery"],
    "Romantic":  ["Romance","Drama"],
    "Scary":     ["Horror","Mystery","Thriller"],
    "Adventure": ["Adventure","Fantasy","Science Fiction"],
}
ERA_RANGES = {
    "Any":   (1900, 2030),
    "80s":   (1980, 1989),
    "90s":   (1990, 1999),
    "2000s": (2000, 2009),
    "2010s": (2010, 2019),
    "2020s": (2020, 2030),
}

# ─────────────────────────────────────────────
# TMDB HELPERS — REAL POSTERS
# ─────────────────────────────────────────────
def get_poster(movie_row) -> str:
    title   = str(movie_row['title'])
    tmdb_id = int(movie_row['id'])
    ck      = f"p_{tmdb_id}"

    if ck in st.session_state.poster_cache:
        return st.session_state.poster_cache[ck]

    fallback = f"https://placehold.co/300x450/13131f/ff2d55?text={requests.utils.quote(title[:18])}"

    if not TMDB_API_KEY:
        st.session_state.poster_cache[ck] = fallback
        return fallback

    # Try direct TMDB id
    try:
        r = requests.get(
            f"https://api.themoviedb.org/3/movie/{tmdb_id}",
            params={"api_key": TMDB_API_KEY}, timeout=6
        )
        path = r.json().get("poster_path")
        if path:
            url = IMAGE_BASE + path
            st.session_state.poster_cache[ck] = url
            return url
    except Exception:
        pass

    # Fallback: search by title
    try:
        year   = int(movie_row['year']) if pd.notna(movie_row.get('year')) else None
        params = {"api_key": TMDB_API_KEY, "query": title}
        if year:
            params["year"] = year
        r      = requests.get("https://api.themoviedb.org/3/search/movie",
                               params=params, timeout=6)
        res    = r.json().get("results", [])
        if res and res[0].get("poster_path"):
            url = IMAGE_BASE + res[0]["poster_path"]
            st.session_state.poster_cache[ck] = url
            return url
    except Exception:
        pass

    st.session_state.poster_cache[ck] = fallback
    return fallback


def get_trailer_key(tmdb_id) -> str:
    ck = f"tr_{tmdb_id}"
    if ck in st.session_state.poster_cache:
        return st.session_state.poster_cache[ck]
    if not TMDB_API_KEY:
        st.session_state.poster_cache[ck] = ""
        return ""
    try:
        r = requests.get(
            f"https://api.themoviedb.org/3/movie/{int(tmdb_id)}/videos",
            params={"api_key": TMDB_API_KEY}, timeout=6
        )
        for v in r.json().get("results", []):
            if v["type"] == "Trailer" and v["site"] == "YouTube":
                st.session_state.poster_cache[ck] = v["key"]
                return v["key"]
    except Exception:
        pass
    st.session_state.poster_cache[ck] = ""
    return ""


def yt_embed(key: str):
    if not key:
        return
    components.html(
        f'<iframe width="100%" height="300"'
        f' src="https://www.youtube.com/embed/{key}"'
        f' frameborder="0" allow="autoplay; encrypted-media" allowfullscreen></iframe>',
        height=310
    )

# ─────────────────────────────────────────────
# RECOMMENDATION ENGINES
# ─────────────────────────────────────────────
def recommend_similar(title, n=8):
    idx = movies[movies['title'] == title].index
    if idx.empty:
        return []
    row = list(enumerate(similarity[idx[0]]))
    return sorted(row, key=lambda x: x[1], reverse=True)[1:n+1]


def recommend_by_filters(mood, genre, era, text_q, n=12):
    mask = pd.Series([True] * len(movies), index=movies.index)

    y0, y1 = ERA_RANGES.get(era, (1900, 2030))
    mask &= movies['year'].between(y0, y1)

    if genre != "Any":
        mask &= movies['genre_list'].apply(lambda gl: genre in gl)

    if mood != "Any":
        mood_gs = MOOD_GENRES.get(mood, [])
        mask   &= movies['genre_list'].apply(lambda gl: any(g in gl for g in mood_gs))

    filtered = movies[mask].copy()
    if filtered.empty:
        return filtered

    if text_q.strip():
        try:
            tfidf  = TfidfVectorizer(stop_words='english')
            corpus = filtered['overview'].fillna('').tolist()
            mat    = tfidf.fit_transform(corpus)
            qvec   = tfidf.transform([text_q])
            sims   = cosine_similarity(qvec, mat).flatten()
            filtered = filtered.copy()
            filtered['_score'] = sims
            filtered = filtered.sort_values('_score', ascending=False)
        except Exception:
            filtered = filtered.sort_values('vote_average', ascending=False)
    else:
        filtered = filtered.sort_values('vote_average', ascending=False)

    return filtered.head(n)


def recommend_from_ratings(n=12):
    if not st.session_state.watchlist:
        return []
    weighted = np.zeros(len(movies))
    seen     = set()
    for title, info in st.session_state.watchlist.items():
        idx    = info.get("movie_idx")
        rating = info.get("rating")
        if idx is None or rating is None:
            continue
        seen.add(idx)
        weighted += (rating - 3) * similarity[idx]
    for i in seen:
        weighted[i] = -9999
    top = np.argsort(weighted)[::-1][:n]
    return [(int(i), float(weighted[i])) for i in top if weighted[i] > -9999]

# ─────────────────────────────────────────────
# REUSABLE MOVIE CARD
# ─────────────────────────────────────────────
def movie_card(movie_row, card_key, show_add=True, match_pct=None):
    title  = movie_row['title']
    score  = round(movie_row['vote_average'], 1)
    year   = int(movie_row['year']) if pd.notna(movie_row.get('year')) else ""
    genres = ", ".join(movie_row.get('genre_list', [])[:2])
    poster = get_poster(movie_row)

    st.image(poster, use_container_width=True)
    st.markdown(f"**{title}**")
    st.caption(f"⭐ {score}/10 · {year} · {genres}")
    if match_pct is not None:
        st.progress(match_pct, text=f"Match {match_pct}%")

    b1, b2 = st.columns(2)
    with b1:
        if st.button("Details", key=f"det_{card_key}"):
            st.session_state.detail_movie = title
            st.rerun()
    with b2:
        if show_add:
            in_wl = title in st.session_state.watchlist
            if not in_wl:
                if st.button("+ List", key=f"add_{card_key}"):
                    idx_list = movies[movies['title'] == title].index
                    st.session_state.watchlist[title] = {
                        "rating":    None,
                        "movie_idx": int(idx_list[0]) if not idx_list.empty else None
                    }
                    st.rerun()
            else:
                st.markdown('<span class="badge">✓ Listed</span>', unsafe_allow_html=True)

# ─────────────────────────────────────────────
# DETAIL PAGE  (intercepts the full app render)
# ─────────────────────────────────────────────
def show_detail_panel(title):
    rows = movies[movies['title'] == title]
    if rows.empty:
        st.error("Movie not found.")
        return

    m           = rows.iloc[0]
    poster_url  = get_poster(m)
    trailer_key = get_trailer_key(m['id'])
    year        = int(m['year'])  if pd.notna(m.get('year'))    else "N/A"
    runtime     = f"{int(m['runtime'])} min" if pd.notna(m.get('runtime')) and m.get('runtime', 0) > 0 else "N/A"
    score       = round(m['vote_average'], 1)
    stars       = "⭐" * int(score // 2)

    if st.button("← Back to Browse", key="detail_back"):
        st.session_state.detail_movie = None
        st.rerun()

    st.markdown("---")
    left, right = st.columns([1, 2], gap="large")

    with left:
        st.image(poster_url, use_container_width=True)
        st.markdown("")
        # Watchlist button inside detail
        in_wl = title in st.session_state.watchlist
        if not in_wl:
            if st.button("➕ Add to Watchlist", key="det_add_wl"):
                st.session_state.watchlist[title] = {
                    "rating":    None,
                    "movie_idx": int(rows.index[0])
                }
                st.rerun()
        else:
            st.success("✅ In your watchlist")
            curr  = st.session_state.watchlist[title].get("rating") or 3
            new_r = st.select_slider(
                "Your rating",
                options=[1, 2, 3, 4, 5],
                value=curr,
                format_func=lambda x: ["😞 1","😐 2","🙂 3","😊 4","🤩 5"][x-1],
                key="det_rate_slider"
            )
            if st.button("Save rating", key="det_save_rate"):
                st.session_state.watchlist[title]["rating"] = new_r
                st.toast("Rating saved!")

    with right:
        st.markdown(f"## {m['title']}  ({year})")
        st.markdown(f"**{stars}** &nbsp; `{score}/10` &nbsp; · &nbsp; `{int(m.get('vote_count', 0)):,} votes`",
                    unsafe_allow_html=True)

        tags_html = " ".join(f'<span class="tag">{g}</span>' for g in m.get('genre_list', []))
        st.markdown(tags_html, unsafe_allow_html=True)
        st.markdown("")

        det = {
            "Director":   m.get('director', 'N/A'),
            "Runtime":    runtime,
            "Year":       year,
            "Budget":     f"${int(m['budget']):,}"  if pd.notna(m.get('budget'))  and m.get('budget',  0) > 0 else "N/A",
            "Box Office": f"${int(m['revenue']):,}" if pd.notna(m.get('revenue')) and m.get('revenue', 0) > 0 else "N/A",
        }
        g1, g2 = st.columns(2)
        for i2, (lbl, val) in enumerate(det.items()):
            (g1 if i2 % 2 == 0 else g2).markdown(
                f'<div class="lbl">{lbl}</div><div class="val">{val}</div><br>',
                unsafe_allow_html=True
            )

        cast = m.get('cast_list', [])
        if cast:
            st.markdown(
                f'<div class="lbl">Cast</div>'
                f'<div class="val">{" &nbsp;·&nbsp; ".join(cast)}</div><br>',
                unsafe_allow_html=True
            )

        st.markdown("### Overview")
        st.write(m['overview'])

    if trailer_key:
        st.markdown("### 🎬 Trailer")
        yt_embed(trailer_key)
    elif TMDB_API_KEY:
        st.caption("No trailer available for this movie.")
    else:
        st.warning("Set the `TMDB_API_KEY` environment variable to load real posters and trailers.")


# ═══════════════════════════════════════════════════════
# TOP-LEVEL DETAIL INTERCEPT
# When detail_movie is set, render ONLY the detail page.
# st.stop() prevents any tab/grid code from running.
# ═══════════════════════════════════════════════════════
st.title("🎬 CINEAI")
st.caption("AI-powered movie discovery")

if st.session_state.detail_movie:
    show_detail_panel(st.session_state.detail_movie)
    st.stop()

# ═══════════════════════════════════════════════════════
# MAIN TABS  (only reached when no detail is open)
# ═══════════════════════════════════════════════════════
tabs = st.tabs(["🔍 Discover", "🎭 Mood & Filter", "📌 Watchlist", "🤖 AI Picks"])

# ──────────────────────────────────────────────────────
# TAB 0 — DISCOVER (cosine similarity)
# ──────────────────────────────────────────────────────
with tabs[0]:
    st.subheader("Find movies similar to one you love")

    sel    = st.selectbox("Pick a base movie", movies['title'].values, key="base_sel")
    go_btn = st.button("🎥 Get Recommendations", key="go_sim")

    if go_btn:
        st.session_state.sim_results = (sel, recommend_similar(sel))

    if st.session_state.sim_results is not None:
        base_title, recs = st.session_state.sim_results
        if not recs:
            st.warning("No results found.")
        else:
            sim_df = pd.DataFrame({
                "Movie": [movies.iloc[r[0]]['title'][:22] for r in recs],
                "Score": [r[1] for r in recs]
            }).sort_values("Score", ascending=False)
            st.markdown(f"#### 📊 Similarity scores for *{base_title}*")
            st.bar_chart(sim_df.set_index("Movie"))
            st.markdown("---")
            cols = st.columns(4)
            for i, (midx, _) in enumerate(recs):
                with cols[i % 4]:
                    movie_card(movies.iloc[midx], f"sim_{midx}")

# ──────────────────────────────────────────────────────
# TAB 1 — MOOD & FILTER
# ──────────────────────────────────────────────────────
with tabs[1]:
    st.subheader("🎭 Filter by Mood, Genre & Era")

    c1, c2, c3 = st.columns(3)
    with c1:
        mood = st.selectbox("🌟 Mood", ["Any"] + list(MOOD_GENRES.keys()), key="f_mood")
    with c2:
        genre_opts = ["Any","Action","Adventure","Animation","Comedy","Crime",
                      "Documentary","Drama","Fantasy","History","Horror",
                      "Music","Mystery","Romance","Science Fiction",
                      "Thriller","War","Western"]
        genre = st.selectbox("🎬 Genre", genre_opts, key="f_genre")
    with c3:
        era = st.selectbox("📅 Era", list(ERA_RANGES.keys()), key="f_era")

    text_q    = st.text_input("🔎 Describe what you want  (optional)",
                               placeholder="e.g. space exploration, strong female lead, heartwarming",
                               key="f_text")
    filt_btn  = st.button("🎯 Find Movies", key="go_filter")

    if filt_btn:
        st.session_state.filter_results = recommend_by_filters(mood, genre, era, text_q)

    if st.session_state.filter_results is not None:
        res = st.session_state.filter_results
        if len(res) == 0:
            st.warning("No movies match these filters — try broadening your search.")
            st.session_state.filter_results = None
        else:
            st.markdown(f"#### Found **{len(res)}** movies")
            cols = st.columns(4)
            for i, (_, row) in enumerate(res.iterrows()):
                with cols[i % 4]:
                    movie_card(row, f"flt_{row['id']}")

# ──────────────────────────────────────────────────────
# TAB 2 — WATCHLIST
# ──────────────────────────────────────────────────────
with tabs[2]:
    st.subheader("📌 My Watchlist")

    if not st.session_state.watchlist:
        st.info("Your watchlist is empty — add movies from the Discover or Mood & Filter tabs.")
    else:
        st.caption("Rate movies you've watched to power your AI Picks.")
        st.markdown("---")

        titles_snap = list(st.session_state.watchlist.keys())  # snapshot to avoid mutation issues

        for title in titles_snap:
            if title not in st.session_state.watchlist:
                continue

            info      = st.session_state.watchlist[title]
            midx      = info.get("movie_idx")
            movie_row = movies.iloc[midx] if midx is not None and midx < len(movies) else None

            ca, cb, cc, cd = st.columns([1, 3, 2, 1])

            with ca:
                if movie_row is not None:
                    st.image(get_poster(movie_row), width=75)

            with cb:
                yr = str(int(movie_row['year'])) if movie_row is not None and pd.notna(movie_row.get('year')) else ""
                st.markdown(f"**{title}** {f'({yr})' if yr else ''}")
                if movie_row is not None:
                    g = ", ".join(movie_row.get('genre_list', [])[:3])
                    s = round(movie_row['vote_average'], 1)
                    st.caption(f"⭐ {s}/10 · {g}")

            with cc:
                curr  = info.get("rating") or 3
                new_r = st.select_slider(
                    "Rating",
                    options=[1, 2, 3, 4, 5],
                    value=curr,
                    format_func=lambda x: ["😞 1","😐 2","🙂 3","😊 4","🤩 5"][x-1],
                    key=f"wl_rate_{title}"
                )
                if st.button("Save", key=f"wl_save_{title}"):
                    st.session_state.watchlist[title]["rating"] = new_r
                    st.toast(f"Saved {new_r}★ for {title}!")

            with cd:
                if st.button("🔍", key=f"wl_det_{title}", help="Details"):
                    st.session_state.detail_movie = title
                    st.rerun()
                if st.button("🗑", key=f"wl_del_{title}", help="Remove"):
                    del st.session_state.watchlist[title]
                    st.rerun()

            st.markdown("---")

# ──────────────────────────────────────────────────────
# TAB 3 — AI PICKS (rating-based personalised recs)
# ──────────────────────────────────────────────────────
with tabs[3]:
    st.subheader("🤖 Personalised AI Picks")

    rated = {t: i for t, i in st.session_state.watchlist.items()
             if i.get("rating") is not None}

    if not rated:
        st.info("Rate at least one movie in your Watchlist to unlock personalised picks.")
        st.markdown("""
**How it works**
- 🤩 **4–5 stars** → finds movies with similar vibes and ranks them higher
- 😞 **1–2 stars** → pushes similar movies down so you don't see more like it
- 🙂 **3 stars**   → neutral — no effect either way
        """)
    else:
        st.markdown(f"Based on **{len(rated)} rated movie(s)** in your watchlist:")
        for t, i in rated.items():
            r  = i['rating']
            em = ["😞","😐","🙂","😊","🤩"][r - 1]
            st.markdown(f"- {em} **{t}** — {r}/5")

        st.markdown("---")
        ai_recs = recommend_from_ratings(n=12)

        if not ai_recs:
            st.warning("Not enough signal yet — try rating a few more movies.")
        else:
            st.markdown("#### 🎬 Movies You'll Probably Love")
            cols = st.columns(4)
            for i, (midx, score) in enumerate(ai_recs):
                pct = min(100, max(0, int((score + 2) / 4 * 100)))
                with cols[i % 4]:
                    movie_card(movies.iloc[midx], f"ai_{midx}", match_pct=pct)
