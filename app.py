import streamlit as st
import pandas as pd
import requests
import ast
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

TMDB_API_KEY = "63c39d2a4b6cbbe2c300411d8980ade1"
IMAGE_BASE = "https://image.tmdb.org/t/p/w500"

MOOD_GENRE_MAP = {
    "Happy":     ["Comedy", "Animation", "Family", "Music"],
    "Sad":       ["Drama", "Romance"],
    "Thrilling": ["Thriller", "Action", "Crime", "Mystery"],
    "Romantic":  ["Romance", "Drama"],
    "Scared":    ["Horror", "Mystery", "Thriller"],
    "Inspired":  ["Biography", "History", "Documentary", "Drama"],
    "Adventurous": ["Adventure", "Action", "Fantasy", "Science Fiction"],
}

ERA_YEAR_MAP = {
    "Before 1950s": (1900, 1949),
    "1950s":        (1950, 1959),
    "1960s":        (1960, 1969),
    "1970s":        (1970, 1979),
    "1980s":        (1980, 1989),
    "1990s":        (1990, 1999),
    "2000s":        (2000, 2009),
    "2010s":        (2010, 2019),
    "2020s":        (2020, 2030),
}

# ─────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────
if "watchlist" not in st.session_state:
    st.session_state.watchlist = {}
elif isinstance(st.session_state.watchlist, list):
    st.session_state.watchlist = {t: 50 for t in st.session_state.watchlist}

if "selected_watch" not in st.session_state:
    st.session_state.selected_watch = None
if "recs" not in st.session_state:
    st.session_state.recs = []

# ─────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("tmdb_5000_movies.csv")
    df = df[['id', 'title', 'overview', 'genres', 'vote_average',
             'release_date', 'vote_count']].dropna()

    def convert(obj):
        L = []
        try:
            for item in ast.literal_eval(obj):
                L.append(item['name'])
        except:
            pass
        return L

    df['genre_list'] = df['genres'].apply(convert)
    df['genres_str'] = df['genre_list'].apply(lambda x: " ".join(x))
    df['tags']       = df['overview'] + " " + df['genres_str']
    df = df.reset_index(drop=True)
    return df

@st.cache_data
def build_vectors(tags_series):
    cv   = CountVectorizer(max_features=5000, stop_words='english')
    vecs = cv.fit_transform(tags_series).toarray()
    sim  = cosine_similarity(vecs)
    return vecs, sim

movies_raw       = load_data()
vectors, similarity = build_vectors(movies_raw['tags'])

# ─────────────────────────────────────────────
# SCORE → WEIGHT
# ─────────────────────────────────────────────
def score_to_weight(score):
    norm = (score - 50) / 50.0
    return norm * 2.0 if abs(norm) >= 0.4 else norm

def build_user_taste_vector():
    if not st.session_state.watchlist:
        return None, None
    pos_sum = np.zeros(vectors.shape[1])
    neg_sum = np.zeros(vectors.shape[1])
    pos_w = neg_w = 0.0
    for title, score in st.session_state.watchlist.items():
        idx_l = movies_raw[movies_raw['title'] == title].index
        if len(idx_l) == 0:
            continue
        vec = vectors[idx_l[0]]
        w   = score_to_weight(score)
        if w > 0:
            pos_sum += vec * w; pos_w += w
        elif w < 0:
            neg_sum += vec * abs(w); neg_w += abs(w)
    return (pos_sum / pos_w if pos_w > 0 else None,
            neg_sum / neg_w if neg_w > 0 else None)

# ─────────────────────────────────────────────
# RECOMMEND  (base_movie is now optional)
# ─────────────────────────────────────────────
def recommend(base_movie, mood_filter, genre_filter, era_filter, text_filter, min_rating):
    pos_vec, neg_vec = build_user_taste_vector()

    # If no base movie selected, start from a uniform score of 0
    if base_movie and base_movie != "— No base movie —":
        idx_l = movies_raw[movies_raw['title'] == base_movie].index
        base_sim = similarity[idx_l[0]].copy() if len(idx_l) > 0 else np.zeros(len(movies_raw))
    else:
        base_sim = np.zeros(len(movies_raw))

    scored = []
    for i in range(len(movies_raw)):
        row = movies_raw.iloc[i]
        s   = base_sim[i]

        if pos_vec is not None:
            s += 0.7 * cosine_similarity([pos_vec], [vectors[i]])[0][0]
        if neg_vec is not None:
            s -= 0.5 * cosine_similarity([neg_vec], [vectors[i]])[0][0]

        if mood_filter != "Any":
            if any(g in row['genre_list'] for g in MOOD_GENRE_MAP.get(mood_filter, [])):
                s += 0.3
        if genre_filter != "Any" and genre_filter in row['genre_list']:
            s += 0.3
        if era_filter != "Any":
            try:
                year = int(str(row['release_date'])[:4])
                lo, hi = ERA_YEAR_MAP[era_filter]
                if lo <= year <= hi:
                    s += 0.15
            except:
                pass
        if text_filter.strip():
            ov = str(row['overview']).lower()
            s += 0.08 * sum(1 for kw in text_filter.lower().split() if kw in ov)

        # Minimum IMDb filter
        if float(row['vote_average']) < min_rating:
            s = -9999

        if row['title'] in st.session_state.watchlist:
            s = -9999
        if base_movie and row['title'] == base_movie:
            s = -9999

        scored.append((i, s))

    scored.sort(key=lambda x: x[1], reverse=True)
    results = []
    for i, s in scored:
        if s > -9000:
            results.append((i, s))
        if len(results) >= 8:
            break
    return results

# ─────────────────────────────────────────────
# TMDB API
# ─────────────────────────────────────────────
@st.cache_data(ttl=86400)
def get_movie_details(movie_id):
    try:
        url  = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={TMDB_API_KEY}&append_to_response=credits"
        return requests.get(url, timeout=6).json()
    except:
        return {}

def get_poster(details):
    p = details.get("poster_path")
    return f"{IMAGE_BASE}{p}" if p else "https://via.placeholder.com/300x450?text=No+Image"

def get_trailer(movie_id):
    try:
        url  = f"https://api.themoviedb.org/3/movie/{movie_id}/videos?api_key={TMDB_API_KEY}"
        data = requests.get(url, timeout=5).json()
        for vid in data.get("results", []):
            if vid["type"] == "Trailer" and vid["site"] == "YouTube":
                return f"https://www.youtube.com/watch?v={vid['key']}"
    except:
        pass
    return None

def score_label(score):
    if score >= 85:   return "❤️ Masterpiece"
    elif score >= 70: return "👍 Really liked it"
    elif score >= 55: return "🙂 Pretty decent"
    elif score == 50: return "😐 Neutral"
    elif score >= 35: return "😕 Didn't enjoy it"
    elif score >= 20: return "👎 Disliked it"
    else:             return "🤮 Hated it"

def stars(rating):
    full  = int(rating // 2)
    return "★" * full + "☆" * (5 - full)

# ─────────────────────────────────────────────
# RICH MOVIE DETAIL CARD
# ─────────────────────────────────────────────
def render_detail_card(m_title):
    match = movies_raw[movies_raw['title'] == m_title]
    if match.empty:
        return
    movie    = match.iloc[0]
    details  = get_movie_details(int(movie['id']))
    trailer  = get_trailer(int(movie['id']))

    poster_url = get_poster(details)
    imdb       = float(movie['vote_average'])
    year       = str(movie['release_date'])[:4]
    genres_str = ", ".join(movie['genre_list']) or "N/A"
    overview   = str(movie['overview'])

    # Runtime
    runtime_min = details.get("runtime")
    runtime_str = f"{runtime_min // 60}h {runtime_min % 60}m" if runtime_min else "N/A"

    # Tagline
    tagline = details.get("tagline", "")

    # Cast — top 6
    cast_list = []
    credits   = details.get("credits", {})
    for c in credits.get("cast", [])[:6]:
        cast_list.append(c.get("name", ""))
    cast_str = " · ".join(cast_list) if cast_list else "N/A"

    # Director
    director = "N/A"
    for c in credits.get("crew", []):
        if c.get("job") == "Director":
            director = c.get("name", "N/A")
            break

    # Budget / Revenue
    budget  = details.get("budget", 0)
    revenue = details.get("revenue", 0)
    budget_str  = f"${budget:,}"  if budget  else "N/A"
    revenue_str = f"${revenue:,}" if revenue else "N/A"

    # Languages
    spoken = details.get("spoken_languages", [])
    lang_str = ", ".join(l.get("english_name", "") for l in spoken[:3]) or "N/A"

    st.markdown("---")
    st.markdown(f"## 🎬 {m_title}")
    if tagline:
        st.markdown(f"*\"{tagline}\"*")

    left, right = st.columns([1, 2])

    with left:
        st.image(poster_url, use_container_width=True)
        if trailer:
            st.video(trailer)
        else:
            st.caption("🎞 No trailer available")

    with right:
        # Score row
        st.markdown(
            f"<h3 style='margin-bottom:4px'>{stars(imdb)}  &nbsp; {imdb}/10</h3>",
            unsafe_allow_html=True
        )
        vc = int(movie.get('vote_count', 0)) if 'vote_count' in movie else 0
        st.caption(f"Based on {vc:,} votes" if vc else "")

        # Info grid
        col_a, col_b = st.columns(2)
        col_a.markdown(f"📅 **Year:** {year}")
        col_b.markdown(f"⏱ **Runtime:** {runtime_str}")
        col_a.markdown(f"🎭 **Genres:** {genres_str}")
        col_b.markdown(f"🌍 **Language:** {lang_str}")
        col_a.markdown(f"🎬 **Director:** {director}")
        col_b.markdown(f"💰 **Budget:** {budget_str}")
        col_a.markdown(f"🏦 **Box Office:** {revenue_str}")

        st.markdown("---")
        st.markdown("### 📖 Synopsis")
        st.markdown(f"> {overview}")

        if cast_str != "N/A":
            st.markdown("---")
            st.markdown("### 🎭 Cast")
            st.markdown(cast_str)

        # Rating widget
        st.markdown("---")
        st.markdown("### ⭐ Your Rating")
        cur_score = st.session_state.watchlist.get(m_title, 50)
        new_score = st.slider("Score (0 = hated · 50 = neutral · 100 = masterpiece)",
                              0, 100, cur_score, key="detail_slider")
        st.caption(score_label(new_score))
        if st.button("💾 Save Rating", key="detail_save"):
            st.session_state.watchlist[m_title] = int(new_score)
            st.success(f"Saved **{new_score}/100** for *{m_title}* ✓")
            st.rerun()

# ─────────────────────────────────────────────
# PAGE UI
# ─────────────────────────────────────────────
st.set_page_config(layout="wide", page_title="CineAI PRO")
st.title("🎬 CineAI PRO")
st.caption("Rate movies 0–100 · the engine learns your taste from every score")

# ── Filters ──────────────────────────────────
col1, col2, col3 = st.columns(3)
with col1:
    mood  = st.selectbox("🎭 Mood",
                         ["Any", "Happy", "Sad", "Thrilling", "Romantic",
                          "Scared", "Inspired", "Adventurous"])
    genre = st.selectbox("🎬 Genre",
                         ["Any", "Action", "Adventure", "Animation", "Biography",
                          "Comedy", "Crime", "Documentary", "Drama", "Fantasy",
                          "History", "Horror", "Music", "Mystery", "Romance",
                          "Sci-Fi", "Science Fiction", "Thriller", "War"])
with col2:
    era = st.selectbox("📅 Era",
                       ["Any"] + list(ERA_YEAR_MAP.keys()))
    min_imdb = st.slider("⭐ Min IMDb Score", 0.0, 10.0, 0.0, 0.5)
with col3:
    text = st.text_input("🔍 Keywords (e.g. 'space war revenge heist')")
    use_base = st.checkbox("Use a base movie as starting point", value=False)

base_movie = None
if use_base:
    base_movie = st.selectbox("🎥 Base Movie", ["— No base movie —"] + list(movies_raw['title'].values))

if st.button("✨ Get Recommendations", use_container_width=True):
    with st.spinner("🧠 Building taste profile and scanning 5,000 movies..."):
        st.session_state.recs = recommend(
            base_movie, mood, genre, era, text, min_imdb
        )
    st.session_state.selected_watch = None

# ── Personalisation banner ───────────────────
if st.session_state.recs:
    rated_count = len(st.session_state.watchlist)
    if rated_count == 0:
        st.info("💡 **Tip:** Rate any movie 0–100 to personalise results. "
                "The engine immediately adjusts every score.")
    else:
        loved    = [t for t, r in st.session_state.watchlist.items() if r >= 70]
        disliked = [t for t, r in st.session_state.watchlist.items() if r <= 30]
        parts    = [f"🧠 Taste profile active — **{rated_count}** movie(s) rated."]
        if loved:    parts.append(f"Pulling toward: _{', '.join(loved[:3])}_.")
        if disliked: parts.append(f"Filtering away from: _{', '.join(disliked[:2])}_.")
        st.success(" ".join(parts))

    st.subheader("🤖 Your Personalised Picks")
    cols = st.columns(4)
    pos_vec, _ = build_user_taste_vector()

    for i, rec in enumerate(st.session_state.recs):
        movie   = movies_raw.iloc[rec[0]]
        imdb    = round(float(movie['vote_average']), 1)
        m_title = str(movie['title'])
        year    = str(movie['release_date'])[:4]
        details = get_movie_details(int(movie['id']))
        poster  = get_poster(details)

        with cols[i % 4]:
            st.image(poster, use_container_width=True)
            with st.expander(f"**{m_title}**  ⭐ {imdb}/10"):

                st.markdown(f"🗓 **{year}**  &nbsp;|&nbsp;  🎭 {', '.join(movie['genre_list'][:3])}")

                tagline = details.get("tagline", "")
                if tagline:
                    st.markdown(f"*\"{tagline}\"*")

                # Full overview — no truncation
                st.markdown("**Synopsis:**")
                st.write(str(movie['overview']))

                # Cast snippet
                cast_top = [c["name"] for c in details.get("credits", {}).get("cast", [])[:4]]
                if cast_top:
                    st.caption("🎭 " + " · ".join(cast_top))

                # AI reason
                reasons = []
                if pos_vec is not None:
                    ts = cosine_similarity([pos_vec], [vectors[rec[0]]])[0][0]
                    if ts > 0.15:
                        reasons.append("matches your taste profile")
                if genre != "Any" and genre in movie['genre_list']:
                    reasons.append(f"{genre} genre match")
                if mood != "Any" and any(g in movie['genre_list']
                                         for g in MOOD_GENRE_MAP.get(mood, [])):
                    reasons.append(f"{mood} mood fit")
                if reasons:
                    st.caption("🤖 **Why:** " + " · ".join(reasons))

                # View full details
                if st.button("🔍 Full Details", key=f"detail_{i}"):
                    st.session_state.selected_watch = m_title
                    st.rerun()

                # Rating
                st.markdown("---")
                if m_title not in st.session_state.watchlist:
                    rv = st.slider("Your score", 0, 100, 50, key=f"rate_{i}")
                    st.caption(score_label(rv))
                    if st.button("➕ Rate & Save", key=f"add_{i}"):
                        st.session_state.watchlist[m_title] = int(rv)
                        st.success(f"Saved {rv}/100 ✓")
                        st.rerun()
                else:
                    cur = st.session_state.watchlist[m_title]
                    st.caption(f"✅ Rated **{cur}/100** — {score_label(cur)}")

# ─────────────────────────────────────────────
# FULL DETAIL PANEL
# ─────────────────────────────────────────────
if st.session_state.selected_watch:
    render_detail_card(st.session_state.selected_watch)

# ─────────────────────────────────────────────
# SIDEBAR — MY RATINGS
# ─────────────────────────────────────────────
st.sidebar.title("📌 My Ratings")

if not st.session_state.watchlist:
    st.sidebar.caption("No ratings yet.\nScore movies to personalise recommendations.")
else:
    vals = list(st.session_state.watchlist.values())
    st.sidebar.metric("Movies Rated", len(vals))
    st.sidebar.metric("Avg Score",    f"{round(sum(vals)/len(vals), 1)}/100")

    loved    = [t for t, r in st.session_state.watchlist.items() if r >= 70]
    disliked = [t for t, r in st.session_state.watchlist.items() if r <= 30]
    if loved:    st.sidebar.success("❤️ " + "  |  ".join(loved[:3]))
    if disliked: st.sidebar.error("👎 "   + "  |  ".join(disliked[:2]))

    st.sidebar.markdown("---")

    for i, (title, rating) in enumerate(list(st.session_state.watchlist.items())):
        st.sidebar.markdown(f"**{title[:26]}{'…' if len(title) > 26 else ''}**")
        c1, c2 = st.sidebar.columns([5, 1])

        new_r = c1.slider("", 0, 100, rating,
                          key=f"rerate_{i}", label_visibility="collapsed")
        c1.caption(score_label(new_r))

        if new_r != rating:
            st.session_state.watchlist[title] = int(new_r)
            st.rerun()

        if c2.button("❌", key=f"rm_{i}"):
            del st.session_state.watchlist[title]
            if st.session_state.selected_watch == title:
                st.session_state.selected_watch = None
            st.rerun()

        if c1.button("📋 View Details", key=f"view_{i}"):
            st.session_state.selected_watch = title
            st.rerun()

        st.sidebar.markdown("---")
