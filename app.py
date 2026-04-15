
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
}

ERA_YEAR_MAP = {
    "2000s": (2000, 2009),
    "2010s": (2010, 2019),
    "2020s": (2020, 2030),
}

# ─────────────────────────────────────────────
# SESSION STATE — initialise first,
# migrate legacy list-based watchlist if needed
# ─────────────────────────────────────────────
if "watchlist" not in st.session_state:
    st.session_state.watchlist = {}          # title -> score (0-100)
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
    df = df[['id', 'title', 'overview', 'genres', 'vote_average', 'release_date']].dropna()

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
    df['tags'] = df['overview'] + " " + df['genres_str']
    df = df.reset_index(drop=True)
    return df

@st.cache_data
def build_vectors(tags_series):
    cv = CountVectorizer(max_features=5000, stop_words='english')
    vecs = cv.fit_transform(tags_series).toarray()
    sim = cosine_similarity(vecs)
    return vecs, sim

movies_raw = load_data()
vectors, similarity = build_vectors(movies_raw['tags'])

# ─────────────────────────────────────────────
# SCORE → WEIGHT MAPPING
# 0-30   strong dislike  → large negative
# 31-49  mild dislike    → small negative
# 50     neutral         → ignored
# 51-69  mild like       → small positive
# 70-100 strong like     → large positive
# ─────────────────────────────────────────────
def score_to_weight(score):
    norm = (score - 50) / 50.0   # maps 0→-1.0, 50→0.0, 100→+1.0
    # Amplify extreme scores
    if abs(norm) >= 0.4:
        return norm * 2.0
    return norm


def build_user_taste_vector():
    if not st.session_state.watchlist:
        return None, None

    pos_sum = np.zeros(vectors.shape[1])
    neg_sum = np.zeros(vectors.shape[1])
    pos_w = 0.0
    neg_w = 0.0

    for title, score in st.session_state.watchlist.items():
        idx_list = movies_raw[movies_raw['title'] == title].index
        if len(idx_list) == 0:
            continue
        vec = vectors[idx_list[0]]
        w = score_to_weight(score)
        if w > 0:
            pos_sum += vec * w
            pos_w += w
        elif w < 0:
            neg_sum += vec * abs(w)
            neg_w += abs(w)

    pos_vec = pos_sum / pos_w if pos_w > 0 else None
    neg_vec = neg_sum / neg_w if neg_w > 0 else None
    return pos_vec, neg_vec


def recommend(base_movie, mood_filter, genre_filter, era_filter, text_filter):
    idx_list = movies_raw[movies_raw['title'] == base_movie].index
    if len(idx_list) == 0:
        return []

    base_distances = similarity[idx_list[0]].copy()
    pos_vec, neg_vec = build_user_taste_vector()
    scored = []

    for i in range(len(movies_raw)):
        row = movies_raw.iloc[i]
        s = base_distances[i]

        if pos_vec is not None:
            s += 0.7 * cosine_similarity([pos_vec], [vectors[i]])[0][0]
        if neg_vec is not None:
            s -= 0.5 * cosine_similarity([neg_vec], [vectors[i]])[0][0]

        if mood_filter != "Any":
            if any(g in row['genre_list'] for g in MOOD_GENRE_MAP.get(mood_filter, [])):
                s += 0.25
        if genre_filter != "Any" and genre_filter in row['genre_list']:
            s += 0.25
        if era_filter != "Any":
            try:
                year = int(str(row['release_date'])[:4])
                lo, hi = ERA_YEAR_MAP[era_filter]
                if lo <= year <= hi:
                    s += 0.1
            except:
                pass
        if text_filter.strip():
            ov = str(row['overview']).lower()
            s += 0.08 * sum(1 for kw in text_filter.lower().split() if kw in ov)

        if row['title'] in st.session_state.watchlist:
            s = -9999

        scored.append((i, s))

    scored.sort(key=lambda x: x[1], reverse=True)
    results = []
    for i, s in scored:
        if movies_raw.iloc[i]['title'] != base_movie:
            results.append((i, s))
        if len(results) >= 8:
            break
    return results

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
def get_poster(movie_id):
    try:
        url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={TMDB_API_KEY}"
        data = requests.get(url, timeout=5).json()
        if data.get("poster_path"):
            return f"{IMAGE_BASE}{data['poster_path']}"
    except:
        pass
    return "https://via.placeholder.com/300x450?text=No+Image"

def get_trailer(movie_id):
    try:
        url = f"https://api.themoviedb.org/3/movie/{movie_id}/videos?api_key={TMDB_API_KEY}"
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

# ─────────────────────────────────────────────
# UI
# ─────────────────────────────────────────────
st.set_page_config(layout="wide", page_title="CineAI PRO")
st.title("🎬 CineAI PRO")
st.caption("Score movies 0–100. The engine learns your taste from every rating.")

col1, col2 = st.columns(2)
with col1:
    mood  = st.selectbox("🎭 Mood",  ["Any", "Happy", "Sad", "Thrilling", "Romantic"])
    genre = st.selectbox("🎬 Genre", ["Any", "Action", "Comedy", "Drama", "Horror",
                                       "Sci-Fi", "Romance", "Thriller", "Animation"])
with col2:
    era  = st.selectbox("📅 Era", ["Any", "2000s", "2010s", "2020s"])
    text = st.text_input("🔍 Keywords (e.g. 'space war revenge')")

selected_movie = st.selectbox("🎥 Base Movie", movies_raw['title'].values)

if st.button("✨ Get Recommendations", use_container_width=True):
    with st.spinner("🧠 Building your taste profile and finding matches..."):
        st.session_state.recs = recommend(selected_movie, mood, genre, era, text)

# ── Results ──────────────────────────────────
if st.session_state.recs:
    rated_count = len(st.session_state.watchlist)
    if rated_count == 0:
        st.info("💡 Rate movies below (0–100) to teach the engine your taste. "
                "Even 3–4 ratings make a noticeable difference.")
    else:
        loved    = [t for t, r in st.session_state.watchlist.items() if r >= 70]
        disliked = [t for t, r in st.session_state.watchlist.items() if r <= 30]
        parts = [f"🧠 Taste profile from **{rated_count}** rating(s)."]
        if loved:    parts.append(f"Pulling toward: _{', '.join(loved[:3])}_.")
        if disliked: parts.append(f"Filtering away: _{', '.join(disliked[:2])}_.")
        st.success(" ".join(parts))

    st.subheader("🤖 Your Personalised Picks")
    cols = st.columns(4)

    for i, rec in enumerate(st.session_state.recs):
        movie   = movies_raw.iloc[rec[0]]
        imdb    = round(movie['vote_average'], 1)
        m_title = str(movie['title'])

        with cols[i % 4]:
            st.image(get_poster(movie['id']))
            with st.expander(f"{m_title}  ⭐{imdb}/10"):
                st.caption("Genres: " + ", ".join(movie['genre_list']))
                st.caption("Year: "   + str(movie['release_date'])[:4])
                st.write(str(movie['overview'])[:200] + "...")

                reasons = []
                pv, _ = build_user_taste_vector()
                if pv is not None:
                    ts = cosine_similarity([pv], [vectors[rec[0]]])[0][0]
                    if ts > 0.15:
                        reasons.append("matches your taste profile")
                if genre != "Any" and genre in movie['genre_list']:
                    reasons.append(f"{genre} genre match")
                if mood != "Any" and any(g in movie['genre_list'] for g in MOOD_GENRE_MAP.get(mood, [])):
                    reasons.append(f"{mood} mood fit")
                if reasons:
                    st.caption("🤖 Why: " + " · ".join(reasons))

                trailer = get_trailer(movie['id'])
                if trailer:
                    st.video(trailer)
                else:
                    st.caption("No trailer available")

                if m_title not in st.session_state.watchlist:
                    rv = st.slider("Your score", 0, 100, 50, key=f"rate_{i}")
                    st.caption(score_label(rv))
                    if st.button("➕ Rate & Save", key=f"add_{i}"):
                        st.session_state.watchlist[m_title] = int(rv)
                        st.success(f"Saved {rv}/100 ✓")
                        st.rerun()
                else:
                    cur = st.session_state.watchlist[m_title]
                    st.caption(f"✅ You rated: **{cur}/100** — {score_label(cur)}")

# ─────────────────────────────────────────────
# SIDEBAR — RATED MOVIES
# ─────────────────────────────────────────────
st.sidebar.title("📌 My Ratings")

if not st.session_state.watchlist:
    st.sidebar.caption("No ratings yet. Score movies to personalise results.")
else:
    vals = list(st.session_state.watchlist.values())
    st.sidebar.metric("Movies Rated", len(vals))
    st.sidebar.metric("Avg Score", f"{round(sum(vals)/len(vals), 1)}/100")

    loved    = [t for t, r in st.session_state.watchlist.items() if r >= 70]
    disliked = [t for t, r in st.session_state.watchlist.items() if r <= 30]
    if loved:    st.sidebar.success("❤️ " + "  |  ".join(loved[:3]))
    if disliked: st.sidebar.error("👎 "   + "  |  ".join(disliked[:2]))

    st.sidebar.markdown("---")
    for i, (title, rating) in enumerate(list(st.session_state.watchlist.items())):
        st.sidebar.markdown(f"**{title[:26]}{'…' if len(title)>26 else ''}**")
        c1, c2 = st.sidebar.columns([5, 1])
        new_r = c1.slider("", 0, 100, rating, key=f"rerate_{i}",
                          label_visibility="collapsed")
        c1.caption(score_label(new_r))
        if new_r != rating:
            st.session_state.watchlist[title] = int(new_r)
            st.rerun()
        if c2.button("❌", key=f"rm_{i}"):
            del st.session_state.watchlist[title]
            st.rerun()
        if c1.button("📋 View Details", key=f"view_{i}"):
            st.session_state.selected_watch = title
            st.rerun()
        st.sidebar.markdown("---")

# ─────────────────────────────────────────────
# MOVIE DETAIL PANEL
# ─────────────────────────────────────────────
if st.session_state.selected_watch:
    match = movies_raw[movies_raw['title'] == st.session_state.selected_watch]
    if not match.empty:
        movie   = match.iloc[0]
        m_title = str(movie['title'])
        st.markdown("---")
        st.markdown("## 🎬 Movie Detail")
        c1, c2 = st.columns([1, 2])
        with c1:
            st.image(get_poster(movie['id']))
        with c2:
            st.subheader(m_title)
            st.write(f"**IMDb:** {movie['vote_average']} / 10")
            st.write(f"**Genres:** {', '.join(movie['genre_list'])}")
            st.write(f"**Year:** {str(movie['release_date'])[:4]}")
            st.write(movie['overview'])
            cur = st.session_state.watchlist.get(m_title, 50)
            new_s = st.slider("Rate (0–100)", 0, 100, cur, key="detail_rating")
            st.caption(score_label(new_s))
            if st.button("💾 Save Rating"):
                st.session_state.watchlist[m_title] = int(new_s)
                st.success(f"Saved {new_s}/100 for '{m_title}'")
                st.rerun()
            trailer = get_trailer(movie['id'])
            if trailer:
                st.video(trailer)
