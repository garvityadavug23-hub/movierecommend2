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
# SESSION STATE
# ─────────────────────────────────────────────
if "watchlist" not in st.session_state:
    st.session_state.watchlist = {}          # title -> rating (1-5)
if "selected_watch" not in st.session_state:
    st.session_state.selected_watch = None
if "recs" not in st.session_state:
    st.session_state.recs = []

# ─────────────────────────────────────────────
# COLLABORATIVE FILTERING CORE
# ─────────────────────────────────────────────
def build_user_taste_vector():
    """
    Weighted average of content vectors for rated movies.
    Rating >= 4  → strong positive signal (weight = rating)
    Rating <= 2  → negative signal (subtracted)
    Rating == 3  → neutral (ignored)
    """
    if not st.session_state.watchlist:
        return None

    pos_vecs, pos_weights = [], []
    neg_vecs = []

    for title, rating in st.session_state.watchlist.items():
        idx_list = movies_raw[movies_raw['title'] == title].index
        if len(idx_list) == 0:
            continue
        idx = idx_list[0]
        vec = vectors[idx]
        if rating >= 4:
            pos_vecs.append(vec * rating)
            pos_weights.append(rating)
        elif rating <= 2:
            neg_vecs.append(vec)

    if not pos_vecs:
        return None

    taste = np.sum(pos_vecs, axis=0) / sum(pos_weights)

    # Subtract negative taste signal
    if neg_vecs:
        neg_avg = np.mean(neg_vecs, axis=0)
        taste = taste - 0.5 * neg_avg

    return taste


def recommend(base_movie, mood_filter, genre_filter, era_filter, text_filter):
    idx_list = movies_raw[movies_raw['title'] == base_movie].index
    if len(idx_list) == 0:
        return []

    idx = idx_list[0]
    base_distances = similarity[idx].copy()
    user_taste = build_user_taste_vector()

    scored = []

    for i in range(len(movies_raw)):
        row = movies_raw.iloc[i]
        score = base_distances[i]

        # ── User taste (collaborative signal) ──────────────────────
        if user_taste is not None:
            taste_sim = cosine_similarity([user_taste], [vectors[i]])[0][0]
            score += 0.6 * taste_sim           # heavier weight than before

        # ── Mood boost ──────────────────────────────────────────────
        if mood_filter != "Any":
            mood_genres = MOOD_GENRE_MAP.get(mood_filter, [])
            if any(g in row['genre_list'] for g in mood_genres):
                score += 0.25

        # ── Genre boost ─────────────────────────────────────────────
        if genre_filter != "Any":
            if genre_filter in row['genre_list']:
                score += 0.25

        # ── Era boost ───────────────────────────────────────────────
        if era_filter != "Any":
            try:
                year = int(str(row['release_date'])[:4])
                start, end = ERA_YEAR_MAP[era_filter]
                if start <= year <= end:
                    score += 0.1
            except:
                pass

        # ── Keyword match ───────────────────────────────────────────
        if text_filter.strip():
            keywords = text_filter.lower().split()
            overview = str(row['overview']).lower()
            matches = sum(1 for kw in keywords if kw in overview)
            score += 0.08 * matches

        # ── Penalise already-rated movies ───────────────────────────
        if row['title'] in st.session_state.watchlist:
            score -= 999

        scored.append((i, score))

    scored.sort(key=lambda x: x[1], reverse=True)

    results = []
    for i, s in scored:
        title = movies_raw.iloc[i]['title']
        if title != base_movie:
            results.append((i, s))
        if len(results) >= 8:
            break

    return results

# ─────────────────────────────────────────────
# TMDB API HELPERS
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

# ─────────────────────────────────────────────
# UI
# ─────────────────────────────────────────────
st.set_page_config(layout="wide", page_title="CineAI PRO")
st.title("🎬 CineAI PRO")
st.caption("Rate movies you've seen → get smarter recommendations over time")

# ── Filters ─────────────────────────────────
col1, col2 = st.columns(2)
with col1:
    mood  = st.selectbox("🎭 Mood",  ["Any", "Happy", "Sad", "Thrilling", "Romantic"])
    genre = st.selectbox("🎬 Genre", ["Any", "Action", "Comedy", "Drama", "Horror",
                                       "Sci-Fi", "Romance", "Thriller", "Animation"])
with col2:
    era  = st.selectbox("📅 Era", ["Any", "2000s", "2010s", "2020s"])
    text = st.text_input("🔍 Describe what you want (keywords)")

selected_movie = st.selectbox("🎥 Base Movie (starting point)", movies_raw['title'].values)

if st.button("✨ Get AI Recommendations", use_container_width=True):
    with st.spinner("🧠 Analysing your taste..."):
        st.session_state.recs = recommend(selected_movie, mood, genre, era, text)

# ── Recommendation Results ───────────────────
if st.session_state.recs:
    rated_count = len(st.session_state.watchlist)
    if rated_count == 0:
        st.info("💡 **Tip:** Rate movies in your watchlist to get personalised recommendations!")
    else:
        st.success(f"🧠 Personalised using {rated_count} rated movie(s) from your watchlist")

    st.subheader("🤖 AI Picks For You")
    cols = st.columns(4)

    for i, rec in enumerate(st.session_state.recs):
        movie = movies_raw.iloc[rec[0]]
        imdb  = round(movie['vote_average'], 1)
        stars = "⭐" * int(imdb // 2)

        with cols[i % 4]:
            st.image(get_poster(movie['id']))

            with st.expander(f"{movie['title']}  {stars} ({imdb}/10)"):
                st.caption("Genres: " + ", ".join(movie['genre_list']))
                st.caption("Year:   " + str(movie['release_date'])[:4])
                st.write(movie['overview'][:200] + "...")

                # AI reason
                reasons = []
                if genre != "Any" and genre in movie['genre_list']:
                    reasons.append(f"matches **{genre}** genre")
                if mood != "Any":
                    mg = MOOD_GENRE_MAP.get(mood, [])
                    if any(g in movie['genre_list'] for g in mg):
                        reasons.append(f"fits **{mood}** mood")
                if build_user_taste_vector() is not None:
                    reasons.append("aligned with your taste profile")
                if reasons:
                    st.caption("🤖 Because: " + ", ".join(reasons))

                # Trailer
                trailer = get_trailer(movie['id'])
                if trailer:
                    st.video(trailer)
                else:
                    st.caption("Trailer not available")

                # Add to watchlist with inline rating
                if movie['title'] not in st.session_state.watchlist:
                    rating = st.select_slider(
                        "Your rating",
                        options=[1, 2, 3, 4, 5],
                        value=3,
                        key=f"rate_{i}",
                        help="1=Hated it  3=It's ok  5=Loved it"
                    )
                    if st.button("➕ Add to Watchlist", key=f"add_{i}"):
                        st.session_state.watchlist[movie['title']] = rating
                        st.success(f"Added '{movie['title']}' with {rating}⭐ rating!")
                        st.rerun()
                else:
                    current = st.session_state.watchlist[movie['title']]
                    st.caption(f"✅ In watchlist — rated {current}⭐")

# ─────────────────────────────────────────────
# SIDEBAR — WATCHLIST
# ─────────────────────────────────────────────
st.sidebar.title("📌 My Watchlist")

if not st.session_state.watchlist:
    st.sidebar.caption("Your watchlist is empty.\nAdd & rate movies to personalise recommendations.")
else:
    st.sidebar.caption(f"{len(st.session_state.watchlist)} movie(s) rated")

    # Taste summary
    ratings = list(st.session_state.watchlist.values())
    avg_r   = round(sum(ratings) / len(ratings), 1)
    st.sidebar.metric("Avg. Rating", f"{avg_r} ⭐")

    liked    = [t for t, r in st.session_state.watchlist.items() if r >= 4]
    disliked = [t for t, r in st.session_state.watchlist.items() if r <= 2]
    if liked:
        st.sidebar.success("❤️ Loved: " + ", ".join(liked[:3]))
    if disliked:
        st.sidebar.error("👎 Disliked: " + ", ".join(disliked[:3]))

    st.sidebar.markdown("---")

    for i, (title, rating) in enumerate(list(st.session_state.watchlist.items())):
        c1, c2, c3 = st.sidebar.columns([3, 1, 1])

        if c1.button(title[:22] + ("…" if len(title) > 22 else ""), key=f"view_{i}"):
            st.session_state.selected_watch = title
            st.rerun()

        # Inline re-rating
        new_rating = c2.selectbox(
            "", [1, 2, 3, 4, 5],
            index=rating - 1,
            key=f"rerate_{i}",
            label_visibility="collapsed"
        )
        if new_rating != rating:
            st.session_state.watchlist[title] = new_rating
            st.rerun()

        if c3.button("❌", key=f"remove_{i}"):
            del st.session_state.watchlist[title]
            st.rerun()

# ─────────────────────────────────────────────
# SELECTED MOVIE DETAIL
# ─────────────────────────────────────────────
if st.session_state.selected_watch:
    match = movies_raw[movies_raw['title'] == st.session_state.selected_watch]

    if not match.empty:
        movie = match.iloc[0]

        st.markdown("---")
        st.markdown("## 🎬 Movie Detail")

        c1, c2 = st.columns([1, 2])

        with c1:
            st.image(get_poster(movie['id']))

        with c2:
            st.subheader(movie['title'])
            st.write(f"**IMDb Score:** {movie['vote_average']} / 10")
            st.write(f"**Genres:** {', '.join(movie['genre_list'])}")
            st.write(f"**Year:** {str(movie['release_date'])[:4]}")
            st.write(movie['overview'])

            # Rating widget in detail view
            current_rating = st.session_state.watchlist.get(movie['title'], 3)
            new_r = st.select_slider(
                "Rate this movie",
                options=[1, 2, 3, 4, 5],
                value=current_rating,
                key="detail_rating"
            )
            if st.button("💾 Save Rating"):
                st.session_state.watchlist[movie['title']] = new_r
                st.success(f"Saved {new_r}⭐ for '{movie['title']}'")
                st.rerun()

            trailer = get_trailer(movie['id'])
            if trailer:
                st.video(trailer)
