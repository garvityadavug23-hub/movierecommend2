import streamlit as st
import pandas as pd
import requests
import ast
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

@st.cache_data
def load_data():
    return pd.read_csv("tmdb_5000_movies.csv")

movies_raw = load_data()
movies_raw = movies_raw[['id', 'title', 'overview', 'genres', 'vote_average', 'release_date']].dropna()

def convert(obj):
    L = []
    try:
        for i in ast.literal_eval(obj):
            L.append(i['name'])
    except:
        pass
    return L

movies_raw['genre_list'] = movies_raw['genres'].apply(convert)
movies_raw['genres_str'] = movies_raw['genre_list'].apply(lambda x: " ".join(x))
movies_raw['tags'] = movies_raw['overview'] + " " + movies_raw['genres_str']

cv = CountVectorizer(max_features=5000, stop_words='english')
vectors = cv.fit_transform(movies_raw['tags']).toarray()
similarity = cosine_similarity(vectors)

# ---------- SESSION ----------
if "watchlist" not in st.session_state:
    st.session_state.watchlist = []
if "selected_watch" not in st.session_state:
    st.session_state.selected_watch = None

# ---------- AI CORE ----------
def build_user_profile(watchlist):
    if not watchlist:
        return None
    indices = movies_raw[movies_raw['title'].isin(watchlist)].index
    if len(indices) == 0:
        return None
    return vectors[indices].mean(axis=0)

def recommend(movie, mood_filter, genre_filter, era_filter, text_filter):
    idx_list = movies_raw[movies_raw['title'] == movie].index
    if len(idx_list) == 0:
        return []

    idx = idx_list[0]
    base_distances = similarity[idx]
    user_vector = build_user_profile(st.session_state.watchlist)

    scored = []

    for i in range(len(movies_raw)):
        row = movies_raw.iloc[i]
        score = base_distances[i]

        # 🧠 User profile boost (collaborative feel)
        if user_vector is not None:
            user_sim = cosine_similarity([user_vector], [vectors[i]])[0][0]
            score += 0.4 * user_sim

        # 🎭 Mood
        if mood_filter != "Any":
            mood_genres = MOOD_GENRE_MAP.get(mood_filter, [])
            if any(g in row['genre_list'] for g in mood_genres):
                score += 0.2

        # 🎬 Genre
        if genre_filter != "Any":
            if genre_filter in row['genre_list']:
                score += 0.2

        # 📅 Era
        if era_filter != "Any":
            try:
                year = int(str(row['release_date'])[:4])
                start, end = ERA_YEAR_MAP[era_filter]
                if start <= year <= end:
                    score += 0.1
            except:
                pass

        # 🔍 Text
        if text_filter.strip():
            keywords = text_filter.lower().split()
            overview = str(row['overview']).lower()
            matches = sum(1 for kw in keywords if kw in overview)
            score += 0.05 * matches

        scored.append((i, score))

    scored = sorted(scored, key=lambda x: x[1], reverse=True)

    results = []
    for i, s in scored:
        title = movies_raw.iloc[i]['title']
        if title != movie and title not in st.session_state.watchlist:
            results.append((i, s))
        if len(results) >= 8:
            break

    return results

# ---------- API ----------
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

# ---------- UI ----------
st.set_page_config(layout="wide")
st.title("🎬 CineAI PRO")

col1, col2 = st.columns(2)

with col1:
    mood = st.selectbox("Mood", ["Any", "Happy", "Sad", "Thrilling", "Romantic"])
    genre = st.selectbox("Genre", ["Any", "Action", "Comedy", "Drama", "Horror", "Sci-Fi", "Romance", "Thriller", "Animation"])

with col2:
    era = st.selectbox("Era", ["Any", "2000s", "2010s", "2020s"])
    text = st.text_input("Describe what you want")

selected_movie = st.selectbox("Base Movie", movies_raw['title'].values)

# ---------- RECOMMEND ----------
if st.button("🎥 Get AI Recommendations"):
    with st.spinner("🧠 AI is analyzing your taste..."):
        recs = recommend(selected_movie, mood, genre, era, text)

    if not recs:
        st.warning("No recommendations found.")
    else:
        st.subheader("🤖 AI Personalized Picks For You")

        cols = st.columns(4)

        for i, rec in enumerate(recs):
            movie = movies_raw.iloc[rec[0]]
            imdb = round(movie['vote_average'], 1)
            stars = "⭐" * int(imdb // 2)

            with cols[i % 4]:
                st.image(get_poster(movie['id']))

                with st.expander(f"{movie['title']} {stars} ({imdb}/10)"):

                    st.caption("Genres: " + ", ".join(movie['genre_list']))
                    st.caption("Year: " + str(movie['release_date'])[:4])
                    st.write(movie['overview'][:200])

                    # 🤖 AI Explanation
                    reason = []

                    if genre != "Any" and genre in movie['genre_list']:
                        reason.append("matches your genre")

                    if mood != "Any":
                        mood_genres = MOOD_GENRE_MAP.get(mood, [])
                        if any(g in movie['genre_list'] for g in mood_genres):
                            reason.append("fits your mood")

                    if st.session_state.watchlist:
                        reason.append("based on your watchlist")

                    if reason:
                        st.caption("🤖 AI says: " + ", ".join(reason))

                    trailer = get_trailer(movie['id'])
                    if trailer:
                        st.video(trailer)
                    else:
                        st.write("Trailer not available")

                    if st.button("➕ Add to Watchlist", key=f"a{i}"):
                        if movie['title'] not in st.session_state.watchlist:
                            st.session_state.watchlist.append(movie['title'])
                            st.success(f"Added {movie['title']}!")

# ---------- WATCHLIST ----------
st.sidebar.title("📌 Watchlist")

if not st.session_state.watchlist:
    st.sidebar.caption("Your watchlist is empty.")

for i, title in enumerate(st.session_state.watchlist):
    c1, c2 = st.sidebar.columns([3, 1])

    if c1.button(title, key=f"view_{i}"):
        st.session_state.selected_watch = title
        st.rerun()

    if c2.button("❌", key=f"remove_{i}"):
        st.session_state.watchlist.remove(title)
        st.rerun()

# ---------- SELECTED MOVIE ----------
if st.session_state.selected_watch:
    match = movies_raw[movies_raw['title'] == st.session_state.selected_watch]

    if not match.empty:
        movie = match.iloc[0]

        st.markdown("---")
        st.markdown("## 🎬 Selected Movie")

        c1, c2 = st.columns([1, 2])

        with c1:
            st.image(get_poster(movie['id']))

        with c2:
            st.write("**Title:**", movie['title'])
            st.write("**IMDb:**", movie['vote_average'])
            st.write("**Genres:**", ", ".join(movie['genre_list']))
            st.write("**Year:**", str(movie['release_date'])[:4])
            st.write(movie['overview'])

            trailer = get_trailer(movie['id'])
            if trailer:
                st.video(trailer)




