import streamlit as st
import pandas as pd
import requests
import os
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------- CONFIG ----------
TMDB_API_KEY = os.getenv("TMDB_API_KEY")
IMAGE_BASE = "https://image.tmdb.org/t/p/w500"

# ---------- LOAD DATA ----------
@st.cache_data
def load_data():
    movies = pd.read_csv("tmdb_5000_movies.csv")
    movies = movies[['id','title','overview','genres','vote_average']].dropna()
    return movies

movies = load_data()

# ---------- MODEL ----------
@st.cache_resource
def create_similarity(data):
    data['tags'] = data['overview'] + data['genres']
    cv = CountVectorizer(max_features=5000, stop_words='english')
    vectors = cv.fit_transform(data['tags']).toarray()
    return cosine_similarity(vectors)

similarity = create_similarity(movies)

# ---------- SESSION ----------
if "watchlist" not in st.session_state:
    st.session_state.watchlist = []

if "selected_watch" not in st.session_state:
    st.session_state.selected_watch = None

# ---------- UI ----------
st.set_page_config(layout="wide")
st.title("🎬 CineAI FINAL PRO")

# ---------- INPUTS ----------
col1, col2 = st.columns(2)

with col1:
    mood = st.selectbox("Mood", ["Happy","Sad","Thrilling","Romantic"])
    genre = st.selectbox("Genre", ["Action","Comedy","Drama","Horror","Sci-Fi"])

with col2:
    era = st.selectbox("Era", ["2000s","2010s","2020s"])
    text = st.text_input("Describe what you want")

selected_movie = st.selectbox("Base Movie", movies['title'].values)

# ---------- POSTER ----------
def get_poster(movie_id):
    if not TMDB_API_KEY:
        return "https://via.placeholder.com/300x450?text=No+API+Key"

    try:
        url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={TMDB_API_KEY}"
        data = requests.get(url).json()
        if data.get("poster_path"):
            return f"{IMAGE_BASE}{data['poster_path']}"
    except:
        pass

    return "https://via.placeholder.com/300x450?text=No+Image"

# ---------- TRAILER ----------
def get_trailer(movie_id):
    if not TMDB_API_KEY:
        return None

    try:
        url = f"https://api.themoviedb.org/3/movie/{movie_id}/videos?api_key={TMDB_API_KEY}"
        data = requests.get(url).json()

        for vid in data.get("results", []):
            if vid["type"] == "Trailer" and vid["site"] == "YouTube":
                return f"https://www.youtube.com/watch?v={vid['key']}"
    except:
        pass

    return None

# ---------- RECOMMEND ----------
def recommend(movie):
    idx = movies[movies['title'] == movie].index[0]
    distances = list(enumerate(similarity[idx]))
    return sorted(distances, key=lambda x: x[1], reverse=True)[1:9]

# ---------- BUTTON ----------
if st.button("🎥 Get Recommendations"):

    with st.spinner("🎞️ Loading cinematic magic..."):
        recs = recommend(selected_movie)

    # ===== GRAPH =====
    movie_names = []
    sim_scores = []

    for rec in recs:
        movie_idx = rec[0]
        sim_score = rec[1]

        movie_names.append(movies.iloc[movie_idx]['title'][:20])
        sim_scores.append(sim_score)

    sim_df = pd.DataFrame({
        "Movie": movie_names,
        "Similarity": sim_scores
    }).sort_values(by="Similarity", ascending=False)

    st.subheader(f"📊 Recommendation Strength for '{selected_movie}'")
    st.bar_chart(sim_df.set_index("Movie"))

    # ===== MOVIES =====
    cols = st.columns(4)

    for i, rec in enumerate(recs):
        movie = movies.iloc[rec[0]]

        score = round(movie['vote_average'], 1)
        stars = "⭐" * int(score // 2)

        with cols[i % 4]:

            st.image(get_poster(movie['id']))

            with st.expander(f"{movie['title']} {stars} ({score}/10)"):

                st.write("IMDb:", movie['vote_average'])
                st.write(movie['overview'][:200])

                trailer = get_trailer(movie['id'])
                if trailer:
                    st.video(trailer)

                if st.button("➕ Add to Watchlist", key=f"a{i}"):
                    if movie['title'] not in st.session_state.watchlist:
                        st.session_state.watchlist.append(movie['title'])

# ---------- WATCHLIST ----------
st.sidebar.title("📌 Watchlist")

for i, title in enumerate(st.session_state.watchlist):

    col1, col2 = st.sidebar.columns([3,1])

    if col1.button(title, key=f"view_{i}"):
        st.session_state.selected_watch = title
        st.rerun()

    if col2.button("❌", key=f"remove_{i}"):
        st.session_state.watchlist.remove(title)
        st.rerun()

# ---------- DETAILS ----------
if st.session_state.selected_watch:

    movie = movies[movies['title'] == st.session_state.selected_watch].iloc[0]

    st.markdown("## 🎬 Selected Movie")

    col1, col2 = st.columns([1,2])

    with col1:
        st.image(get_poster(movie['id']))

    with col2:
        st.write("Title:", movie['title'])
        st.write("IMDb:", movie['vote_average'])
        st.write(movie['overview'])

        trailer = get_trailer(movie['id'])
        if trailer:
            st.video(trailer)
