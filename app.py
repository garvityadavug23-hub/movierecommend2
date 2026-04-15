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

# SESSION
if "watchlist" not in st.session_state:
    st.session_state.watchlist = {}
elif isinstance(st.session_state.watchlist, list):
    st.session_state.watchlist = {t: 50 for t in st.session_state.watchlist}

if "selected_watch" not in st.session_state:
    st.session_state.selected_watch = None
if "recs" not in st.session_state:
    st.session_state.recs = []

# DATA
@st.cache_data
def load_data():
    df = pd.read_csv("tmdb_5000_movies.csv")
    df = df[['id', 'title', 'overview', 'genres', 'vote_average', 'release_date']].dropna()

    def convert(obj):
        try:
            return [i['name'] for i in ast.literal_eval(obj)]
        except:
            return []

    df['genre_list'] = df['genres'].apply(convert)
    df['genres_str'] = df['genre_list'].apply(lambda x: " ".join(x))
    df['tags'] = df['overview'] + " " + df['genres_str']
    return df.reset_index(drop=True)

@st.cache_data
def build_vectors(tags):
    cv = CountVectorizer(max_features=5000, stop_words='english')
    v = cv.fit_transform(tags).toarray()
    return v, cosine_similarity(v)

movies_raw = load_data()
vectors, similarity = build_vectors(movies_raw['tags'])

# AI DESCRIPTION
def generate_description(movie, mood=None, watchlist=None):
    title = movie['title']
    overview = str(movie['overview'])
    genres = ", ".join(movie['genre_list'])
    year = str(movie['release_date'])[:4]

    desc = f"""
### 🎬 {title} ({year})

**🎭 Genres:** {genres}

**📖 Storyline:**  
{overview}
"""

    if any(g in movie['genre_list'] for g in ["Action", "Thriller"]):
        desc += "\n\n🔥 High-stakes action with intense pacing."
    if "Drama" in movie['genre_list']:
        desc += "\n🎭 Emotionally rich storytelling."
    if "Comedy" in movie['genre_list']:
        desc += "\n😂 Light and entertaining."
    if "Sci-Fi" in movie['genre_list']:
        desc += "\n🚀 Futuristic or mind-bending concepts."
    if "Romance" in movie['genre_list']:
        desc += "\n❤️ Emotional and relationship-driven."

    if watchlist:
        desc += "\n\n🤖 Matches your taste profile."
    if mood and mood != "Any":
        desc += f"\n🎯 Fits your {mood.lower()} mood."

    return desc

# ORIGINAL RECOMMENDER (UNCHANGED)
def score_to_weight(score):
    norm = (score - 50) / 50.0
    return norm * 2 if abs(norm) >= 0.4 else norm

def build_user_taste_vector():
    if not st.session_state.watchlist:
        return None, None

    pos_sum = np.zeros(vectors.shape[1])
    neg_sum = np.zeros(vectors.shape[1])
    pos_w = neg_w = 0.0

    for title, score in st.session_state.watchlist.items():
        idx = movies_raw[movies_raw['title'] == title].index
        if len(idx) == 0: continue
        vec = vectors[idx[0]]
        w = score_to_weight(score)
        if w > 0:
            pos_sum += vec * w; pos_w += w
        elif w < 0:
            neg_sum += vec * abs(w); neg_w += abs(w)

    return (pos_sum/pos_w if pos_w else None,
            neg_sum/neg_w if neg_w else None)

def recommend(base_movie, mood, genre, era, text):
    idx = movies_raw[movies_raw['title'] == base_movie].index[0]
    base_sim = similarity[idx]
    pos_vec, neg_vec = build_user_taste_vector()

    scored = []
    for i in range(len(movies_raw)):
        row = movies_raw.iloc[i]
        s = base_sim[i]

        if pos_vec is not None:
            s += 0.7 * cosine_similarity([pos_vec], [vectors[i]])[0][0]
        if neg_vec is not None:
            s -= 0.5 * cosine_similarity([neg_vec], [vectors[i]])[0][0]

        if mood != "Any" and any(g in row['genre_list'] for g in MOOD_GENRE_MAP[mood]):
            s += 0.25
        if genre != "Any" and genre in row['genre_list']:
            s += 0.25

        if row['title'] in st.session_state.watchlist:
            s = -9999

        scored.append((i, s))

    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[1:9]

# HELPERS
def get_poster(movie_id):
    try:
        data = requests.get(f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={TMDB_API_KEY}").json()
        return f"{IMAGE_BASE}{data['poster_path']}"
    except:
        return "https://via.placeholder.com/300x450?text=No+Image"

def get_trailer(movie_id):
    try:
        data = requests.get(f"https://api.themoviedb.org/3/movie/{movie_id}/videos?api_key={TMDB_API_KEY}").json()
        for v in data["results"]:
            if v["type"] == "Trailer":
                return f"https://youtube.com/watch?v={v['key']}"
    except:
        return None

# UI
st.set_page_config(layout="wide")
st.title("🎬 CineAI PRO")

mood = st.selectbox("Mood", ["Any", "Happy", "Sad", "Thrilling", "Romantic"])
genre = st.selectbox("Genre", ["Any", "Action", "Comedy", "Drama", "Sci-Fi", "Romance"])
selected_movie = st.selectbox("Movie", movies_raw['title'])

if st.button("Get Recommendations"):
    st.session_state.recs = recommend(selected_movie, mood, genre, "Any", "")

if st.session_state.recs:
    cols = st.columns(4)

    for i, rec in enumerate(st.session_state.recs):
        movie = movies_raw.iloc[rec[0]]

        with cols[i % 4]:
            st.image(get_poster(movie['id']))

            with st.expander(movie['title']):
                st.markdown(generate_description(movie, mood, st.session_state.watchlist))

                trailer = get_trailer(movie['id'])
                if trailer:
                    st.video(trailer)
