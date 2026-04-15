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

# ✅ NEW: AI DESCRIPTION FUNCTION
def generate_description(movie, mood=None, watchlist=None):
    title = movie['title']
    overview = str(movie['overview'])
    genres = ", ".join(movie['genre_list'])
    year = str(movie['release_date'])[:4]

    desc = f"""
### 🎬 {title} ({year})

**🎭 Genre:** {genres}

**📖 Storyline:**  
{overview}
"""

    if any(g in movie['genre_list'] for g in ["Action", "Thriller"]):
        desc += "\n\n🔥 High-energy action with intense and gripping moments."
    if "Drama" in movie['genre_list']:
        desc += "\n🎭 Emotionally rich storytelling with strong characters."
    if "Comedy" in movie['genre_list']:
        desc += "\n😂 Fun, light-hearted, and entertaining."
    if "Sci-Fi" in movie['genre_list']:
        desc += "\n🚀 Explores futuristic or mind-bending concepts."
    if "Romance" in movie['genre_list']:
        desc += "\n❤️ Focus on relationships and emotional connections."

    if watchlist:
        desc += "\n\n🤖 **Why this fits you:** Matches your taste profile."

    if mood and mood != "Any":
        desc += f"\n• Fits your **{mood.lower()} mood**"

    return desc

# SESSION
if "watchlist" not in st.session_state:
    st.session_state.watchlist = {}
elif isinstance(st.session_state.watchlist, list):
    st.session_state.watchlist = {t: 50 for t in st.session_state.watchlist}

if "selected_watch" not in st.session_state:
    st.session_state.selected_watch = None
if "recs" not in st.session_state:
    st.session_state.recs = []

# LOAD DATA
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
    df['tags'] = df['overview'] + " " + df['genre_list'].apply(lambda x: " ".join(x))
    return df.reset_index(drop=True)

@st.cache_data
def build_vectors(tags):
    cv = CountVectorizer(max_features=5000, stop_words='english')
    v = cv.fit_transform(tags).toarray()
    return v, cosine_similarity(v)

movies_raw = load_data()
vectors, similarity = build_vectors(movies_raw['tags'])

# RECOMMEND (same as yours, unchanged logic)
def recommend(base_movie, mood_filter, genre_filter, era_filter, text_filter):
    idx = movies_raw[movies_raw['title'] == base_movie].index[0]
    base_sim = similarity[idx]
    scored = []

    for i in range(len(movies_raw)):
        row = movies_raw.iloc[i]
        s = base_sim[i]

        if mood_filter != "Any":
            if any(g in row['genre_list'] for g in MOOD_GENRE_MAP[mood_filter]):
                s += 0.25
        if genre_filter != "Any" and genre_filter in row['genre_list']:
            s += 0.25

        scored.append((i, s))

    return sorted(scored, key=lambda x: x[1], reverse=True)[1:9]

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
                
                # ✅ REPLACED OVERVIEW WITH AI DESCRIPTION
                st.markdown(generate_description(
                    movie,
                    mood=mood,
                    watchlist=st.session_state.watchlist
                ))

                trailer = get_trailer(movie['id'])
                if trailer:
                    st.video(trailer)

# DETAIL PANEL
if st.session_state.selected_watch:
    movie = movies_raw[movies_raw['title'] == st.session_state.selected_watch].iloc[0]
    
    st.markdown("---")
    st.markdown("## 🎬 Movie Detail")

    # ✅ ALSO USING AI DESCRIPTION HERE
    st.markdown(generate_description(
        movie,
        mood=mood,
        watchlist=st.session_state.watchlist
    ))
