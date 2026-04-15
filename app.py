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
    "1950s": (1950, 1959),
    "1960s": (1960, 1969),
    "1970s": (1970, 1979),
    "1980s": (1980, 1989),
    "1990s": (1990, 1999),
    "2000s": (2000, 2009),
    "2010s": (2010, 2019),
    "2020s": (2020, 2030),
}

# SESSION
if "watchlist" not in st.session_state:
    st.session_state.watchlist = {}

if "selected_watch" not in st.session_state:
    st.session_state.selected_watch = None

if "recs" not in st.session_state:
    st.session_state.recs = []

# LOAD DATA
@st.cache_data
def load_data():
    df = pd.read_csv("tmdb_5000_movies.csv")
    df = df[['id', 'title', 'overview', 'genres', 'cast',
             'vote_average', 'release_date']].dropna()

    def convert(obj):
        L = []
        try:
            for i in ast.literal_eval(obj):
                L.append(i['name'])
        except:
            pass
        return L

    def get_cast(obj):
        L = []
        try:
            for i in ast.literal_eval(obj)[:3]:
                L.append(i['name'])
        except:
            pass
        return L

    df['genre_list'] = df['genres'].apply(convert)
    df['cast_list'] = df['cast'].apply(get_cast)

    df['tags'] = df['overview'] + " " + \
                 df['genre_list'].apply(lambda x: " ".join(x)) + " " + \
                 df['cast_list'].apply(lambda x: " ".join(x))

    return df.reset_index(drop=True)

@st.cache_data
def build_vectors(tags):
    cv = CountVectorizer(max_features=5000, stop_words='english')
    vectors = cv.fit_transform(tags).toarray()
    sim = cosine_similarity(vectors)
    return vectors, sim

movies_raw = load_data()
vectors, similarity = build_vectors(movies_raw['tags'])

# SCORE WEIGHT
def score_to_weight(score):
    norm = (score - 50) / 50.0
    return norm * 2 if abs(norm) >= 0.4 else norm

def build_user_taste_vector():
    if not st.session_state.watchlist:
        return None, None

    pos, neg = np.zeros(vectors.shape[1]), np.zeros(vectors.shape[1])
    pw, nw = 0, 0

    for t, s in st.session_state.watchlist.items():
        idx = movies_raw[movies_raw['title'] == t].index
        if len(idx) == 0: continue
        v = vectors[idx[0]]
        w = score_to_weight(s)

        if w > 0:
            pos += v * w; pw += w
        elif w < 0:
            neg += v * abs(w); nw += abs(w)

    return (pos/pw if pw else None), (neg/nw if nw else None)

# RECOMMENDATION
def recommend(base_movie, mood, genre, era, text):
    pos_vec, neg_vec = build_user_taste_vector()
    scored = []
    use_base = base_movie != "None"

    if use_base:
        idx = movies_raw[movies_raw['title'] == base_movie].index
        base_sim = similarity[idx[0]]

    for i in range(len(movies_raw)):
        row = movies_raw.iloc[i]
        s = 0

        if use_base:
            s += base_sim[i]

        if pos_vec is not None:
            s += 0.7 * cosine_similarity([pos_vec], [vectors[i]])[0][0]
        if neg_vec is not None:
            s -= 0.5 * cosine_similarity([neg_vec], [vectors[i]])[0][0]

        if mood != "Any" and any(g in row['genre_list'] for g in MOOD_GENRE_MAP[mood]):
            s += 0.3

        if genre != "Any" and genre in row['genre_list']:
            s += 0.3

        if era != "Any":
            try:
                y = int(str(row['release_date'])[:4])
                lo, hi = ERA_YEAR_MAP[era]
                if lo <= y <= hi:
                    s += 0.2
            except:
                pass

        if text:
            s += 0.1 * sum(1 for w in text.lower().split() if w in row['overview'].lower())

        if row['title'] in st.session_state.watchlist:
            s = -9999

        scored.append((i, s))

    scored.sort(key=lambda x: x[1], reverse=True)
    return [x for x in scored][:8]

# HELPERS
def get_poster(id):
    try:
        data = requests.get(f"https://api.themoviedb.org/3/movie/{id}?api_key={TMDB_API_KEY}").json()
        return IMAGE_BASE + data['poster_path'] if data.get("poster_path") else ""
    except:
        return ""

def get_trailer(id):
    try:
        data = requests.get(f"https://api.themoviedb.org/3/movie/{id}/videos?api_key={TMDB_API_KEY}").json()
        for v in data.get("results", []):
            if v["type"] == "Trailer":
                return f"https://youtube.com/watch?v={v['key']}"
    except:
        return None

# UI
st.title("🎬 CineAI PRO")

col1, col2 = st.columns(2)

with col1:
    mood = st.selectbox("Mood", ["Any","Happy","Sad","Thrilling","Romantic"])
    genre = st.selectbox("Genre", ["Any","Action","Comedy","Drama","Horror","Romance","Thriller"])

with col2:
    era = st.selectbox("Era", ["Any","1950s","1960s","1970s","1980s","1990s","2000s","2010s","2020s"])
    text = st.text_input("Keywords")

selected_movie = st.selectbox("Base Movie (Optional)", ["None"] + list(movies_raw['title']))

if st.button("Recommend"):
    st.session_state.recs = recommend(selected_movie, mood, genre, era, text)

# RESULTS
if st.session_state.recs:
    cols = st.columns(4)

    for i, rec in enumerate(st.session_state.recs):
        movie = movies_raw.iloc[rec[0]]

        with cols[i % 4]:
            st.image(get_poster(movie['id']))

            with st.expander(movie['title']):
                st.markdown("### 📖 Story")
                st.write(movie['overview'])

                st.markdown("### 🎭 Genres")
                st.write(", ".join(movie['genre_list']))

                st.markdown("### 👥 Cast")
                st.write(", ".join(movie['cast_list']))

                st.markdown("### ⭐ Rating")
                st.write(movie['vote_average'])

                # WHY
                reasons = []

                if genre != "Any" and genre in movie['genre_list']:
                    reasons.append("Genre match")

                if mood != "Any" and any(g in movie['genre_list'] for g in MOOD_GENRE_MAP[mood]):
                    reasons.append("Mood match")

                user_cast = []
                for t in st.session_state.watchlist:
                    idx = movies_raw[movies_raw['title'] == t].index
                    if len(idx):
                        user_cast += movies_raw.iloc[idx[0]]['cast_list']

                if any(a in user_cast for a in movie['cast_list']):
                    reasons.append("Actor you like")

                if reasons:
                    st.info("Why: " + ", ".join(reasons))

                trailer = get_trailer(movie['id'])
                if trailer:
                    st.video(trailer)

                score = st.slider("Rate", 0, 100, 50, key=f"s{i}")
                if st.button("Save", key=f"b{i}"):
                    st.session_state.watchlist[movie['title']] = score
                    st.rerun()

