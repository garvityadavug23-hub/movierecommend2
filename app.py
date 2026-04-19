import streamlit as st
import pandas as pd
import requests
import os
import ast
import numpy as np
import streamlit.components.v1 as components
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------- CONFIG ----------
TMDB_API_KEY = os.getenv("TMDB_API_KEY")
IMAGE_BASE = "https://image.tmdb.org/t/p/w500"

st.set_page_config(layout="wide", page_title="🎬 CineAI", page_icon="🎬")

# ---------- CUSTOM CSS ----------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=Inter:wght@300;400;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
        background-color: #0d0d0d;
        color: #f0f0f0;
    }
    h1 { font-family: 'Bebas Neue', sans-serif; font-size: 3rem; letter-spacing: 4px; color: #E50914; }
    h2, h3 { font-family: 'Bebas Neue', sans-serif; letter-spacing: 2px; color: #E50914; }

    .stButton > button {
        background: #E50914;
        color: white;
        border: none;
        border-radius: 4px;
        font-weight: 600;
        transition: all 0.2s;
    }
    .stButton > button:hover { background: #b0070f; transform: scale(1.03); }

    .movie-card {
        background: #1a1a1a;
        border-radius: 10px;
        padding: 10px;
        margin-bottom: 10px;
        border: 1px solid #2a2a2a;
        transition: border 0.2s;
    }
    .movie-card:hover { border: 1px solid #E50914; }

    .rating-badge {
        background: #E50914;
        color: white;
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 0.75rem;
        font-weight: bold;
    }
    .genre-tag {
        display: inline-block;
        background: #2a2a2a;
        color: #aaa;
        padding: 2px 10px;
        border-radius: 12px;
        font-size: 0.75rem;
        margin: 2px;
    }
    .detail-label { color: #888; font-size: 0.8rem; text-transform: uppercase; letter-spacing: 1px; }
    .detail-value { color: #fff; font-size: 1rem; font-weight: 600; }

    [data-testid="stSidebar"] {
        background: #111;
        border-right: 1px solid #222;
    }
    .stSelectbox label, .stTextInput label { color: #aaa !important; }
    .stExpander { background: #1a1a1a; border: 1px solid #2a2a2a; border-radius: 8px; }
</style>
""", unsafe_allow_html=True)

# ---------- LOAD DATA ----------
@st.cache_data
def load_data():
    movies = pd.read_csv("tmdb_5000_movies.csv")
    credits = None
    try:
        credits = pd.read_csv("tmdb_5000_credits.csv")
    except FileNotFoundError:
        pass

    movies = movies[['id', 'title', 'overview', 'genres', 'vote_average',
                      'release_date', 'runtime', 'budget', 'revenue',
                      'vote_count', 'popularity']].dropna(subset=['overview'])

    # Parse genres from JSON string
    def parse_genres(genre_str):
        try:
            return [g['name'] for g in ast.literal_eval(genre_str)]
        except:
            return []

    movies['genre_list'] = movies['genres'].apply(parse_genres)
    movies['genre_str'] = movies['genre_list'].apply(lambda x: ' '.join(x))
    movies['year'] = pd.to_datetime(movies['release_date'], errors='coerce').dt.year

    if credits is not None:
        credits = credits[['movie_id', 'cast', 'crew']] if 'movie_id' in credits.columns else credits[['id', 'cast', 'crew']].rename(columns={'id': 'movie_id'})
        movies = movies.merge(credits, left_on='id', right_on='movie_id', how='left')

        def parse_cast(cast_str):
            try:
                return [c['name'] for c in ast.literal_eval(cast_str)[:5]]
            except:
                return []

        def parse_director(crew_str):
            try:
                for c in ast.literal_eval(crew_str):
                    if c['job'] == 'Director':
                        return c['name']
            except:
                pass
            return "N/A"

        movies['cast_list'] = movies['cast'].apply(parse_cast) if 'cast' in movies.columns else [[] for _ in range(len(movies))]
        movies['director'] = movies['crew'].apply(parse_director) if 'crew' in movies.columns else ['N/A'] * len(movies)
    else:
        movies['cast_list'] = [[] for _ in range(len(movies))]
        movies['director'] = ['N/A'] * len(movies)

    return movies.reset_index(drop=True)

movies = load_data()

# ---------- SIMILARITY MODEL ----------
@st.cache_resource
def create_similarity(_data):
    data = _data.copy()
    data['tags'] = data['overview'].fillna('') + ' ' + data['genre_str'].fillna('')
    cv = CountVectorizer(max_features=5000, stop_words='english')
    vectors = cv.fit_transform(data['tags']).toarray()
    return cosine_similarity(vectors)

similarity = create_similarity(movies)

# ---------- SESSION STATE ----------
for key, default in [
    ("watchlist", {}),           # {title: {rating, movie_idx}}
    ("selected_watch", None),
    ("poster_cache", {}),
]:
    if key not in st.session_state:
        st.session_state[key] = default

# ---------- TMDB HELPERS ----------
def search_movie_tmdb(title, year=None):
    """Search TMDB by title and return movie_id + poster_path."""
    if not TMDB_API_KEY:
        return None, None
    cache_key = f"search_{title}"
    if cache_key in st.session_state.poster_cache:
        return st.session_state.poster_cache[cache_key]
    try:
        params = {"api_key": TMDB_API_KEY, "query": title}
        if year:
            params["year"] = int(year)
        r = requests.get("https://api.themoviedb.org/3/search/movie", params=params, timeout=5)
        results = r.json().get("results", [])
        if results:
            res = results[0]
            val = (res.get("id"), res.get("poster_path"))
            st.session_state.poster_cache[cache_key] = val
            return val
    except:
        pass
    return None, None

def get_poster_url(movie_row):
    """Get poster URL using local id first, then search fallback."""
    if not TMDB_API_KEY:
        return f"https://placehold.co/300x450/1a1a1a/E50914?text={movie_row['title'][:15]}"
    tmdb_id = int(movie_row['id'])
    cache_key = f"poster_{tmdb_id}"
    if cache_key in st.session_state.poster_cache:
        return st.session_state.poster_cache[cache_key]
    try:
        r = requests.get(
            f"https://api.themoviedb.org/3/movie/{tmdb_id}",
            params={"api_key": TMDB_API_KEY},
            timeout=5
        )
        data = r.json()
        path = data.get("poster_path")
        if path:
            url = f"{IMAGE_BASE}{path}"
            st.session_state.poster_cache[cache_key] = url
            return url
    except:
        pass
    return f"https://placehold.co/300x450/1a1a1a/E50914?text={movie_row['title'][:15]}"

def get_trailer_key(tmdb_id):
    """Return YouTube video key for trailer."""
    if not TMDB_API_KEY:
        return None
    cache_key = f"trailer_{tmdb_id}"
    if cache_key in st.session_state.poster_cache:
        return st.session_state.poster_cache[cache_key]
    try:
        r = requests.get(
            f"https://api.themoviedb.org/3/movie/{int(tmdb_id)}/videos",
            params={"api_key": TMDB_API_KEY},
            timeout=5
        )
        for vid in r.json().get("results", []):
            if vid["type"] == "Trailer" and vid["site"] == "YouTube":
                key = vid["key"]
                st.session_state.poster_cache[cache_key] = key
                return key
    except:
        pass
    return None

def embed_youtube(video_key):
    """Embed YouTube video via iframe."""
    components.html(
        f"""<iframe width="100%" height="315"
            src="https://www.youtube.com/embed/{video_key}"
            frameborder="0"
            allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
            allowfullscreen></iframe>""",
        height=320
    )

# ---------- RECOMMEND ----------
def recommend(movie_title, top_n=8):
    """Standard cosine similarity recommendations."""
    matches = movies[movies['title'] == movie_title]
    if matches.empty:
        return []
    idx = matches.index[0]
    distances = list(enumerate(similarity[idx]))
    return sorted(distances, key=lambda x: x[1], reverse=True)[1:top_n + 1]

def recommend_from_ratings(top_n=10):
    """
    Rating-based recommendations:
    - Build a weighted average similarity vector from all rated movies.
    - Movies rated 4-5 positively influence, rated 1-2 negatively influence.
    - Returns top N movies not already in watchlist.
    """
    if not st.session_state.watchlist:
        return []

    weighted_sim = np.zeros(len(movies))
    watchlist_indices = set()

    for title, info in st.session_state.watchlist.items():
        movie_idx = info.get("movie_idx")
        rating = info.get("rating", 3)
        if movie_idx is None:
            continue
        watchlist_indices.add(movie_idx)
        # Weight: rating 5 → +2.0, rating 4 → +1.0, rating 3 → 0, rating 2 → -1.0, rating 1 → -2.0
        weight = (rating - 3)
        weighted_sim += weight * similarity[movie_idx]

    # Zero out already-watched movies
    for idx in watchlist_indices:
        weighted_sim[idx] = -999

    top_indices = np.argsort(weighted_sim)[::-1][:top_n]
    return [(int(i), float(weighted_sim[i])) for i in top_indices if weighted_sim[i] > -999]

# ---------- MOVIE DETAIL PANEL ----------
def show_movie_detail(movie):
    """Show rich movie detail layout."""
    poster_url = get_poster_url(movie)
    trailer_key = get_trailer_key(movie['id'])

    col1, col2 = st.columns([1, 2])
    with col1:
        st.image(poster_url, use_container_width=True)

    with col2:
        year = int(movie['year']) if pd.notna(movie.get('year')) else "N/A"
        runtime = f"{int(movie['runtime'])} min" if pd.notna(movie.get('runtime')) and movie['runtime'] > 0 else "N/A"
        score = round(movie['vote_average'], 1)
        stars = "⭐" * int(score // 2)

        st.markdown(f"## {movie['title']} ({year})")
        st.markdown(f"**{stars}** `{score}/10` &nbsp;&nbsp; `{int(movie.get('vote_count', 0)):,} votes`", unsafe_allow_html=True)

        # Genre tags
        genres_html = "".join([f'<span class="genre-tag">{g}</span>' for g in movie.get('genre_list', [])])
        st.markdown(genres_html, unsafe_allow_html=True)
        st.markdown("")

        # Details grid
        details = {
            "Director": movie.get('director', 'N/A'),
            "Runtime": runtime,
            "Release Year": year,
            "Budget": f"${int(movie['budget']):,}" if pd.notna(movie.get('budget')) and movie['budget'] > 0 else "N/A",
            "Revenue": f"${int(movie['revenue']):,}" if pd.notna(movie.get('revenue')) and movie['revenue'] > 0 else "N/A",
        }
        d1, d2 = st.columns(2)
        items = list(details.items())
        for i, (label, value) in enumerate(items):
            col = d1 if i % 2 == 0 else d2
            col.markdown(f'<div class="detail-label">{label}</div><div class="detail-value">{value}</div><br>', unsafe_allow_html=True)

        cast = movie.get('cast_list', [])
        if cast:
            st.markdown(f'<div class="detail-label">Top Cast</div><div class="detail-value">{" · ".join(cast)}</div>', unsafe_allow_html=True)

    st.markdown("### 📖 Overview")
    st.write(movie['overview'])

    if trailer_key:
        st.markdown("### 🎬 Trailer")
        embed_youtube(trailer_key)
    elif TMDB_API_KEY:
        st.info("No trailer available for this movie.")
    else:
        st.warning("Set TMDB_API_KEY environment variable to load trailers and posters.")

# ==========================================================
# MAIN UI
# ==========================================================

st.title("🎬 CINEAI")
st.markdown("*AI-powered movie recommendations*")
st.markdown("---")

tabs = st.tabs(["🔍 Discover", "📌 My Watchlist", "🤖 AI Picks (from your ratings)"])

# ==================== TAB 1: DISCOVER ====================
with tabs[0]:
    col1, col2 = st.columns([3, 1])
    with col1:
        selected_movie = st.selectbox("Pick a base movie", movies['title'].values, key="base_movie")
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        run_btn = st.button("🎥 Find Similar Movies", use_container_width=True)

    if run_btn:
        recs = recommend(selected_movie)

        if not recs:
            st.error("Couldn't find recommendations for that movie.")
        else:
            # Similarity bar chart
            sim_df = pd.DataFrame({
                "Movie": [movies.iloc[r[0]]['title'][:25] for r in recs],
                "Similarity Score": [r[1] for r in recs]
            }).sort_values("Similarity Score", ascending=False)

            st.subheader(f"📊 Similarity to '{selected_movie}'")
            st.bar_chart(sim_df.set_index("Movie"))
            st.markdown("---")

            st.subheader("🎬 Recommended Movies")
            cols = st.columns(4)

            for i, rec in enumerate(recs):
                movie = movies.iloc[rec[0]]
                score = round(movie['vote_average'], 1)

                with cols[i % 4]:
                    poster_url = get_poster_url(movie)
                    st.image(poster_url, use_container_width=True)

                    genres = ", ".join(movie.get('genre_list', [])[:2])
                    year = int(movie['year']) if pd.notna(movie.get('year')) else ""

                    st.markdown(f"**{movie['title']}** ({year})")
                    st.markdown(f"⭐ `{score}/10` · _{genres}_")

                    btn_col1, btn_col2 = st.columns(2)
                    with btn_col1:
                        if st.button("🔍 Details", key=f"det_{i}"):
                            st.session_state.selected_watch = movie['title']
                            st.rerun()
                    with btn_col2:
                        already = movie['title'] in st.session_state.watchlist
                        if not already:
                            if st.button("➕ Watch", key=f"add_{i}"):
                                st.session_state.watchlist[movie['title']] = {
                                    "rating": None,
                                    "movie_idx": rec[0]
                                }
                                st.success(f"Added!")
                                st.rerun()
                        else:
                            st.markdown("✅ *In list*")
                    st.markdown("---")

    # Movie detail panel (shown below if selected)
    if st.session_state.selected_watch:
        match = movies[movies['title'] == st.session_state.selected_watch]
        if not match.empty:
            st.markdown("---")
            st.subheader(f"📽️ Movie Details")
            if st.button("✖ Close Details"):
                st.session_state.selected_watch = None
                st.rerun()
            show_movie_detail(match.iloc[0])

# ==================== TAB 2: WATCHLIST ====================
with tabs[1]:
    st.subheader("📌 My Watchlist")

    if not st.session_state.watchlist:
        st.info("Your watchlist is empty. Add movies from the Discover tab!")
    else:
        st.markdown("Rate movies you've watched (1–5⭐) to power your AI recommendations.")
        st.markdown("---")

        to_remove = []

        for title, info in st.session_state.watchlist.items():
            movie_idx = info.get("movie_idx")
            movie = movies.iloc[movie_idx] if movie_idx is not None else None

            with st.container():
                c1, c2, c3, c4 = st.columns([1, 3, 2, 1])

                with c1:
                    if movie is not None:
                        st.image(get_poster_url(movie), width=80)

                with c2:
                    year = int(movie['year']) if movie is not None and pd.notna(movie.get('year')) else ""
                    st.markdown(f"**{title}** ({year})")
                    if movie is not None:
                        genres = ", ".join(movie.get('genre_list', [])[:3])
                        score = round(movie['vote_average'], 1)
                        st.caption(f"⭐ TMDB: {score}/10 · {genres}")

                with c3:
                    current_rating = info.get("rating")
                    rating = st.select_slider(
                        "Your Rating",
                        options=[1, 2, 3, 4, 5],
                        value=current_rating if current_rating else 3,
                        key=f"rate_{title}",
                        format_func=lambda x: ["😞 1", "😐 2", "🙂 3", "😊 4", "🤩 5"][x - 1]
                    )
                    if st.button("Save Rating", key=f"save_{title}"):
                        st.session_state.watchlist[title]["rating"] = rating
                        st.success("Rating saved!")
                        st.rerun()

                with c4:
                    if st.button("🔍", key=f"wdet_{title}", help="View details"):
                        st.session_state.selected_watch = title
                        st.rerun()
                    if st.button("🗑️", key=f"del_{title}", help="Remove"):
                        to_remove.append(title)

                st.markdown("---")

        for t in to_remove:
            del st.session_state.watchlist[t]
        if to_remove:
            st.rerun()

        # Show selected movie detail from watchlist
        if st.session_state.selected_watch:
            match = movies[movies['title'] == st.session_state.selected_watch]
            if not match.empty:
                st.subheader(f"📽️ Details: {st.session_state.selected_watch}")
                if st.button("✖ Close", key="close_wdet"):
                    st.session_state.selected_watch = None
                    st.rerun()
                show_movie_detail(match.iloc[0])

# ==================== TAB 3: AI PICKS ====================
with tabs[2]:
    st.subheader("🤖 Personalized AI Picks")

    rated = {t: i for t, i in st.session_state.watchlist.items() if i.get("rating") is not None}

    if not rated:
        st.info("Rate at least one movie in your Watchlist to get personalized recommendations!")
        st.markdown("**How it works:** Rate movies 1–5 stars. Movies you loved (4–5⭐) pull similar films up. Movies you disliked (1–2⭐) push similar films down. The AI learns your taste.")
    else:
        st.markdown(f"Based on **{len(rated)} rated movie(s)** in your watchlist:")
        for t, i in rated.items():
            r = i['rating']
            emoji = ["😞", "😐", "🙂", "😊", "🤩"][r - 1]
            st.markdown(f"- {emoji} **{t}** — {r}/5")

        st.markdown("---")

        ai_recs = recommend_from_ratings(top_n=12)

        if not ai_recs:
            st.warning("Not enough data to generate recommendations. Try rating more movies.")
        else:
            st.subheader("🎬 Movies You'll Probably Love")
            cols = st.columns(4)

            for i, (movie_idx, score) in enumerate(ai_recs):
                movie = movies.iloc[movie_idx]
                with cols[i % 4]:
                    poster_url = get_poster_url(movie)
                    st.image(poster_url, use_container_width=True)

                    year = int(movie['year']) if pd.notna(movie.get('year')) else ""
                    genres = ", ".join(movie.get('genre_list', [])[:2])
                    tmdb_score = round(movie['vote_average'], 1)

                    st.markdown(f"**{movie['title']}** ({year})")
                    st.markdown(f"⭐ `{tmdb_score}/10` · _{genres}_")

                    ai_score_pct = min(100, max(0, int((score + 2) / 4 * 100)))
                    st.progress(ai_score_pct, text=f"Match: {ai_score_pct}%")

                    btn1, btn2 = st.columns(2)
                    with btn1:
                        if st.button("🔍", key=f"aidet_{i}"):
                            st.session_state.selected_watch = movie['title']
                            st.rerun()
                    with btn2:
                        if movie['title'] not in st.session_state.watchlist:
                            if st.button("➕", key=f"aiadd_{i}"):
                                st.session_state.watchlist[movie['title']] = {
                                    "rating": None,
                                    "movie_idx": movie_idx
                                }
                                st.rerun()
                    st.markdown("---")

        # Detail panel
        if st.session_state.selected_watch:
            match = movies[movies['title'] == st.session_state.selected_watch]
            if not match.empty:
                st.markdown("---")
                if st.button("✖ Close", key="close_aidet"):
                    st.session_state.selected_watch = None
                    st.rerun()
                show_movie_detail(match.iloc[0])
