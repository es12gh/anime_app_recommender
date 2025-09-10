# Anime Recommendation System (Streamlit)
# Full, structured, self-contained Python app.
# Save this file as `app.py` and run with `streamlit run app.py`.

"""
Features included:
1. Cleans `Title` column from the uploaded anime.csv (stored at /mnt/data/anime.csv).
2. Builds a TF-IDF vectorizer over cleaned anime titles and computes cosine similarity.
3. Content-based recommendations (TF-IDF + cosine similarity) with filtering by Year, Type, and Popularity (Rank).
4. Sorting options (Score, Year, Rank).
5. Jikan API integration for poster, synopsis, episodes, genres, and trailer preview (cached).
6. Grid-style UI with posters and "More Info" expanders.
7. Favorites/watchlist with persistent storage (favorites.json). Add & Remove supported.
8. Basic fuzzy matching for title lookup if the exact title isn't found.

Instructions:
- Requirements: pandas, scikit-learn, streamlit, requests
- Install: pip install pandas scikit-learn streamlit requests
- Run: streamlit run app.py
- The app will create a file `favorites.json` in the same folder to persist your favorites across sessions.
"""

import os
import json
import re
import urllib.parse
from difflib import get_close_matches

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st
import requests
st.set_page_config(page_title="Anime Recommender", layout='wide')
# ----------------- Configuration -----------------
DATA_PATH =  "anime.csv"  # change if needed
FAV_FILE = "favorites.json"
JIKAN_BASE = "https://api.jikan.moe/v4/anime"

# ----------------- Utility Functions -----------------

def clean_title(raw_title: str) -> str:
    """Return a cleaned anime title string from the raw Title field.
    Strategy:
      - Split on common type keywords (TV, Movie, OVA, ONA, Special) and take left side.
      - Remove trailing parenthetical info or dates using a regex.
    """
    if not isinstance(raw_title, str):
        return ""
    # split on type keywords (these may be attached without space in dataset)
    left = re.split(r'(?:TV|Movie|OVA|ONA|Special)', raw_title)[0]
    # remove leftover parentheses/brackets and trailing extra text
    left = re.sub(r"[\(\[\-].*$", "", left).strip()
    # final strip
    return left


def extract_type(raw_title: str) -> str:
    if not isinstance(raw_title, str):
        return "Other"
    if "Movie" in raw_title:
        return "Movie"
    if "OVA" in raw_title:
        return "OVA"
    if "ONA" in raw_title:
        return "ONA"
    if "TV" in raw_title:
        return "TV"
    return "Other"


def extract_year(raw_title: str):
    m = re.search(r"(19|20)\d{2}", str(raw_title))
    return int(m.group()) if m else None

# ----------------- Load & Prepare Dataset -----------------

@st.cache_data
def load_dataset(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Basic cleaning and feature extraction
    df['Clean_Title'] = df['Title'].apply(clean_title)
    df['Type'] = df['Title'].apply(extract_type)
    df['Year'] = df['Title'].apply(extract_year)
    df['Rank'] = pd.to_numeric(df['Rank'], errors='coerce')
    df['Score'] = pd.to_numeric(df['Score'], errors='coerce')
    return df

anime_df = load_dataset(DATA_PATH)

# ----------------- Vectorization & Similarity -----------------

@st.cache_data
def build_tfidf_similarity(titles_series: pd.Series):
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_mat = vectorizer.fit_transform(titles_series.fillna('').astype(str))
    cosine_sim = cosine_similarity(tfidf_mat, tfidf_mat)
    return vectorizer, tfidf_mat, cosine_sim

vectorizer, tfidf_matrix, cosine_sim = build_tfidf_similarity(anime_df['Clean_Title'])

# index map for title -> dataframe index
TITLE_TO_INDEX = pd.Series(anime_df.index, index=anime_df['Clean_Title'].str.lower())

# ----------------- Recommendation Logic -----------------

def recommend_anime(title: str, n: int = 5, year_filter=None, type_filter=None, popularity_filter=None, sort_by='Score'):
    """Return top-n similar anime rows (DataFrame) and an optional message if not exact match found."""
    if not isinstance(title, str) or title.strip() == "":
        return None, "Please provide a title."

    key = title.strip().lower()
    used_match = None
    if key not in TITLE_TO_INDEX:
        # try fuzzy match
        matches = get_close_matches(key, TITLE_TO_INDEX.index, n=1, cutoff=0.6)
        if matches:
            used_match = matches[0]
            idx = TITLE_TO_INDEX[used_match]
        else:
            return None, f"Title not found. Closest matches: {get_close_matches(key, TITLE_TO_INDEX.index, n=5)}"
    else:
        idx = TITLE_TO_INDEX[key]

    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    # exclude the queried item itself
    sim_scores = sim_scores[1:]
    indices = [i for i, score in sim_scores]

    results = anime_df.loc[indices, ['Rank', 'Clean_Title', 'Score', 'Year', 'Type']].copy()

    # Apply filters
    if year_filter:
        results = results[results['Year'] == year_filter]
    if type_filter and type_filter != 'All':
        results = results[results['Type'] == type_filter]
    if popularity_filter:
        results = results[results['Rank'] <= popularity_filter]

    # Sorting
    if sort_by == 'Score':
        results = results.sort_values(by='Score', ascending=False)
    elif sort_by == 'Year':
        results = results.sort_values(by='Year', ascending=False)
    elif sort_by == 'Rank':
        results = results.sort_values(by='Rank', ascending=True)

    return results.head(n), (f"Using fuzzy match: {used_match}" if used_match else None)

# ----------------- Jikan API (cached) -----------------

@st.cache_data(ttl=60*60*24)
def get_anime_info(title: str):
    """Query Jikan API for poster, synopsis, episodes, genres, trailer. Results cached for 24h."""
    try:
        q = urllib.parse.quote(title)
        url = f"{JIKAN_BASE}?q={q}&limit=1"
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        js = resp.json()
        if 'data' in js and len(js['data']) > 0:
            d = js['data'][0]
            poster = d.get('images', {}).get('jpg', {}).get('image_url')
            synopsis = d.get('synopsis') or 'No synopsis available.'
            episodes = d.get('episodes') or 'Unknown'
            genres = ', '.join([g['name'] for g in d.get('genres', [])])
            trailer = d.get('trailer', {}).get('url')
            return {
                'poster': poster,
                'synopsis': synopsis,
                'episodes': episodes,
                'genres': genres,
                'trailer': trailer
            }
    except Exception:
        return None
    return None

# ----------------- Favorites Persistence -----------------

def load_favorites() -> list:
    if os.path.exists(FAV_FILE):
        try:
            with open(FAV_FILE, 'r') as f:
                return json.load(f)
        except Exception:
            return []
    return []


def save_favorites(favs: list):
    try:
        with open(FAV_FILE, 'w') as f:
            json.dump(favs, f)
    except Exception as e:
        st.error(f"Failed to save favorites: {e}")

# ----------------- Streamlit UI -----------------

# st.set_page_config(page_title="Anime Recommender", layout='wide')
st.title("🎌 Anime Recommendation System")
st.write("Content-based recommendations (TF-IDF on anime titles) with posters, filters, and a persistent favorites list.")

# Initialize session state for favorites
if 'favorites' not in st.session_state:
    st.session_state.favorites = load_favorites()

# --- Controls ---
anime_list = sorted(anime_df['Clean_Title'].dropna().unique())
default_index = anime_list.index('Attack on Titan') if 'Attack on Titan' in anime_list else 0
anime_input = st.selectbox('Choose an anime title:', anime_list, index=default_index)

col1, col2, col3, col4 = st.columns(4)
with col1:
    year_filter = st.selectbox('Filter by Year', [None] + sorted(anime_df['Year'].dropna().unique().tolist()))
with col2:
    type_filter = st.selectbox('Filter by Type', ['All'] + anime_df['Type'].unique().tolist())
with col3:
    popularity_filter = st.selectbox('Popularity Filter', [None, 100, 500, 1000], index=0)
with col4:
    sort_by = st.selectbox('Sort By', ['Score', 'Year', 'Rank'])

num_recs = st.slider('Number of recommendations', 1, 12, 6)

if st.button('Recommend'):
    results, msg = recommend_anime(anime_input, n=num_recs, year_filter=year_filter, type_filter=type_filter, popularity_filter=popularity_filter, sort_by=sort_by)
    if results is None or results.empty:
        if msg:
            st.warning(msg)
        else:
            st.error('No matching recommendations found. Try adjusting filters!')
    else:
        st.subheader('Recommended Anime')
        # grid with 3 columns per row
        cards_per_row = 3
        cols = st.columns(cards_per_row)
        for i, row in results.reset_index(drop=True).iterrows():
            col = cols[i % cards_per_row]
            with col:
                info = get_anime_info(row['Clean_Title'])
                if info and info.get('poster'):
                    st.image(info['poster'], width=180)
                else:
                    st.write('No Image')
                st.markdown(f"**{row['Clean_Title']}**")
                st.write(f"⭐ Score: {row['Score']}")
                st.write(f"📅 Year: {row['Year']} | 🎬 Type: {row['Type']}")
                st.write(f"🏆 Rank: {row['Rank']}")

                # Add to favorites button (unique key per title)
                add_key = f"add_fav_{row['Clean_Title']}"
                if st.button('Add to Favorites ❤️', key=add_key):
                    if row['Clean_Title'] not in st.session_state.favorites:
                        st.session_state.favorites.append(row['Clean_Title'])
                        save_favorites(st.session_state.favorites)
                        st.success(f"Added {row['Clean_Title']} to favorites!")

                # More Info expander
                with st.expander('More Info'):
                    if info:
                        st.write(f"**Episodes:** {info['episodes']}")
                        st.write(f"**Genres:** {info['genres']}")
                        st.write(f"**Synopsis:** {info['synopsis']}")
                        if info.get('trailer'):
                            st.video(info['trailer'])

# --- Favorites display with remove support ---
if st.session_state.favorites:
    st.subheader('❤️ Your Favorites List')
    # show each favorite with a Remove button
    for fav in st.session_state.favorites.copy():
        colA, colB = st.columns([8, 1])
        with colA:
            st.write(fav)
        with colB:
            remove_key = f"rem_{fav}"
            if st.button('Remove', key=remove_key):
                try:
                    st.session_state.favorites.remove(fav)
                    save_favorites(st.session_state.favorites)
                    st.success(f"Removed {fav} from favorites")
                    # rerun to update the displayed list immediately
                    st.experimental_rerun()
                except ValueError:
                    st.warning('Item not found in favorites')

# Footer / usage notes
st.markdown("---")
st.caption('Notes: The app uses the Jikan API (unofficial MyAnimeList API). Jikan may impose rate-limits — API calls are cached to reduce requests.')

