import pandas as pd
import streamlit as st
import requests
from PIL import Image
from io import BytesIO
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Custom CSS for background and styling
st.markdown(
    """
    <style>
    body {
        background-color: #f4f4f4;
        font-family: 'Arial', sans-serif;
        margin: 0;
        padding: 0;
    }
    .stApp {
        background: linear-gradient(to right, #141e30, #243b55);
        color: white;
    }
    .stTitle {
        color: #f39c12;
        font-size: 36px;
        font-weight: bold;
        text-align: center;
    }
    .movie-card {
        background-color: rgba(255, 255, 255, 0.1);
        border-radius: 8px;
        padding: 10px;
        margin: 5px;
        text-align: center;
        color: white;
    }
    .movie-title {
        color: #f39c12;
        font-weight: bold;
    }
    .movie-genre {
        color: #9b59b6;
    }
    .movie-rating {
        color: #2ecc71;
        font-weight: bold;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Title of the App
st.markdown('<h1 class="stTitle">🎬 Movie Recommendation System 🎥</h1>', unsafe_allow_html=True)

df = pd.read_csv("imdb_top_1000.csv")


df['Genre'] = df['Genre'].fillna('')
df['IMDB_Rating'] = df['IMDB_Rating'].fillna(0)

df['Features'] = df['Genre'] + " " + df['Overview'] + " " + df['Director'] + " " + df['Star1'] + " " + df['Star2']


vectorizer = TfidfVectorizer(stop_words='english')
feature_matrix = vectorizer.fit_transform(df['Features'])


similarity = cosine_similarity(feature_matrix)

# Function to recommend movies
def recommend_movies(title, n=5):
    if title not in df['Series_Title'].values:
        return []
    idx = df[df['Series_Title'] == title].index[0]
    scores = list(enumerate(similarity[idx]))
    sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)[1:n+1]
    recommendations = [df.iloc[i[0]] for i in sorted_scores]
    return recommendations

# Function to fetch and enhance the image
def enhance_image(image_url):
    try:
        response = requests.get(image_url)
        img = Image.open(BytesIO(response.content))
        img.thumbnail((150, 200))  # Resize to fit card layout
        return img
    except Exception:
        return None


st.subheader("🔍 Search for a Movie:")
movie_list = df['Series_Title'].unique()
selected_movie = st.selectbox("Choose a movie:", movie_list)


if st.button("Get Recommendations"):
    recommendations = recommend_movies(selected_movie, n=5)
    if recommendations:
        st.markdown("### Recommended Movies:")
        cols = st.columns(len(recommendations))  # Create columns for display
        for idx, movie in enumerate(recommendations):
            with cols[idx]:
                st.markdown('<div class="movie-card">', unsafe_allow_html=True)
                movie_title = movie['Series_Title']
                movie_genre = movie['Genre']
                movie_rating = movie['IMDB_Rating']
                movie_poster_url = movie['Poster_Link']

                # Fetch and display poster
                enhanced_poster = enhance_image(movie_poster_url)
                if enhanced_poster:
                    st.image(enhanced_poster, use_container_width=True)
                else:
                    st.write("🎥 Poster Not Available")

                # Display movie details
                st.markdown(f'<p class="movie-title">{movie_title}</p>', unsafe_allow_html=True)
                st.markdown(f'<p class="movie-genre">{movie_genre}</p>', unsafe_allow_html=True)
                st.markdown(f'<p class="movie-rating">⭐ {movie_rating}</p>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.error("No recommendations found. Try another movie.")
