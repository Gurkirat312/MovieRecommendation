import pandas as pd
import streamlit as st
import requests
from PIL import Image
from io import BytesIO
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load the dataset
df = pd.read_csv("imdb_top_1000.csv")

# Preview the dataset
print(df.head())

# Fill missing values in relevant columns
df['Genre'] = df['Genre'].fillna('')
df['IMDB_Rating'] = df['IMDB_Rating'].fillna(0)

# Combine important features into a single string
df['Features'] = df['Genre'] + " " + df['Overview'] + " " + df['Director'] + " " + df['Star1'] + " " + df['Star2']

# Vectorize the combined features
vectorizer = CountVectorizer(stop_words='english')
feature_matrix = vectorizer.fit_transform(df['Features'])

# Calculate cosine similarity
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
        # Fetch image from URL
        response = requests.get(image_url)
        img = Image.open(BytesIO(response.content))
        
        # Ensure the image is not too large for display
        img.thumbnail((150, 150))  # Resize image to a max width/height of 150px for better display
        return img
    except Exception as e:
        st.error(f"Error fetching image: {e}")
        return None

# Streamlit app setup
st.title("IMDb Movie Recommendation System")

# Select a movie
movie_list = df['Series_Title'].unique()
selected_movie = st.selectbox("Choose a movie:", movie_list)

# Display recommendations
if st.button("Get Recommendations"):
    recommendations = recommend_movies(selected_movie, n=5)
    if recommendations:
        # Create columns for displaying movie posters and details in a row
        columns = st.columns(5)  # Create 5 columns for the 5 recommendations

        for i, movie in enumerate(recommendations):
            movie_title = movie['Series_Title']
            movie_genre = movie['Genre']
            movie_rating = movie['IMDB_Rating']
            movie_poster_url = movie['Poster_Link']

            # Fetch and enhance the movie poster image
            enhanced_poster = enhance_image(movie_poster_url)

            # Display movie poster and details in the corresponding column
            with columns[i]:
                if enhanced_poster:
                    # Display the enhanced image with movie details
                    st.image(enhanced_poster, caption=f"{movie_title} ({movie_genre}) - {movie_rating}", use_container_width=True)
                else:
                    st.write(f"Image not available for {movie_title}")
    else:
        st.write("No recommendations found.")