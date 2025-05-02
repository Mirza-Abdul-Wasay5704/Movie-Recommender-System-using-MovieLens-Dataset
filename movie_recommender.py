import pandas as pd
import numpy as np
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity
import time

# Set page configuration
st.set_page_config(
    page_title="Movie Recommender System",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for better UI
def apply_custom_css():
    st.markdown("""
    <style>
    .main {
        background-color: #f5f5f5;
    }
    .movie-card {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 15px;
    }
    .movie-title {
        font-size: 22px;
        font-weight: bold;
        color: #1E3A8A;
    }
    .movie-genres {
        font-size: 16px;
        color: #4B5563;
    }
    .similarity-score {
        font-size: 18px;
        font-weight: bold;
        color: #047857;
    }
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    .header {
        background-color: #1E3A8A;
        padding: 20px;
        border-radius: 10px;
        color: white;
        margin-bottom: 20px;
        text-align: center;
    }
    .button-container {
        display: flex;
        justify-content: center;
        margin-top: 20px;
    }
    </style>
    """, unsafe_allow_html=True)

# Load the data more efficiently
@st.cache_data
def load_data():
    movies = pd.read_csv('movies.csv')
    # Only load necessary columns from ratings to save memory
    ratings = pd.read_csv('ratings.csv', usecols=['userId', 'movieId', 'rating'])
    return movies, ratings

# More efficient preprocessing
@st.cache_data
def preprocess_data(ratings):
    # Create user-item matrix more efficiently
    # Filter users with at least 5 ratings to reduce matrix size
    user_counts = ratings['userId'].value_counts()
    active_users = user_counts[user_counts >= 5].index
    filtered_ratings = ratings[ratings['userId'].isin(active_users)]
    
    # Create a sparse pivot table
    user_item_matrix = filtered_ratings.pivot(index='userId', columns='movieId', values='rating')
    
    # Calculate mean rating for each user
    user_means = user_item_matrix.mean(axis=1)
    
    # Center the ratings by subtracting user means
    centered_ratings = user_item_matrix.sub(user_means, axis=0)
    
    # Fill NaN values with 0
    centered_ratings = centered_ratings.fillna(0)
    
    return centered_ratings, user_means

# Optimize similarity calculation with caching
@st.cache_data
def calculate_adjusted_cosine_similarity(centered_ratings):
    # Only compute similarity for movies with sufficient ratings
    # Convert to sparse matrix for more efficient computation
    item_similarity = cosine_similarity(centered_ratings.T)
    item_similarity = pd.DataFrame(item_similarity, 
                                 index=centered_ratings.columns, 
                                 columns=centered_ratings.columns)
    return item_similarity

def get_movie_recommendations(movie_title, movies, item_similarity, user_means, n_recommendations=5):
    # Find movie ID from title
    movie_id = movies[movies['title'].str.contains(movie_title, case=False, na=False)]['movieId'].values
    if len(movie_id) == 0:
        return None
    
    movie_id = movie_id[0]
    
    # Get similar movies
    similar_movies = item_similarity[movie_id].sort_values(ascending=False)
    
    # Exclude the input movie itself
    similar_movies = similar_movies[similar_movies.index != movie_id]
    
    # Get top N recommendations
    top_movies = similar_movies.head(n_recommendations)
    
    # Get movie titles for recommendations
    recommendations = movies[movies['movieId'].isin(top_movies.index)][['movieId', 'title', 'genres']]
    recommendations['similarity'] = recommendations['movieId'].map(top_movies)
    
    # Sort by similarity score in descending order
    recommendations = recommendations.sort_values('similarity', ascending=False)
    
    return recommendations

def main():
    apply_custom_css()
    
    # Create a header
    st.markdown('<div class="header"><h1>🎬 Movie Recommender System</h1></div>', unsafe_allow_html=True)
    
    # Improved intro text
    st.markdown("""
    <p style="font-size: 18px; text-align: center;">
    This system recommends movies based on item-based collaborative filtering using adjusted cosine similarity.
    Enter a movie title below to get personalized recommendations!
    </p>
    """, unsafe_allow_html=True)
    
    # Create a loading spinner for data loading
    with st.spinner('Loading data and preparing model...'):
        # Load data
        start_time = time.time()
        movies, ratings = load_data()
        
        # Preprocess data
        centered_ratings, user_means = preprocess_data(ratings)
        
        # Calculate item similarities
        item_similarity = calculate_adjusted_cosine_similarity(centered_ratings)
        load_time = time.time() - start_time
    
    st.success(f"Model ready! (Loaded in {load_time:.2f} seconds)")
    
    # Initialize session state for selected movie if it doesn't exist
    if 'selected_movie' not in st.session_state:
        st.session_state.selected_movie = "The Dark Knight"
    
    # Display a list of popular movies for easy selection (BEFORE the input field)
    with st.expander("Popular Movies - Click to select"):
        popular_movies = movies.sample(10)['title'].tolist()
        cols = st.columns(2)
        for i, pop_movie in enumerate(popular_movies):
            col_idx = i % 2
            with cols[col_idx]:
                if st.button(pop_movie, key=pop_movie):
                    st.session_state.selected_movie = pop_movie
                    st.experimental_rerun()  # Rerun to update the text input field
    
    # Create columns for better layout
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Get user input (now using the session state)
        movie_title = st.text_input("Enter a movie title:", value=st.session_state.selected_movie)
        # Update session state when manually changed
        if movie_title != st.session_state.selected_movie:
            st.session_state.selected_movie = movie_title
    
    with col2:
        # Center the button
        with st.container():
            st.markdown('<div class="button-container">', unsafe_allow_html=True)
            get_recs = st.button("Get Recommendations", type="primary")
            st.markdown('</div>', unsafe_allow_html=True)
    
    if get_recs:
        with st.spinner('Finding the best movies for you...'):
            start_time = time.time()
            recommendations = get_movie_recommendations(movie_title, movies, item_similarity, user_means)
            rec_time = time.time() - start_time
        
        if recommendations is None:
            st.error("Movie not found. Please try another title.")
        else:
            st.success(f"Found recommendations in {rec_time:.2f} seconds!")
            
            st.markdown("## Top 5 Recommended Movies")
            
            for idx, row in recommendations.iterrows():
                st.markdown(f"""
                <div class="movie-card">
                    <div class="movie-title">{row['title']}</div>
                    <div class="movie-genres">Genres: {row['genres']}</div>
                    <div class="similarity-score">Similarity Score: {row['similarity']:.4f}</div>
                </div>
                """, unsafe_allow_html=True)

if __name__ == "__main__":
    main() 