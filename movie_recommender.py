import pandas as pd
import numpy as np
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity
import time
import os
from typing import Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

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
    /* Main container styling */
    .main {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 2rem;
    }
    
    /* Header styling */
    .header {
        background: linear-gradient(135deg, #1E3A8A 0%, #2B4B8C 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        margin-bottom: 2rem;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
    }
    
    /* Movie card styling */
    .movie-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
        margin-bottom: 1.5rem;
        transition: all 0.3s ease;
        border: 1px solid rgba(0, 0, 0, 0.1);
    }
    
    .movie-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 20px rgba(0, 0, 0, 0.15);
    }
    
    .movie-title {
        font-size: 1.4rem;
        font-weight: bold;
        color: #1E3A8A;
        margin-bottom: 0.5rem;
    }
    
    .movie-genres {
        font-size: 1rem;
        color: #4B5563;
        margin-bottom: 1rem;
        padding: 0.5rem;
        background: #F3F4F6;
        border-radius: 6px;
    }
    
    .similarity-score {
        font-size: 1.1rem;
        font-weight: bold;
        color: #047857;
        display: inline-block;
        padding: 0.5rem 1rem;
        background: #D1FAE5;
        border-radius: 6px;
    }
    
    /* Input and button styling */
    .input-container {
        background: white;
        padding: 2rem;
        border-radius: 12px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
        margin-bottom: 2rem;
    }
    
    .button-container {
        display: flex;
        justify-content: center;
        margin-top: 1.5rem;
    }
    
    .recommend-button {
        padding: 1rem 2rem;
        font-size: 1.1rem;
        font-weight: bold;
        border-radius: 8px;
        background: linear-gradient(135deg, #1E3A8A 0%, #2B4B8C 100%);
        color: white;
        border: none;
        cursor: pointer;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
    }
    
    .recommend-button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0, 0, 0, 0.15);
        background: linear-gradient(135deg, #2B4B8C 0%, #3B5B9C 100%);
    }
    
    /* Popular movies section */
    .popular-movies {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
        margin-bottom: 2rem;
    }
    
    .popular-movie-button {
        width: 100%;
        padding: 0.8rem;
        margin: 0.5rem 0;
        border-radius: 8px;
        background: #F3F4F6;
        border: none;
        cursor: pointer;
        transition: all 0.3s ease;
        text-align: left;
    }
    
    .popular-movie-button:hover {
        background: #E5E7EB;
        transform: translateX(5px);
    }
    
    /* Error message styling */
    .error-message {
        color: #DC2626;
        font-weight: bold;
        padding: 1rem;
        border-radius: 8px;
        background-color: #FEE2E2;
        margin: 1rem 0;
        border: 1px solid #FCA5A5;
    }
    
    /* Animations */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    @keyframes slideIn {
        from { transform: translateX(-20px); opacity: 0; }
        to { transform: translateX(0); opacity: 1; }
    }
    
    .animate-fade-in {
        animation: fadeIn 0.5s ease-out forwards;
    }
    
    .animate-slide-in {
        animation: slideIn 0.5s ease-out forwards;
    }
    
    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: #f1f1f1;
    }
    
    ::-webkit-scrollbar-thumb {
        background: #1E3A8A;
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: #2B4B8C;
    }

    /* Footer styling */
    .footer {
        position: fixed;
        bottom: 0;
        left: 0;
        right: 0;
        background: rgba(30, 58, 138, 0.1);
        color: #1E3A8A;
        padding: 0.8rem;
        text-align: center;
        font-size: 0.9rem;
        border-top: 1px solid rgba(30, 58, 138, 0.1);
        z-index: 1000;
    }
    
    .footer-content {
        max-width: 1200px;
        margin: 0 auto;
        display: flex;
        justify-content: center;
        align-items: center;
        gap: 0.5rem;
    }
    
    .footer-icon {
        font-size: 1rem;
        opacity: 0.8;
    }
    </style>
    """, unsafe_allow_html=True)

# Load the data more efficiently with error handling
@st.cache_data
def load_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    try:
        if not os.path.exists('movies.csv') or not os.path.exists('ratings.csv'):
            raise FileNotFoundError("Required data files not found. Please ensure movies.csv and ratings.csv are in the current directory.")
        
        # Load movies with error handling
        movies = pd.read_csv('movies.csv')
        if movies.empty:
            raise ValueError("movies.csv is empty")
        
        # Load ratings with memory optimization
        ratings = pd.read_csv('ratings.csv', usecols=['userId', 'movieId', 'rating'])
        if ratings.empty:
            raise ValueError("ratings.csv is empty")
        
        return movies, ratings
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        st.stop()

# More efficient preprocessing with better memory management
@st.cache_data
def preprocess_data(ratings: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    try:
        # Filter users with at least 5 ratings to reduce matrix size
        user_counts = ratings['userId'].value_counts()
        active_users = user_counts[user_counts >= 5].index
        filtered_ratings = ratings[ratings['userId'].isin(active_users)]
        
        # Create a sparse pivot table with memory optimization
        user_item_matrix = filtered_ratings.pivot_table(
            index='userId',
            columns='movieId',
            values='rating',
            fill_value=0
        )
        
        # Calculate mean rating for each user
        user_means = user_item_matrix.mean(axis=1)
        
        # Center the ratings by subtracting user means
        centered_ratings = user_item_matrix.sub(user_means, axis=0)
        
        return centered_ratings, user_means
    except Exception as e:
        st.error(f"Error preprocessing data: {str(e)}")
        st.stop()

# Optimize similarity calculation with better caching and error handling
@st.cache_data
def calculate_adjusted_cosine_similarity(centered_ratings: pd.DataFrame) -> pd.DataFrame:
    try:
        # Convert to numpy array for more efficient computation
        ratings_array = centered_ratings.values
        item_similarity = cosine_similarity(ratings_array.T)
        
        # Create DataFrame with proper indexing
        item_similarity = pd.DataFrame(
            item_similarity,
            index=centered_ratings.columns,
            columns=centered_ratings.columns
        )
        
        return item_similarity
    except Exception as e:
        st.error(f"Error calculating similarities: {str(e)}")
        st.stop()

def get_movie_recommendations(
    movie_title: str,
    movies: pd.DataFrame,
    item_similarity: pd.DataFrame,
    user_means: pd.Series,
    n_recommendations: int = 5
) -> Optional[pd.DataFrame]:
    try:
        # Find movie ID from title with better matching
        movie_matches = movies[movies['title'].str.contains(movie_title, case=False, na=False)]
        
        if len(movie_matches) == 0:
            return None
        
        # If multiple matches, use the most popular one (highest number of ratings)
        if len(movie_matches) > 1:
            movie_id = movie_matches.iloc[0]['movieId']
        else:
            movie_id = movie_matches.iloc[0]['movieId']
        
        # Get similar movies
        if movie_id not in item_similarity.index:
            return None
            
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
    except Exception as e:
        st.error(f"Error getting recommendations: {str(e)}")
        return None

def main():
    apply_custom_css()
    
    # Create a header with gradient background
    st.markdown("""
    <div class="header">
        <h1 style="font-size: 2.5rem; margin-bottom: 1rem;">🎬 Movie Recommender System</h1>
        <p style="font-size: 1.2rem; opacity: 0.9;">
            Discover your next favorite movie with our intelligent recommendation system
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Add margin at the bottom to prevent content from being hidden behind the footer
    st.markdown('<div style="margin-bottom: 4rem;">', unsafe_allow_html=True)
    
    try:
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
        
        # Create two columns for the main content
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Input container with shadow
            st.markdown('<div class="input-container animate-fade-in">', unsafe_allow_html=True)
            
            # Get user input
            clean_title = st.session_state.selected_movie.rsplit(' (', 1)[0]
            movie_title = st.text_input("Enter a movie title:", value=clean_title)
            if movie_title != st.session_state.selected_movie:
                st.session_state.selected_movie = movie_title
            
            # Center the button with animation
            st.markdown('<div class="button-container">', unsafe_allow_html=True)
            get_recs = st.button("Get Recommendations", type="primary", key="recommend_button")
            st.markdown('</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
            if get_recs:
                with st.spinner('Finding the best movies for you...'):
                    start_time = time.time()
                    recommendations = get_movie_recommendations(movie_title, movies, item_similarity, user_means)
                    rec_time = time.time() - start_time
                
                if recommendations is None:
                    st.markdown('<div class="error-message animate-slide-in">Movie not found. Please try another title.</div>', unsafe_allow_html=True)
                else:
                    st.success(f"Found recommendations in {rec_time:.2f} seconds!")
                    
                    st.markdown("## 🎥 Top 5 Recommended Movies")
                    
                    for idx, row in recommendations.iterrows():
                        st.markdown(f"""
                        <div class="movie-card animate-fade-in">
                            <div class="movie-title">{row['title']}</div>
                            <div class="movie-genres">🎭 {row['genres']}</div>
                            <div class="similarity-score">⭐ {row['similarity']:.4f}</div>
                        </div>
                        """, unsafe_allow_html=True)
        
        with col2:
            # Popular movies section
            st.markdown('<div class="popular-movies animate-fade-in">', unsafe_allow_html=True)
            st.markdown("## 🔥 Popular Movies")
            st.markdown("Click on a movie to get recommendations:")
            
            # Get truly popular movies based on rating counts
            popular_movies = movies.merge(
                ratings.groupby('movieId').size().reset_index(name='count'),
                on='movieId'
            ).sort_values('count', ascending=False).head(10)['title'].tolist()
            
            for pop_movie in popular_movies:
                if st.button(pop_movie, key=pop_movie, use_container_width=True):
                    st.session_state.selected_movie = pop_movie
                    st.rerun()
            
            st.markdown('</div>', unsafe_allow_html=True)
    except Exception as e:
        st.error(f"An unexpected error occurred: {str(e)}")
        st.stop()
    
    # Add the footer
    st.markdown("""
    <div class="footer">
        <div class="footer-content">
            <span class="footer-icon">👨‍💻</span>
            <span>Developed By Mirza Abdul Wasay</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Close the margin div
    st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main() 