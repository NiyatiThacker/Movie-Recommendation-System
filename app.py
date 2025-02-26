import requests
from flask import Flask, request, render_template, jsonify
import pandas as pd
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Load datasets
movies = pd.read_csv("movies_metadata_cleaned.csv")  # Movie dataset
ratings = pd.read_csv("ratings_small.csv")  # User ratings dataset

# ✅ Ensure Movie ID is Numeric
movies['id'] = pd.to_numeric(movies['id'], errors='coerce')  # Convert to numeric
movies.dropna(subset=['id'], inplace=True)  # Remove NaN values
movies['id'] = movies['id'].astype(int)  # Convert to integer

# ✅ Fill missing overviews
movies['overview'] = movies['overview'].fillna("No description available")

# Train SVD Model (Collaborative Filtering)
reader = Reader(rating_scale=(0.5, 5))
data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)
trainset, testset = train_test_split(data, test_size=0.2)
svd = SVD()
svd.fit(trainset)

# Compute TF-IDF & Cosine Similarity (Content-Based Filtering)
tfidf_vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf_vectorizer.fit_transform(movies['overview'])
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

def get_streaming_platforms(movie_title):
    """
    Returns a JustWatch search URL where users can check where to stream the movie.
    """
    if not movie_title:  # ✅ Ensure title exists before generating link
        return ["Not Available"]
    
    movie_title_cleaned = movie_title.replace(" ", "+")
    return [f"https://www.justwatch.com/us/search?q={movie_title_cleaned}"]

# ✅ Route 1: Recommend Movies for a User (Collaborative Filtering)
@app.route('/recommend_user', methods=['POST'])
def recommend_user():
    data = request.get_json(force=True)
    user_id = int(data.get('user_id', -1))  # ✅ Get user_id safely

    if user_id not in ratings['userId'].values:
        return jsonify({"error": "User ID not found in dataset"}), 404  # ✅ Handle invalid user_id

    # Get movies rated by user
    rated_movies = ratings[ratings['userId'] == user_id]['movieId'].unique()

    if len(rated_movies) == 0:
        return jsonify({"error": "No ratings found for this user"}), 404  # ✅ Handle no rated movies

    # Get unrated movies
    all_movie_ids = movies['id'].unique()
    unrated_movie_ids = [movie_id for movie_id in all_movie_ids if movie_id not in rated_movies]

    if not unrated_movie_ids:
        return jsonify({"error": "No unrated movies found"}), 404  # ✅ Handle if all movies are rated

    # Predict ratings for unrated movies
    predictions = [svd.predict(user_id, movie_id) for movie_id in unrated_movie_ids]
    predictions.sort(key=lambda x: x.est, reverse=True)

    # Get top 5 recommended movies
    top_movie_ids = [pred.iid for pred in predictions[:5]]
    recommended_movies = movies[movies['id'].isin(top_movie_ids)][['title', 'id']].to_dict(orient='records')

    return jsonify(recommended_movies)


# ✅ Route 2: Recommend Similar Movies (Content-Based Filtering)
@app.route('/recommend_movie', methods=['POST'])
def recommend_movie():
    data = request.get_json(force=True)
    movie_title = data.get('movie_title', '').strip().lower()  # ✅ Get movie title safely

    # Convert all movie titles to lowercase for case-insensitive comparison
    movies['title_lower'] = movies['title'].str.lower()

    if movie_title not in movies['title_lower'].values:
        return jsonify({"error": "Movie not found"}), 404  # ✅ Handle missing movies

    # ✅ Get valid index safely
    idx_list = movies[movies['title_lower'] == movie_title].index.tolist()
    if not idx_list:
        return jsonify({"error": "Movie not found in database"}), 404
    
    idx = idx_list[0]

    # Get similar movies
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:6]

    # Get movie titles
    movie_indices = [i[0] for i in sim_scores]
    similar_movies = movies.iloc[movie_indices][['title', 'id']].to_dict(orient='records')

    for movie in similar_movies:
        movie["platforms"] = get_streaming_platforms(movie["title"])  
    
    return jsonify(similar_movies)


# ✅ Route 3: Search Suggestions (Auto-Suggestions)
@app.route('/search_suggestions')
def search_suggestions():
    query = request.args.get('query', '').strip().lower()  # ✅ Get query safely
    if not query:
        return jsonify([])  # ✅ Handle empty query

    suggestions = movies[movies['title'].str.lower().str.contains(query, na=False)]['title'].head(5).tolist()
    return jsonify(suggestions)

# ✅ Fixed: Homepage Route (Prevents 404 Error)
@app.route('/')
def home():
    return render_template('index.html')

# Start Flask Server
if __name__ == '__main__':
    app.run(debug=True, port=5001)
