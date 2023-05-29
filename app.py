from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer

import streamlit as st
import pickle
import requests

def get_similarity(new_df):
    cv = CountVectorizer(max_features=2500, stop_words='english')
    vectors = cv.fit_transform(new_df.tags).toarray()
    similarity = cosine_similarity(vectors)
    return similarity


def fetch_poster(movie_id):
    api_key_tmdb = 'd846098b38373843c6a233e1f8f16887'
    response = requests.get(f'https://api.themoviedb.org/3/movie/{movie_id}?api_key={api_key_tmdb}&language=en-us')
    data = response.json()
    return "https://image.tmdb.org/t/p/w500" + data['poster_path']

def recommend(movie_name):
    movie_index = movies[movies.title == movie_name].index[0]
    distances = similarity[movie_index]
    movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]

    recommended_movies = []
    for i in movies_list:
        recommend_id = movies.iloc[i[0]].movie_id
        recommend_name = movies.iloc[i[0]].title
        recommended_movies.append((recommend_name, fetch_poster(recommend_id)))
    return recommended_movies

movies = pickle.load(open('movies.pkl', 'rb'))
movies_list = movies['title'].values

similarity = get_similarity(movies)

st.title('Movie Recommender System')

selected_movie_name = st.selectbox('List of movies', movies_list)

if st.button('Recommend'):
    recommendation = recommend(selected_movie_name)
    cols = st.columns(len(recommendation))
    for i in range(len(recommendation)):
        with cols[i]:
            st.text(recommendation[i][0])
            st.image(recommendation[i][1])