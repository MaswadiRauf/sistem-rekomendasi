from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
import os
import traceback
import joblib
from huggingface_hub import hf_hub_download
import requests
import random
import json
import numpy as np

app = Flask(__name__)
REPO_ID = "maswadi/hybrid-recommender-model"

# Lazy loading variables
movie_svdpp = None
movie_knn = None

lastfm_svdpp = None
lastfm_knn = None

def load_hf_file(filename):
    return hf_hub_download(
        repo_id=REPO_ID,
        filename=filename,
        token=os.getenv("HF_TOKEN")
    )
# LOAD MODEL

def get_movie_svdpp():
    global movie_svdpp

    if movie_svdpp is None:
        movie_svdpp = joblib.load(
            load_hf_file("movie_svdpp_tuned.pkl")
        )

    return movie_svdpp


def get_movie_knn():
    global movie_knn

    if movie_knn is None:
        movie_knn = joblib.load(
            load_hf_file("movie_knn.pkl")
        )

    return movie_knn

# HYBRID ALPHA

with open(
    load_hf_file("movie_hybrid_tuned_alpha.json"),
    "r"
) as f:
    hybrid_config = json.load(f)

ALPHA = hybrid_config["alpha"]

# LOAD LASTFM MODEL

def get_lastfm_svdpp():
    global lastfm_svdpp

    if lastfm_svdpp is None:
        lastfm_svdpp = joblib.load(
            load_hf_file("lastfm_svdpp_tuned.pkl")
        )

    return lastfm_svdpp


def get_lastfm_knn():
    global lastfm_knn

    if lastfm_knn is None:
        lastfm_knn = joblib.load(
            load_hf_file("lastfm_knn.pkl")
        )

    return lastfm_knn

with open(
    load_hf_file("lastfm_hybrid_tuned_alpha.json"),
    "r"
) as f:
    lastfm_config = json.load(f)

LASTFM_ALPHA = lastfm_config["alpha"]

# LOAD MOVIELENS DATA

movies_df = pd.read_csv(
    "data/movies.dat",
    sep="::",
    engine="python",
    header=None,
    encoding="latin-1"
)

movies_df.columns = [
    "movieId",
    "title",
    "genres"
]

# FORMAT MOVIE ID

movies_df["movieId"] = (
    movies_df["movieId"]
    .astype(str)
    .str.strip()
)

# LOAD MOVIES POSTER CSV

movies_poster_df = pd.read_csv(
    "data/movies.csv"
)

movies_poster_df["movie_id"] = (
    movies_poster_df["movie_id"]
    .astype(str)
    .str.strip()
)

# MERGE POSTER DATA

movies_df = movies_df.merge(

    movies_poster_df[
        ["movie_id", "poster_url"]
    ],

    left_on="movieId",
    right_on="movie_id",
    how="left"

)

print(
    movies_df[
        ["movieId", "title", "poster_url"]
    ].head()
)

# LOAD RATINGS CSV

ratings_df = pd.read_csv(

    "data/ratings.dat",

    sep="::",

    engine="python",

    names=[
        "userId",
        "movieId",
        "rating",
        "timestamp"
    ]

)

# MOVIE RATING SUMMARY

movie_rating_summary = ratings_df.groupby(
    "movieId"
).agg({

    "rating": ["mean", "count"]

}).reset_index()

movie_rating_summary.columns = [

    "movieId",

    "avg_rating",

    "rating_count"

]

movie_rating_summary["movieId"] = (
    movie_rating_summary["movieId"]
    .astype(str)
)

movie_rating_summary["avg_rating"] = (
    movie_rating_summary["avg_rating"]
    .round(1)
)

# MERGE RATING KE MOVIES

movies_df = movies_df.merge(

    movie_rating_summary,

    on="movieId",

    how="left"

)

# LOAD LAST.FM DATA

artists_df = pd.read_csv(
    "data/artists.dat",
    sep="\t",
    encoding="utf-8"
)

artists_df = artists_df.rename(
    columns={
        "id": "artistId",
        "name": "artistName"
    }
)

artists_df["artistId"] = artists_df[
    "artistId"
].astype(str)

# LOAD USER_ARTISTS

lastfm_df = pd.read_csv(
    "data/user_artists.dat",
    sep="\t"
)

lastfm_df["log_weight"] = np.log(
    lastfm_df["weight"]
)

log_min = lastfm_df["log_weight"].min()
log_max = lastfm_df["log_weight"].max()

lastfm_df["rating"] = (

    1 +

    (
        (
            lastfm_df["log_weight"]
            - log_min
        )
        /
        (
            log_max
            - log_min
        )
    ) * 4

)

lastfm_df["rating"] = (
    lastfm_df["rating"]
    .round(4)
)

print(lastfm_df.head())
print(lastfm_df.columns)

artist_threshold = (
    lastfm_df["rating"].median()
)

print(
    "LASTFM THRESHOLD:",
    artist_threshold
)

# HOME

@app.route("/")
def home():

    genres_list = [

        "Action",
        "Comedy",
        "Drama",
        "Adventure",
        "Animation",
        "Sci-Fi",
        "Thriller",
        "Romance",
        "Horror"

    ]

    genre_sections = {}

    for genre in genres_list:

        genre_sections[genre] = movies_df[
            movies_df["genres"].str.contains(
                genre,
                na=False
            )
        ].head(10).to_dict("records")

    return render_template(

        "index.html",

        genre_sections=genre_sections,

        user_id=1

    )

# HYBRID RECOMMENDATION

def recommend_movies(movie_id, n=10):

    movie_id_int = int(movie_id)


    # USER YANG MENYUKAI FILM TERSEBUT


    target_users = ratings_df[

        (ratings_df["movieId"] == movie_id_int)
        &
        (ratings_df["rating"] >= 4)

    ]["userId"].unique()

    if len(target_users) == 0:
        return []


# FILM LAIN YANG DISUKAI USER TERSEBUT


    candidate_movies = ratings_df[
        ratings_df["userId"].isin(
            target_users
        )
    ]

    # hanya film yang benar-benar disukai
    candidate_movies = candidate_movies[
        candidate_movies["rating"] >= 4
    ]

    candidate_movies = candidate_movies[
        candidate_movies["movieId"]
        != movie_id_int
    ]


# PILIH KANDIDAT TERBAIK


    candidate_stats = (
        candidate_movies
        .groupby("movieId")
        .agg({
            "rating": ["count", "mean"]
        })
    )

    candidate_stats.columns = [
        "count",
        "mean_rating"
    ]

    candidate_stats["candidate_score"] = (
        candidate_stats["count"]
        *
        candidate_stats["mean_rating"]
    )

    top_candidates = (
        candidate_stats
        .sort_values(
            "candidate_score",
            ascending=False
        )
        .head(50)
        .index
        .tolist()
    )

    print(
        "Target Users:",
        len(target_users)
    )

    print(
        "Candidate Artists:",
        len(top_candidates)
    )


# HYBRID SVD++ KNN


    recommendations = []

    for candidate_movie in top_candidates:

        hybrid_scores = []

        sample_users = (
            target_users[:20]
            if len(target_users) > 20
            else target_users
        )

        for user_id in sample_users:

            try:

                pred_svdpp = (
                    get_movie_svdpp().predict(
                        str(user_id),
                        str(candidate_movie)
                    ).est
                )
                print("SVD:", pred_svdpp)

                pred_knn = (
                    get_movie_knn().predict(
                        str(user_id),
                        str(candidate_movie)
                    ).est
                )
                print("KNN:", pred_knn)

                hybrid_score = (

                    ALPHA * pred_svdpp
                    +
                    (1 - ALPHA)
                    * pred_knn

                )

                hybrid_scores.append(
                    hybrid_score
                )
                print("Hybrid:", hybrid_score)

            except Exception:
                traceback.print_exc()

        if len(hybrid_scores) == 0:
            print(f"Tidak ada hybrid score untuk movie {candidate_movie}")
            continue

        final_score = (
            sum(hybrid_scores)
            /
            len(hybrid_scores)
        )

        movie_data = movies_df[

            movies_df["movieId"]
            .astype(str)

            ==

            str(candidate_movie)
        ]
        
        print(candidate_movie, len(movie_data))
        

        if len(movie_data) == 0:
            continue

        recommendations.append({

            "movieId":
            int(candidate_movie),

            "title":
            movie_data.iloc[0]["title"],

            "genres":
            movie_data.iloc[0]["genres"],

            "poster":
            movie_data.iloc[0]["poster_url"],

            "avg_rating":
            movie_data.iloc[0]["avg_rating"],

            "rating_count":
            movie_data.iloc[0]["rating_count"],

            "score":
            round(final_score, 4)

        })

    recommendations = sorted(

        recommendations,

        key=lambda x:
        x["score"],

        reverse=True

    )
    
    print("TOTAL RECOMMENDATIONS:", len(recommendations))

    return recommendations[:n]

# MOVIE DETAIL

@app.route("/movie/<movie_id>")
def movie_detail(movie_id):

    top_n = request.args.get(
        "top",
        default=10,
        type=int
    )

    selected_movie = movies_df[
        movies_df["movieId"].astype(str)
        == str(movie_id)
    ].iloc[0]

    recommendations = recommend_movies(
    movie_id,
    n=top_n
    )

    return render_template(
        "movie_detail.html",
        movie=selected_movie,
        recommendations=recommendations,
        top_n=top_n,
    )

# SEARCH MOVIE

@app.route("/search")
def search_movie():

    from flask import request

    query = request.args.get("query")

    if query:

        query = query.strip()

        filtered_movies = movies_df[
            movies_df["title"].str.lower().str.contains(
                query.lower(),
                na=False
            )
        ]

    else:
        filtered_movies = movies_df.head(20)

    movies = filtered_movies.to_dict("records")

    return render_template(
        "movie_search.html",
        movies=filtered_movies.to_dict("records"),
        query=query
    )

# MUSIC RECOMMENDATION FUNCTION

def recommend_music(artist_id, n=10):

    artist_id_int = int(artist_id)

    target_users = lastfm_df[
        (lastfm_df["artistID"] == artist_id_int)
        &
        (lastfm_df["rating"] >= artist_threshold)
    ]["userID"].unique()
    print("TARGET USERS:", len(target_users))

    if len(target_users) == 0:
        return []

    candidate_artists = lastfm_df[

        lastfm_df["userID"].isin(
            target_users
        )

    ]

    candidate_artists = candidate_artists[

        candidate_artists["rating"] >= artist_threshold

    ]

    candidate_artists = candidate_artists[

        candidate_artists["artistID"]
        != artist_id_int

    ]

    candidate_stats = (

        candidate_artists

        .groupby("artistID")

        .agg({

            "rating": ["count", "mean"]

        })

    )

    candidate_stats.columns = [

        "count",

        "mean_rating"

    ]

    candidate_stats["candidate_score"] = (
        candidate_stats["count"]
        *
        candidate_stats["mean_rating"]
    )

    top_candidates = (

        candidate_stats

        .sort_values(
            "candidate_score",
            ascending=False
        )

        .head(50)

        .index

        .tolist()

    )

    print(
        candidate_stats
        .sort_values(
            "candidate_score",
            ascending=False
        )
        .head(20)
    )

    print(
        "Target Users:",
        len(target_users)
    )

    print(
        "Candidate Artists:",
        len(top_candidates)
    )

    recommendations = []

    for candidate_artist in top_candidates:

        hybrid_scores = []

        sample_users = (
            target_users[:20]
            if len(target_users) > 20
            else target_users
        )

        for user_id in sample_users:

            try:

                pred_svdpp = get_lastfm_svdpp().predict(
                    user_id,
                    candidate_artist
                ).est

                pred_knn = get_lastfm_knn().predict(
                    user_id,
                    candidate_artist
                ).est

                hybrid_score = (

                    LASTFM_ALPHA
                    * pred_svdpp
                    +
                    (1 - LASTFM_ALPHA)
                    * pred_knn

                )

                hybrid_scores.append(
                    hybrid_score
                )

            except Exception:
                traceback.print_exc()

        if len(hybrid_scores) == 0:
            continue

        final_score = (
            sum(hybrid_scores)
            /
            len(hybrid_scores)
        )

        artist_data = artists_df[

            artists_df["artistId"]
            .astype(str)

            ==

            str(candidate_artist)

        ]

        if len(artist_data) == 0:
            continue

        recommendations.append({

            "artistId":
            candidate_artist,

            "artistName":
            artist_data.iloc[0]["artistName"],

            "score":
            round(final_score, 4)

        })

    recommendations = sorted(

        recommendations,

        key=lambda x:
        x["score"],

        reverse=True

    )
    print("TOTAL RECOMMENDATIONS:",
      len(recommendations))

    return recommendations[:n]

# MUSIC HOME

@app.route("/music")
def music_home():

    musics = artists_df.head(20).to_dict("records")

    return render_template(
        "music_index.html",
        musics=musics
    )

# SEARCH MUSIC

@app.route("/music/search")
def search_music():

    query = request.args.get("query")

    results = artists_df[
        artists_df["artistName"].str.contains(
            query,
            case=False,
            na=False
        )
    ].head(20)

    return render_template(
        "music_search.html",
        musics=results.to_dict("records"),
        query=query
    )

# MUSIC DETAIL

@app.route("/music/<artist_id>")
def music_detail(artist_id):

    top_n = int(
        request.args.get(
            "top",
            10
        )
    )

    selected_music = artists_df[
        artists_df["artistId"].astype(str)
        == str(artist_id)
    ].iloc[0]

    recommendations = recommend_music(
        artist_id,
        n=top_n
    )

    return render_template(
        "music_detail.html",
        music=selected_music,
        recommendations=recommendations,
        top_n=top_n
    )

# ABOUT

@app.route("/about")
def about():

    return render_template(
        "about.html"
    )

# METRICS PAGE

@app.route("/metrics")
def metrics():

    metrics_data = {

        "movielens": {

            "dataset": "MovieLens",

            "model": "Hybrid SVD++-KNN",

            "rmse": 0.8594,

            "mae": 0.6747,

            "alpha": 0.80

        },

        "lastfm": {

            "dataset": "Last.fm",

            "model": "Hybrid SVD++-KNN",

            "rmse": 0.2632,

            "mae": 0.1955,

            "alpha": 0.90

        }

    }

    return render_template(
        "metrics.html",
        metrics=metrics_data
    )

if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=5000
    )
    
    
# kalo mau lokal
# if __name__ == "__main__":
#     app.run(debug=True)