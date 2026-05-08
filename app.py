import os, warnings
import pandas as pd
import numpy as np
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse.linalg import svds

warnings.filterwarnings("ignore")

st.set_page_config(page_title="Movie Recommender", page_icon="🎬", layout="wide")
st.title("🎬 Hybrid Movie Recommendation System")
st.caption("Content-Based + Collaborative Filtering (SVD)")

@st.cache_data
def load_data():
    BASE    = os.path.dirname(os.path.abspath(__file__))
    movies  = pd.read_csv(os.path.join(BASE, "movies.csv"))
    ratings = pd.read_csv(os.path.join(BASE, "ratings.csv"))
    movies.dropna(inplace=True)
    ratings.dropna(inplace=True)
    movies.drop_duplicates(subset="movieId", inplace=True)
    ratings.drop_duplicates(inplace=True)
    movies  = movies[movies["genres"] != "(no genres listed)"]
    if "timestamp" in ratings.columns:
        ratings.drop(columns=["timestamp"], inplace=True)
    valid   = set(movies["movieId"])
    ratings = ratings[ratings["movieId"].isin(valid)].reset_index(drop=True)
    return movies, ratings

@st.cache_resource
def build_cb_model(movies):
    mc = movies.copy().reset_index(drop=True)
    mc["features"] = (mc["genres"].str.replace("|", " ", regex=False)
                      + " " + mc["title"].str.replace(r"\(\d{4}\)", "", regex=True).str.strip())
    tfidf = TfidfVectorizer(token_pattern=r"[A-Za-z\-]+")
    mat   = tfidf.fit_transform(mc["features"])
    sim   = cosine_similarity(mat, mat)
    idx   = pd.Series(mc.index, index=mc["movieId"])
    return mc, sim, idx

@st.cache_resource
def build_cf_model(ratings, movies):
    # Build user-item matrix
    user_movie = ratings.pivot_table(index="userId", columns="movieId", values="rating").fillna(0)
    mat = user_movie.values
    # Normalize
    mat_mean = mat.mean(axis=1, keepdims=True)
    mat_norm = mat - mat_mean
    # SVD
    k = min(50, min(mat_norm.shape) - 1)
    U, sigma, Vt = svds(mat_norm, k=k)
    sigma_diag = np.diag(sigma)
    predicted = np.dot(np.dot(U, sigma_diag), Vt) + mat_mean
    pred_df = pd.DataFrame(predicted, index=user_movie.index, columns=user_movie.columns)
    return pred_df, user_movie

with st.spinner("Loading data and training models..."):
    movies, ratings = load_data()
    mc, cos_sim, idx_map = build_cb_model(movies)
    pred_df, user_movie = build_cf_model(ratings, movies)

all_movie_ids = set(movies["movieId"])

def cb_recs(movie_id, top_n=10):
    if movie_id not in idx_map:
        return pd.DataFrame()
    idx    = idx_map[movie_id]
    scores = sorted(enumerate(cos_sim[idx]), key=lambda x: x[1], reverse=True)[1:top_n+1]
    res    = mc.iloc[[i[0] for i in scores]][["movieId","title","genres"]].copy()
    res["cb_score"] = [i[1] for i in scores]
    res["cb_score"] /= (res["cb_score"].max() + 1e-9)
    return res.reset_index(drop=True)

def cf_recs(user_id, top_n=10):
    if user_id not in pred_df.index:
        return pd.DataFrame()
    user_row  = pred_df.loc[user_id]
    seen      = set(user_movie.loc[user_id][user_movie.loc[user_id] > 0].index) if user_id in user_movie.index else set()
    unseen    = [(mid, user_row[mid]) for mid in user_row.index if mid not in seen]
    unseen.sort(key=lambda x: x[1], reverse=True)
    res = pd.DataFrame(unseen[:top_n], columns=["movieId","cf_score"])
    res = res.merge(movies[["movieId","title","genres"]], on="movieId")
    mn, mx = res["cf_score"].min(), res["cf_score"].max()
    res["cf_score"] = (res["cf_score"] - mn) / (mx - mn + 1e-9)
    return res.reset_index(drop=True)

def hybrid_recs(user_id, movie_id, alpha=0.5, top_n=10):
    cb = cb_recs(movie_id, top_n=50)
    cf = cf_recs(user_id,  top_n=50)
    if cb.empty: return cf.head(top_n)
    if cf.empty: return cb.head(top_n)
    merged = pd.merge(
        cb[["movieId","title","genres","cb_score"]],
        cf[["movieId","cf_score"]],
        on="movieId", how="outer"
    ).fillna(0)
    missing = merged["title"] == 0
    if missing.any():
        merged.loc[missing,"title"] = merged.loc[missing,"movieId"].map(
            movies.set_index("movieId")["title"])
    merged["hybrid_score"] = alpha * merged["cf_score"] + (1-alpha) * merged["cb_score"]
    return (merged.dropna(subset=["title"])
            .sort_values("hybrid_score", ascending=False)
            .head(top_n).reset_index(drop=True))

# Sidebar
st.sidebar.header("⚙️ Settings")
mode  = st.sidebar.radio("Recommendation Mode", ["Content-Based", "Collaborative", "Hybrid"])
alpha = st.sidebar.slider("Hybrid Alpha (CF weight)", 0.0, 1.0, 0.5, 0.1)
top_n = st.sidebar.slider("Number of Recommendations", 5, 20, 10)

# Inputs
col1, col2 = st.columns(2)
with col1:
    movie_title = st.selectbox("🎥 Select a seed movie", options=sorted(movies["title"].tolist()))
    movie_id    = int(movies[movies["title"] == movie_title]["movieId"].iloc[0])
with col2:
    user_id = st.number_input("👤 Enter User ID", min_value=1, max_value=610, value=1, step=1)

if st.button("🚀 Get Recommendations"):
    with st.spinner("Computing recommendations..."):
        if mode == "Content-Based":
            recs = cb_recs(movie_id, top_n)
            recs = recs.rename(columns={"cb_score": "score"})
        elif mode == "Collaborative":
            recs = cf_recs(user_id, top_n)
            recs = recs.rename(columns={"cf_score": "score"})
        else:
            recs = hybrid_recs(user_id, movie_id, alpha=alpha, top_n=top_n)
            recs = recs.rename(columns={"hybrid_score": "score"})

    st.subheader(f"Top {top_n} Recommendations ({mode})")
    for _, row in recs.iterrows():
        with st.container():
            c1, c2, c3 = st.columns([3, 2, 1])
            c1.write(f"**{row['title']}**")
            c2.caption(row["genres"])
            c3.metric("", f"{row['score']:.2f}")
            st.divider()
