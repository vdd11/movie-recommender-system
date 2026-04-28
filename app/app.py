
import streamlit as st
import pandas as pd
import random
from datetime import datetime

# ----------------------------------------------------------
# PAGE CONFIG
# ----------------------------------------------------------
st.set_page_config(
    page_title="🎬 Movie Recommender 4.1",
    layout="wide",
    page_icon="🍿"
)

# ----------------------------------------------------------
# SAMPLE DATA
# Replace later with real CSV / SQL / API data
# ----------------------------------------------------------
movies = pd.DataFrame({
    "movieId": [1,2,3,4,5,6,7,8],
    "title": [
        "Inception",
        "Interstellar",
        "John Wick",
        "Dune Part Two",
        "The Batman",
        "Avengers Endgame",
        "Top Gun Maverick",
        "The Dark Knight"
    ],
    "genre": [
        "Sci-Fi",
        "Sci-Fi",
        "Action",
        "Sci-Fi",
        "Action",
        "Action",
        "Action",
        "Action"
    ],
    "rating": [4.8,4.9,4.6,4.7,4.5,4.9,4.8,5.0]
})

# ----------------------------------------------------------
# STREAMING DATA (MOCK)
# Replace with TMDb API later
# ----------------------------------------------------------
streaming_map = {
    "Inception": ["Netflix", "Max"],
    "Interstellar": ["Paramount+", "Prime Video"],
    "John Wick": ["Peacock", "Prime Video"],
    "Dune Part Two": ["Max"],
    "The Batman": ["Max", "Hulu"],
    "Avengers Endgame": ["Disney+"],
    "Top Gun Maverick": ["Paramount+"],
    "The Dark Knight": ["Netflix", "Max"]
}

# ----------------------------------------------------------
# SOCIAL BUZZ
# ----------------------------------------------------------
def social_mentions(movie):
    return {
        "X": random.randint(1000,50000),
        "Reddit": random.randint(500,15000),
        "Facebook": random.randint(800,25000),
        "Instagram": random.randint(1000,60000)
    }

def sentiment():
    return random.randint(72,98)

def youtube_views():
    return random.randint(50000,900000)

# ----------------------------------------------------------
# HYPE SCORE
# ----------------------------------------------------------
def hype(movie):
    buzz = social_mentions(movie)
    total = sum(buzz.values())
    score = (
        (total / 150000)*0.45 +
        (sentiment()/100)*0.35 +
        (youtube_views()/900000)*0.20
    ) * 100
    return round(score,2)

# ----------------------------------------------------------
# RECOMMENDER
# ----------------------------------------------------------
def recommend(genre="Action", top_n=5):
    df = movies[movies["genre"] == genre].copy()
    df["Hype Score"] = df["title"].apply(hype)
    return df.sort_values("Hype Score", ascending=False).head(top_n)

# ----------------------------------------------------------
# STREAMING FUNCTION
# ----------------------------------------------------------
def get_streaming(movie):
    return streaming_map.get(movie, ["Unavailable"])

# ----------------------------------------------------------
# AI CHAT
# ----------------------------------------------------------
def ai_assistant(prompt):
    prompt = prompt.lower()

    if "space" in prompt or "smart" in prompt:
        return recommend("Sci-Fi")

    if "fight" in prompt or "action" in prompt:
        return recommend("Action")

    return movies.sample(5)

# ----------------------------------------------------------
# SIDEBAR
# ----------------------------------------------------------
st.sidebar.title("🎬 Movie Recommender 4.1")
username = st.sidebar.text_input("Username", "Brandon")
top_n = st.sidebar.slider("Recommendations", 3, 10, 5)

st.sidebar.markdown("---")
st.sidebar.success("Now Includes Streaming Availability 📺")

# ----------------------------------------------------------
# TABS
# ----------------------------------------------------------
tabs = st.tabs([
    "🏠 Home",
    "🎯 Recommendations",
    "📺 Streaming",
    "🔥 Trending",
    "📱 Social Buzz",
    "▶️ YouTube Reviews",
    "🤖 AI Assistant",
    "📊 Analytics",
    "👤 Profile"
])

# ----------------------------------------------------------
# HOME
# ----------------------------------------------------------
with tabs[0]:
    st.title("🎬 Movie Recommender 4.1")
    st.subheader("AI + Streaming + Social Intelligence")

    st.image(
        "https://images.unsplash.com/photo-1489599849927-2ee91cede3ba",
        use_column_width=True
    )

# ----------------------------------------------------------
# RECOMMENDATIONS
# ----------------------------------------------------------
with tabs[1]:
    st.header("🎯 Personalized Recommendations")

    genre = st.selectbox("Choose Genre", movies["genre"].unique())
    recs = recommend(genre, top_n)

    for _, row in recs.iterrows():
        with st.container():
            st.subheader(row["title"])
            st.write(f"Genre: {row['genre']}")
            st.write(f"Rating: {row['rating']}")
            st.write(f"Hype Score: {row['Hype Score']}")
            st.write("Streaming On:", ", ".join(get_streaming(row["title"])))
            st.markdown("---")

# ----------------------------------------------------------
# STREAMING TAB
# ----------------------------------------------------------
with tabs[2]:
    st.header("📺 Where To Watch")

    movie = st.selectbox("Choose Movie", movies["title"])

    platforms = get_streaming(movie)

    st.subheader(movie)
    for p in platforms:
        st.success(p)

# ----------------------------------------------------------
# TRENDING
# ----------------------------------------------------------
with tabs[3]:
    st.header("🔥 Most Hyped Movies")

    trend = movies.copy()
    trend["Hype Score"] = trend["title"].apply(hype)

    st.dataframe(
        trend.sort_values("Hype Score", ascending=False),
        use_container_width=True
    )

# ----------------------------------------------------------
# SOCIAL BUZZ
# ----------------------------------------------------------
with tabs[4]:
    st.header("📱 Social Media Mentions")

    movie = st.selectbox("Select Movie", movies["title"], key="buzz")

    buzz = social_mentions(movie)

    c1,c2,c3,c4 = st.columns(4)

    c1.metric("X", buzz["X"])
    c2.metric("Reddit", buzz["Reddit"])
    c3.metric("Facebook", buzz["Facebook"])
    c4.metric("Instagram", buzz["Instagram"])

    st.subheader("Audience Sentiment")
    st.progress(sentiment()/100)

# ----------------------------------------------------------
# YOUTUBE REVIEWS
# ----------------------------------------------------------
with tabs[5]:
    st.header("▶️ YouTube Reviews")

    movie = st.selectbox("Movie", movies["title"], key="yt")

    st.write(f"Top YouTube review for {movie}")
    st.video("https://www.youtube.com/watch?v=YoHD9XEInc0")

# ----------------------------------------------------------
# AI ASSISTANT
# ----------------------------------------------------------
with tabs[6]:
    st.header("🤖 AI Movie Assistant")

    prompt = st.text_input(
        "Ask AI",
        "Recommend exciting space movies"
    )

    if st.button("Ask Assistant"):
        results = ai_assistant(prompt)
        st.dataframe(results, use_container_width=True)

# ----------------------------------------------------------
# ANALYTICS
# ----------------------------------------------------------
with tabs[7]:
    st.header("📊 Platform Analytics")

    st.metric("Users", "12,450")
    st.metric("Movies", len(movies))
    st.metric("Avg Rating", round(movies["rating"].mean(),2))

    st.subheader("Ratings")
    st.bar_chart(movies.set_index("title")["rating"])

# ----------------------------------------------------------
# PROFILE
# ----------------------------------------------------------
with tabs[8]:
    st.header("👤 User Profile")

    st.write(f"Username: **{username}**")
    st.write("Favorite Genres: Action, Sci-Fi")
    st.write("Movies Rated: 31")
    st.write("Average Rating Given: 4.7")

# ----------------------------------------------------------
# FOOTER
# ----------------------------------------------------------
st.markdown("---")
st.caption(
    f"Movie Recommender 4.1 | Built with Streamlit | {datetime.now().year}"