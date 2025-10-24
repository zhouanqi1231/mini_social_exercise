import pandas as pd
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
import sqlite3
import matplotlib.pyplot as plt

# get all the posts and the comments
db = sqlite3.connect("database.sqlite")
cursor = db.cursor()

post_query = "SELECT content FROM posts"
cursor.execute(post_query)
posts = cursor.fetchall()

comment_query = "SELECT content FROM comments"
cursor.execute(comment_query)
comments = cursor.fetchall()

df = pd.DataFrame(posts + comments, columns=["content"])

# Download the VADER lexicon
nltk.download("vader_lexicon")

# convert query results into df

# Initialize the VADER sentiment analyser
sia = SentimentIntensityAnalyzer()

# Calculate sentiment scores for each review
df["sentiment_score"] = df["content"].apply(lambda review: sia.polarity_scores(review)["compound"])

print(df["sentiment_score"].mean())
print(df["sentiment_score"].median())

# scatter diagram
plt.figure(figsize=(6, 4))
plt.scatter(df.index, df["sentiment_score"], s=15, alpha=0.5, edgecolors="none")


plt.axhline(0, color="grey", linestyle="--", linewidth=0.8)
plt.title("scatter diagram of sentiment scores")
plt.xlabel("index")
plt.ylabel("sentiment score")
plt.grid(True, linestyle="--", linewidth=0.5)

plt.show()
