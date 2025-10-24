import pandas as pd
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
import sqlite3
import matplotlib.pyplot as plt

df = pd.read_csv("post_topic.csv")

# Download the VADER lexicon
nltk.download("vader_lexicon")

# convert query results into df

# Initialize the VADER sentiment analyser
sia = SentimentIntensityAnalyzer()

# Calculate sentiment scores for each content
df["sentiment_score"] = df["content"].apply(lambda content: sia.polarity_scores(content)["compound"])


topic22_posts = df[df["topic"] == 22]
topic10_posts = df[df["topic"] == 10]
topic11_posts = df[df["topic"] == 11]
topic20_posts = df[df["topic"] == 20]
topic21_posts = df[df["topic"] == 21]
topic0_posts = df[df["topic"] == 0]
topic28_posts = df[df["topic"] == 28]
topic26_posts = df[df["topic"] == 26]
topic7_posts = df[df["topic"] == 7]
topic14_posts = df[df["topic"] == 14]

topic22_sentiments = topic22_posts["sentiment_score"]
topic10_sentiments = topic10_posts["sentiment_score"]
topic11_sentiments = topic11_posts["sentiment_score"]
topic20_sentiments = topic20_posts["sentiment_score"]
topic21_sentiments = topic21_posts["sentiment_score"]
topic0_sentiments = topic0_posts["sentiment_score"]
topic28_sentiments = topic28_posts["sentiment_score"]
topic26_sentiments = topic26_posts["sentiment_score"]
topic7_sentiments = topic7_posts["sentiment_score"]
topic14_sentiments = topic14_posts["sentiment_score"]


print("topic22_sentiments: " + str(topic22_sentiments.mean()))
print("topic22_sentiments: " + str(topic22_sentiments.median()))
print("topic10_sentiments: " + str(topic10_sentiments.mean()))
print("topic10_sentiments: " + str(topic10_sentiments.median()))
print("topic11_sentiments: " + str(topic11_sentiments.mean()))
print("topic11_sentiments: " + str(topic11_sentiments.median()))
print("topic20_sentiments: " + str(topic20_sentiments.mean()))
print("topic20_sentiments: " + str(topic20_sentiments.median()))
print("topic21_sentiments: " + str(topic21_sentiments.mean()))
print("topic21_sentiments: " + str(topic21_sentiments.median()))
print("topic0_sentiments: " + str(topic0_sentiments.mean()))
print("topic0_sentiments: " + str(topic0_sentiments.median()))
print("topic28_sentiments: " + str(topic28_sentiments.mean()))
print("topic28_sentiments: " + str(topic28_sentiments.median()))
print("topic26_sentiments: " + str(topic26_sentiments.mean()))
print("topic26_sentiments: " + str(topic26_sentiments.median()))
print("topic7_sentiments: " + str(topic7_sentiments.mean()))
print("topic7_sentiments: " + str(topic7_sentiments.median()))
print("topic14_sentiments: " + str(topic14_sentiments.mean()))
print("topic14_sentiments: " + str(topic14_sentiments.median()))

# scatter diagram
plt.figure(figsize=(6, 4))
plt.scatter(topic22_posts.index, topic22_sentiments, s=15, alpha=0.5, edgecolors="none")


plt.axhline(0, color="grey", linestyle="--", linewidth=0.8)
plt.title("scatter plot topic22_sentiments")
plt.xlabel("index")
plt.ylabel("sentiment score")
plt.grid(True, linestyle="--", linewidth=0.5)

plt.savefig("topic22_sentiments.png", dpi=300)
plt.show()
# scatter diagram
plt.figure(figsize=(6, 4))
plt.scatter(topic10_posts.index, topic10_sentiments, s=15, alpha=0.5, edgecolors="none")


plt.axhline(0, color="grey", linestyle="--", linewidth=0.8)
plt.title("scatter plot topic10_sentiments")
plt.xlabel("index")
plt.ylabel("sentiment score")
plt.grid(True, linestyle="--", linewidth=0.5)

plt.savefig("topic10_sentiments.png", dpi=300)
plt.show()
# scatter diagram
plt.figure(figsize=(6, 4))
plt.scatter(topic11_posts.index, topic11_sentiments, s=15, alpha=0.5, edgecolors="none")


plt.axhline(0, color="grey", linestyle="--", linewidth=0.8)
plt.title("scatter plot topic11_sentiments")
plt.xlabel("index")
plt.ylabel("sentiment score")
plt.grid(True, linestyle="--", linewidth=0.5)

plt.savefig("topic11_sentiments.png", dpi=300)
plt.show()
# scatter diagram
plt.figure(figsize=(6, 4))
plt.scatter(topic20_posts.index, topic20_sentiments, s=15, alpha=0.5, edgecolors="none")


plt.axhline(0, color="grey", linestyle="--", linewidth=0.8)
plt.title("scatter plot topic20_sentiments")
plt.xlabel("index")
plt.ylabel("sentiment score")
plt.grid(True, linestyle="--", linewidth=0.5)

plt.savefig("topic20_sentiments.png", dpi=300)
plt.show()
# scatter diagram
plt.figure(figsize=(6, 4))
plt.scatter(topic21_posts.index, topic21_sentiments, s=15, alpha=0.5, edgecolors="none")


plt.axhline(0, color="grey", linestyle="--", linewidth=0.8)
plt.title("scatter plot topic21_sentiments")
plt.xlabel("index")
plt.ylabel("sentiment score")
plt.grid(True, linestyle="--", linewidth=0.5)

plt.savefig("topic21_sentiments.png", dpi=300)
plt.show()
# scatter diagram
plt.figure(figsize=(6, 4))
plt.scatter(topic0_posts.index, topic0_sentiments, s=15, alpha=0.5, edgecolors="none")


plt.axhline(0, color="grey", linestyle="--", linewidth=0.8)
plt.title("scatter plot topic0_sentiments")
plt.xlabel("index")
plt.ylabel("sentiment score")
plt.grid(True, linestyle="--", linewidth=0.5)

plt.savefig("topic0_sentiments.png", dpi=300)
plt.show()
# scatter diagram
plt.figure(figsize=(6, 4))
plt.scatter(topic28_posts.index, topic28_sentiments, s=15, alpha=0.5, edgecolors="none")


plt.axhline(0, color="grey", linestyle="--", linewidth=0.8)
plt.title("scatter plot topic28_sentiments")
plt.xlabel("index")
plt.ylabel("sentiment score")
plt.grid(True, linestyle="--", linewidth=0.5)

plt.savefig("topic28_sentiments.png", dpi=300)
plt.show()
# scatter diagram
plt.figure(figsize=(6, 4))
plt.scatter(topic26_posts.index, topic26_sentiments, s=15, alpha=0.5, edgecolors="none")


plt.axhline(0, color="grey", linestyle="--", linewidth=0.8)
plt.title("scatter plot topic26_sentiments")
plt.xlabel("index")
plt.ylabel("sentiment score")
plt.grid(True, linestyle="--", linewidth=0.5)

plt.savefig("topic26_sentiments.png", dpi=300)
plt.show()
# scatter diagram
plt.figure(figsize=(6, 4))
plt.scatter(topic7_posts.index, topic7_sentiments, s=15, alpha=0.5, edgecolors="none")


plt.axhline(0, color="grey", linestyle="--", linewidth=0.8)
plt.title("scatter plot topic7_sentiments")
plt.xlabel("index")
plt.ylabel("sentiment score")
plt.grid(True, linestyle="--", linewidth=0.5)

plt.savefig("topic7_sentiments.png", dpi=300)
plt.show()
# scatter diagram
plt.figure(figsize=(6, 4))
plt.scatter(topic14_posts.index, topic14_sentiments, s=15, alpha=0.5, edgecolors="none")


plt.axhline(0, color="grey", linestyle="--", linewidth=0.8)
plt.title("scatter plot topic14_sentiments")
plt.xlabel("index")
plt.ylabel("sentiment score")
plt.grid(True, linestyle="--", linewidth=0.5)

plt.savefig("topic14_sentiments.png", dpi=300)
plt.show()
