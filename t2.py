import matplotlib.pyplot as plt

topic_counts = [
    42,
    21,
    22,
    29,
    17,
    21,
    25,
    20,
    10,
    20,
    33,
    21,
    15,
    45,
    27,
    13,
    11,
    17,
    30,
    22,
    22,
    20,
    16,
    15,
    17,
    25,
    34,
    15,
    13,
    16,
    12,
    27,
    27,
    21,
    16,
    24,
    16,
    31,
    22,
    32,
    21,
    27,
    20,
    10,
    32,
    22,
    17,
    26,
    20,
    24,
    22,
    24,
    23,
    24,
    27,
    22,
    7,
    23,
    23,
    7,
]

topic_counts = [
    45,
    36,
    39,
    40,
    36,
    34,
    32,
    41,
    35,
    30,
    51,
    51,
    34,
    35,
    41,
    34,
    38,
    38,
    33,
    39,
    51,
    50,
    60,
    36,
    41,
    39,
    43,
    40,
    45,
    30,
    36,
    30,
    40,
]
top5 = sorted(enumerate(topic_counts), key=lambda x: x[1], reverse=True)[:5]

for idx, value in top5:
    print(f"Index: {idx}, Value: {value}")


topics = list(range(len(topic_counts)))

plt.figure(figsize=(15, 6))
plt.bar(topics, topic_counts, color="skyblue")
plt.xlabel("Topic")
plt.ylabel("Number of Posts")
plt.title("Number of Posts per Topic")
plt.xticks(topics)
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.tight_layout()
plt.show()
