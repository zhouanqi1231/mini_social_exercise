import pandas as pd
from gensim.corpora import Dictionary
from gensim.models.ldamodel import LdaModel
from gensim.models.coherencemodel import CoherenceModel
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import nltk
import sqlite3
import re


def split_camel_case(token):
    split_token = re.sub(r"([a-z])([A-Z])", r"\1 \2", token)
    return split_token.lower().split()


def main():
    # get all the posts
    db = sqlite3.connect("database.sqlite")
    cursor = db.cursor()
    query = "SELECT id, content FROM posts"
    cursor.execute(query)
    posts = cursor.fetchall()
    # print(posts)
    print("posts fetched")

    # for later querying the post content of a given id
    df_post = pd.DataFrame(posts, columns=["id", "content"])

    # for topic storing (id, content, topic)
    df_topic = pd.DataFrame(columns=["content", "topic"])

    # NLTK data
    nltk.download("punkt")
    nltk.download("stopwords")
    nltk.download("wordnet")

    # stopword
    stop_words = stopwords.words("english")
    stop_words.extend(["would", "best", "always", "amazing", "bought", "quick" "people", "new", "fun", "think", "know", "believe", "many", "thing", "need", "small", "even", "make", "love", "mean", "fact", "question", "time", "reason", "also", "could", "true", "well", "life", "said", "year", "going", "good", "really", "much", "want", "back", "look", "article", "host", "university", "reply", "thanks", "mail", "post", "please", "like", "wait", "every", "day", "last", "might", "sometimes", "today", "anyone", "else", "get", "doe", "way", "another"])

    lemmatizer = WordNetLemmatizer()

    # BOW: [[post_1_words],[post_2_words],[post_3_words]]
    bow_list = []
    bow_list_content_id = []  # store the post_id so the original content can be found later

    for post in posts:
        text = post[1]
        # print(text)
        tokens = word_tokenize(text)

        # remove # and process camel case
        new_tokens = []
        for t in tokens:
            if t.startswith("#"):
                t = t[1:]
                new_tokens.extend(split_camel_case(t))
            else:
                new_tokens.append(t.lower())

        new_tokens = [lemmatizer.lemmatize(t) for t in new_tokens]  # lemmatise
        new_tokens = [t for t in new_tokens if len(t) > 2]  # filter out words with less than 3 letter s
        new_tokens = [t for t in new_tokens if t.isalpha() and t not in stop_words]  # filter out stopwords
        # if there's at least 1 word left for this post, append to list
        if len(new_tokens) > 0:
            bow_list.append(new_tokens)
            current_post_id = post[0]
            bow_list_content_id.append(current_post_id)

    # Create dictionary and corpus
    dictionary = Dictionary(bow_list)
    # Filter words that appear less than 2 times or in more than 30% of posts
    dictionary.filter_extremes(no_below=2, no_above=0.3)
    corpus = [dictionary.doc2bow(tokens) for tokens in bow_list]
    # print(corpus)

    # finding K
    optimal_coherence = -100
    optimal_lda = None
    optimal_k = 0
    for i in range(120, 121):
        K = 33

        # train model
        lda = LdaModel(corpus, num_topics=K, id2word=dictionary, passes=10, random_state=2)  # , alpha=1 / 80, eta=0.5)

        # evaluate the model using coherence score
        coherence_model = CoherenceModel(model=lda, texts=bow_list, dictionary=dictionary, coherence="c_v")
        coherence_score = coherence_model.get_coherence()

        if coherence_score > optimal_coherence:
            print(f"Trained LDA with {K} topics. Average topic coherence (higher is better): {coherence_score} which is the best so far!")
            optimal_coherence = coherence_score
            optimal_lda = lda
            optimal_k = K
        else:
            print(f"Trained LDA with {K} topics. Average topic coherence (higher is better): {coherence_score} which is not very good.")

    # for every topic, top 5 most representative words per topic
    print(f"These are the words most representative of each of the {optimal_k} topics:")
    for i, topic in optimal_lda.print_topics(num_topics=optimal_k, num_words=10):
        print(f"Topic {i}: {topic}")

    # Then, let's determine how many posts we have for each topic
    # Count the dominant topic for each document
    topic_counts = [0] * optimal_k  # one counter per topic
    for i in range(len(corpus)):  # so the index is i
        topic_dist = optimal_lda.get_document_topics(corpus[i])  # list of (topic_id, probability)
        if not topic_dist:
            continue
        dominant_topic = max(topic_dist, key=lambda x: x[1])[0]  # find the top probability
        topic_counts[dominant_topic] += 1  # add 1 to the most probable topic's counter

        # add this to the data frame to store this topic classification later
        post_id = bow_list_content_id[i]
        original_content = df_post[df_post["id"] == post_id]["content"].values[0]
        # df_topic: (content, topic_id)
        df_topic.loc[len(df_topic)] = [original_content, dominant_topic]

    df_topic.to_csv("post_topic.csv", index=False, encoding="utf-8")

    # Display the topic counts
    topic_count = []
    for i, count in enumerate(topic_counts):
        print(f"Topic {i}: {count} posts")
        topic_count.append(count)

    top10 = sorted(enumerate(topic_counts), key=lambda x: x[1], reverse=True)[:10]

    print("\nTop 10 topics with the most posts and their content:")

    for idx, value in top10:
        print(f"\nTopic {idx}: {value} posts")

        for i, topic_words in optimal_lda.print_topics(num_topics=optimal_k, num_words=10):
            if i == idx:
                print(f"Top words: {topic_words}")
                break


if __name__ == "__main__":
    main()
