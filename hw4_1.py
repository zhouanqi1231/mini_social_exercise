import pandas as pd
from gensim.corpora import Dictionary
from gensim.models.ldamodel import LdaModel
from gensim.models.coherencemodel import CoherenceModel
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import nltk
import sqlite3


def main():
    # get all the posts
    db = sqlite3.connect("database.sqlite")
    cursor = db.cursor()
    query = "SELECT content FROM posts"
    cursor.execute(query)
    posts = cursor.fetchall()
    print("posts fetched")

    # Download necessary NLTK data, without these the below functions wouldn't work
    nltk.download("punkt")
    print("punkt loaded")
    nltk.download("stopwords")
    print("stopwords loaded")
    nltk.download("wordnet")
    print("wordnet loaded")

    # Get a basic stopword list
    stop_words = stopwords.words("english")

    # Add extra words to make our analysis even better
    stop_words.extend(["would", "best", "always", "amazing", "bought", "quick" "people", "new", "fun", "think", "know", "believe", "many", "thing", "need", "small", "even", "make", "love", "mean", "fact", "question", "time", "reason", "also", "could", "true", "well", "life", "said", "year", "going", "good", "really", "much", "want", "back", "look", "article", "host", "university", "reply", "thanks", "mail", "post", "please"])

    # this object will help us lemmatise words (i.e. get the word stem)
    lemmatizer = WordNetLemmatizer()

    # after the below for loop, we will transform each post into "bags of words" where each BOW is a set of words from one post
    bow_list = []

    for post in posts:
        text = post[0]
        # print(text)
        tokens = word_tokenize(text.lower())  # tokenise (i.e. get the words from the post)
        tokens = [lemmatizer.lemmatize(t) for t in tokens]  # lemmatise
        tokens = [t for t in tokens if len(t) > 2]  # filter out words with less than 3 letter s
        tokens = [t for t in tokens if t.isalpha() and t not in stop_words]  # filter out stopwords
        # if there's at least 1 word left for this post, append to list
        if len(tokens) > 0:
            bow_list.append(tokens)

    # Create dictionary and corpus
    dictionary = Dictionary(bow_list)
    # Filter words that appear less than 2 times or in more than 30% of posts
    dictionary.filter_extremes(no_below=2, no_above=0.3)
    corpus = [dictionary.doc2bow(tokens) for tokens in bow_list]

    # We don't know at this point how many topics there are.
    # Therefore, we try training the LDA algorithm at different topic counts,
    # and picking the one that results in the best coherence.
    optimal_coherence = -100
    optimal_lda = None
    optimal_k = 0
    for K in range(2, 200):

        # Train LDA model. We want to determine how we can best split the data into 4 topics
        lda = LdaModel(corpus, num_topics=K, id2word=dictionary, passes=10, random_state=2)

        # Now that the LDA model is done, let's see how good it is by computing its 'coherence score'
        coherence_model = CoherenceModel(model=lda, texts=bow_list, dictionary=dictionary, coherence="c_v")
        coherence_score = coherence_model.get_coherence()

        if coherence_score > optimal_coherence:
            print(f"Trained LDA with {K} topics. Average topic coherence (higher is better): {coherence_score} which is the best so far!")
            optimal_coherence = coherence_score
            optimal_lda = lda
            optimal_k = K
        else:
            print(f"Trained LDA with {K} topics. Average topic coherence (higher is better): {coherence_score} which is not very good.")

    # Okay, we tried many topic numbers and selected the best one. Let's see how our trained LDA model for the optimal number of topics performed.

    # First, to see the topics, print top 5 most representative words per topic
    print(f"These are the words most representative of each of the {optimal_k} topics:")
    for i, topic in optimal_lda.print_topics(num_words=5):
        print(f"Topic {i}: {topic}")

    # Then, let's determine how many posts we have for each topic
    # Count the dominant topic for each document
    topic_counts = [0] * optimal_k  # one counter per topic
    for bow in corpus:
        topic_dist = optimal_lda.get_document_topics(bow)  # list of (topic_id, probability)
        dominant_topic = max(topic_dist, key=lambda x: x[1])[0]  # find the top probability
        topic_counts[dominant_topic] += 1  # add 1 to the most probable topic's counter

    # Display the topic counts
    for i, count in enumerate(topic_counts):
        print(f"Topic {i}: {count} posts")


if __name__ == "__main__":
    main()
