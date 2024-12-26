############################################
# -------------------- Developed by Partha Pratim Ray -------------------- #
# Contact: parthapratimray1986@gmail.com
# GitHub: https://github.com/ParthaPRay/
##################### GRADIO INTERFACE #####################

##### Sample Inputs

# text_input[]

# OpenAI, based in San Francisco, has developed the GPT model, which is widely used for natural language processing tasks.  The company aims to make artificial intelligence accessible and useful to people worldwide. In 2023, they released GPT-4.

########

# docs_input[]

# OpenAI is an artificial intelligence research lab that focuses on developing safe AI. The lab is well-known for the GPT series of models. GPT-4 is the latest release by OpenAI, showcasing advanced natural language processing capabilities. Artificial intelligence tools like GPT have become integral for tasks like summarization, translation, and content generation.

#######

import spacy
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.cluster import KMeans
from openai import OpenAI
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import networkx as nx
from wordcloud import WordCloud
from rake_nltk import Rake
import numpy as np
from textstat import textstat
import gradio as gr

# If you run into issues with different backends, uncomment the next line:
# matplotlib.use('Agg')

# Download NLTK resources (ensure these are downloaded at least once)
nltk.download("punkt")
nltk.download("punkt_tab")
nltk.download("stopwords")

# Load SpaCy model
nlp = spacy.load("en_core_web_sm")

# Instantiate OpenAI client (replace with your actual API key "your-openai-api-key" if needed)
client = OpenAI(api_key="your-openai-api-key")

# -------------------- Utility Functions -------------------- #

def compute_tfidf(documents, top_n=5):
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(documents)
    feature_names = vectorizer.get_feature_names_out()
    dense = tfidf_matrix.todense()
    scores = dense[0].tolist()[0]
    tfidf_scores = [(feature_names[i], scores[i]) for i in range(len(scores))]
    sorted_scores = sorted(tfidf_scores, key=lambda x: x[1], reverse=True)
    return sorted_scores[:top_n]

def topic_modeling(documents, n_topics=3):
    vectorizer = TfidfVectorizer(stop_words="english")
    doc_term_matrix = vectorizer.fit_transform(documents)
    lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
    lda.fit(doc_term_matrix)
    topics = {}
    for idx, topic in enumerate(lda.components_):
        topics[f"Topic {idx + 1}"] = [
            vectorizer.get_feature_names_out()[i] for i in topic.argsort()[-5:]
        ]
    return topics

def summarize_text(text, length="short"):
    length_prompt = {
        "short": "Summarize in one sentence.",
        "medium": "Summarize in a short paragraph.",
        "long": "Summarize in detail.",
    }
    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "user", "content": f"{length_prompt[length]} The text is:\n{text}"}
        ],
    )
    return completion.choices[0].message.content.strip()

def classify_sentiment(sentiment):
    if sentiment["polarity"] > 0.1:
        return "Positive"
    elif sentiment["polarity"] < -0.1:
        return "Negative"
    else:
        return "Neutral"

# -------------------- Visualization Functions -------------------- #

def visualize_tfidf_figure(tfidf_scores):
    fig, ax = plt.subplots()
    words, scores = zip(*tfidf_scores) if tfidf_scores else ([], [])
    ax.barh(words, scores)
    ax.set_xlabel("TF-IDF Score")
    ax.set_title("Top TF-IDF Keywords")
    plt.tight_layout()
    return fig

def generate_wordcloud_figure(text):
    wordcloud = WordCloud(width=800, height=400, background_color="white").generate(text)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation="bilinear")
    ax.axis("off")
    ax.set_title("Word Cloud")
    plt.tight_layout()
    return fig

def create_cooccurrence_network_figure(tokens):
    cooccurrence_graph = nx.Graph()
    for i, token1 in enumerate(tokens):
        for token2 in tokens[i + 1 : i + 5]:
            if token1 != token2:
                if cooccurrence_graph.has_edge(token1, token2):
                    cooccurrence_graph[token1][token2]["weight"] += 1
                else:
                    cooccurrence_graph.add_edge(token1, token2, weight=1)
    pos = nx.spring_layout(cooccurrence_graph, seed=42)
    fig, ax = plt.subplots(figsize=(12, 8))
    nx.draw(
        cooccurrence_graph, pos, with_labels=True,
        node_color="lightblue", edge_color="gray", font_size=10, ax=ax
    )
    ax.set_title("Co-occurrence Network")
    plt.tight_layout()
    return fig

def generate_polarity_heatmap_figure(text):
    sentences = text.split(". ")
    polarities = [
        TextBlob(sentence).sentiment.polarity for sentence in sentences if sentence
    ]
    if not polarities:
        polarities = [0.0]
    fig, ax = plt.subplots(figsize=(10, 2))
    data = np.array(polarities).reshape(1, -1)
    sns.heatmap(
        data, annot=True, fmt=".2f", cmap="coolwarm", cbar=True,
        xticklabels=range(1, len(polarities) + 1), yticklabels=["Polarity"], ax=ax
    )
    ax.set_title("Sentence Polarity Heatmap")
    ax.set_xlabel("Sentence Index")
    plt.tight_layout()
    return fig

# -------------------- Other NLP Functions -------------------- #

def dependency_parsing(text):
    doc = nlp(text)
    for token in doc:
        print(f"{token.text} -> {token.dep_} -> {token.head.text}")

def compute_semantic_similarity(text, documents):
    base_doc = nlp(text)
    # This will show a warning if you're using a small model that doesn't have word vectors.
    similarities = [(doc, base_doc.similarity(nlp(doc))) for doc in documents]
    return sorted(similarities, key=lambda x: x[1], reverse=True)

def extract_keywords_rake(text):
    rake = Rake()
    rake.extract_keywords_from_text(text)
    return rake.get_ranked_phrases_with_scores()

# -------------------- FIX: Graceful KMeans Clustering -------------------- #
def cluster_documents(documents, n_clusters=3):
    # If the user doesn't provide enough documents, lower the cluster count or skip
    if len(documents) < n_clusters:
        # If there's only 0 or 1 document, skip clustering
        if len(documents) <= 1:
            return [0] * len(documents)  # Return 0 if there's exactly 1 doc
        else:
            n_clusters = len(documents)

    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(documents)
    km = KMeans(n_clusters=n_clusters, random_state=42)
    km.fit(tfidf_matrix)
    return km.labels_

def calculate_readability(text):
    readability_scores = {
        "flesch_reading_ease": textstat.flesch_reading_ease(text),
        "gunning_fog_index": textstat.gunning_fog(text),
        "smog_index": textstat.smog_index(text),
        "automated_readability_index": textstat.automated_readability_index(text),
    }
    return readability_scores

def pos_tagging_analysis(text):
    doc = nlp(text)
    pos_counts = Counter([token.pos_ for token in doc])
    return dict(pos_counts)

# -------------------- Main Pipeline -------------------- #

def process_text_with_pipeline(text, documents):
    # Step 1: Named Entity Recognition (NER)
    doc = nlp(text)
    entities = [(ent.text, ent.label_) for ent in doc.ents]

    # Step 2: Tokenization and Stopword Removal
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words("english"))
    clean_tokens = [
        word for word in tokens if word.isalnum() and word.lower() not in stop_words
    ]

    # Step 3: Word Frequencies
    word_freq = Counter(clean_tokens)

    # Step 4: Sentiment Analysis
    blob = TextBlob(text)
    sentiment = {
        "polarity": blob.sentiment.polarity,
        "subjectivity": blob.sentiment.subjectivity,
        "classification": classify_sentiment(blob.sentiment._asdict()),
    }

    # Step 5: TF-IDF
    tfidf_keywords = compute_tfidf(documents)

    # Step 6: Topic Modeling
    topics = topic_modeling(documents)

    # Step 7: Summarization
    summary = summarize_text(text)

    # Step 8: Dependency Parsing (print in console)
    dependency_parsing(text)

    # Step 9: Semantic Similarity
    similarities = compute_semantic_similarity(text, documents)

    # Step 10: RAKE Keywords
    keywords = extract_keywords_rake(text)

    # Step 11: Clustering
    clusters = cluster_documents(documents)

    # Figures
    polarity_heatmap_fig = generate_polarity_heatmap_figure(text)
    wordcloud_fig = generate_wordcloud_figure(text)
    cooccurrence_fig = create_cooccurrence_network_figure(clean_tokens)

    # Step 16: POS Tagging
    pos_counts = pos_tagging_analysis(text)

    # Step 17: Readability
    readability_scores = calculate_readability(text)

    results = {
        "entities": entities,
        "clean_tokens": clean_tokens,
        "word_frequencies": word_freq.most_common(10),
        "sentiment": sentiment,
        "tfidf_keywords": tfidf_keywords,
        "topics": topics,
        "summary": summary,
        "semantic_similarities": similarities,
        "rake_keywords": keywords,
        "clusters": clusters,
        "pos_counts": pos_counts,
        "readability_scores": readability_scores,
    }

    # Final TF-IDF Figure
    tfidf_fig = visualize_tfidf_figure(tfidf_keywords)

    return results, wordcloud_fig, cooccurrence_fig, polarity_heatmap_fig, tfidf_fig

# -------------------- Gradio Interface -------------------- #

def gradio_pipeline(text, documents):
    # Convert multiline box into a list of documents
    if isinstance(documents, str):
        docs_list = [doc.strip() for doc in documents.split("\n") if doc.strip()]
    else:
        docs_list = documents

    results, wordcloud_fig, cooccurrence_fig, polarity_heatmap_fig, tfidf_fig = process_text_with_pipeline(
        text, docs_list
    )

    return (
        results["entities"],
        results["clean_tokens"],
        results["word_frequencies"],
        results["sentiment"],
        results["tfidf_keywords"],
        results["topics"],
        results["summary"],
        results["semantic_similarities"],
        results["rake_keywords"],
        results["clusters"],
        results["pos_counts"],
        results["readability_scores"],
        wordcloud_fig,
        cooccurrence_fig,
        polarity_heatmap_fig,
        tfidf_fig,
    )

with gr.Blocks() as demo:
    gr.Markdown(
        "## NLP Pipeline with Multiple Results Panels\n"
        "### Developed by Partha Pratim Ray\n"
        "Contact: [parthapratimray1986@gmail.com](mailto:parthapratimray1986@gmail.com)\n"
        "GitHub: [https://github.com/ParthaPRay/](https://github.com/ParthaPRay/)"
    )
    with gr.Row():
        text_input = gr.Textbox(
            label="Enter your text here",
            lines=5,
            placeholder="Type or paste the text to analyze...",
        )
        docs_input = gr.Textbox(
            label="Enter your documents",
            lines=5,
            placeholder="Doc1...\nDoc2...\nDoc3...",
        )

    submit_button = gr.Button("Submit")

    # Panels (outputs)
    named_entities = gr.JSON(label="Named Entities")
    clean_tokens = gr.JSON(label="Clean Tokens")
    word_frequencies = gr.JSON(label="Word Frequencies")
    sentiment_analysis = gr.JSON(label="Sentiment Analysis")
    tfidf_keywords = gr.JSON(label="Top TF-IDF Keywords")
    topics = gr.JSON(label="Topics")
    summary = gr.Textbox(label="Summary")
    semantic_similarities = gr.JSON(label="Semantic Similarities")
    rake_keywords = gr.JSON(label="RAKE Keywords")
    clusters = gr.JSON(label="Document Clusters")
    pos_counts = gr.JSON(label="POS Tagging Counts")
    readability_scores = gr.JSON(label="Readability Scores")

    # Plots
    wordcloud_plot = gr.Plot(label="Word Cloud")
    cooccurrence_plot = gr.Plot(label="Co-occurrence Network")
    polarity_heatmap_plot = gr.Plot(label="Polarity Heatmap")
    tfidf_plot = gr.Plot(label="TF-IDF Chart")

    submit_button.click(
        fn=gradio_pipeline,
        inputs=[text_input, docs_input],
        outputs=[
            named_entities,
            clean_tokens,
            word_frequencies,
            sentiment_analysis,
            tfidf_keywords,
            topics,
            summary,
            semantic_similarities,
            rake_keywords,
            clusters,
            pos_counts,
            readability_scores,
            wordcloud_plot,
            cooccurrence_plot,
            polarity_heatmap_plot,
            tfidf_plot,
        ],
    )

if __name__ == "__main__":
    demo.launch()

