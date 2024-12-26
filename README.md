# NLP Pipeline with Graceful Clustering

This **Natural Language Processing (NLP) Pipeline** is designed to analyze, cluster, and visualize text data using advanced machine learning techniques. With a user-friendly Gradio interface, the application offers intuitive interactions for users to input text and explore the results through dynamic visualizations and structured outputs.

---

## Key Features

1. **Text Preprocessing**: Tokenization, stopword removal, and POS tagging.
2. **Text Analysis**:
   - **TF-IDF Analysis**: Identify the most significant terms.
   - **Sentiment Analysis**: Evaluate polarity and subjectivity of text.
   - **Readability Metrics**: Assess text complexity with metrics like Flesch Reading Ease.
   - **Keyword Extraction**: Highlight significant terms using RAKE.
   - **Dependency Parsing**: Visualize linguistic dependencies.
3. **Clustering**:
   - Group documents into meaningful clusters using KMeans.
4. **Topic Modeling**:
   - Identify dominant themes using Latent Dirichlet Allocation (LDA).
5. **Visualization**:
   - **Word Cloud**: Visualize frequent terms.
   - **TF-IDF Chart**: Highlight top keywords.
   - **Co-occurrence Network**: Show word relationships.
   - **Polarity Heatmap**: Visualize sentence-level sentiment.
6. **Gradio Interface**: Interactive panels to explore results easily.

---

## Requirements

### Dependencies
The required Python packages are listed in `requirements.txt`:
```plaintext
spacy
wordcloud
networkx
nltk
textblob
scikit-learn
seaborn
matplotlib
rake-nltk
textstat
gradio
openai
```

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/nlp-pipeline.git
   cd nlp-pipeline
   ```

2. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Download required NLTK and SpaCy resources:
   ```python
   python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
   python -m spacy download en_core_web_sm
   ```

---

## Workflow Overview

### Pipeline Steps

1. **Input Text and Documents**: Users input a text string for analysis and optional documents for comparison.
2. **Preprocessing**: 
   - Tokenization and stopword removal using `nltk`.
   - POS tagging and named entity recognition using `spacy`.
3. **Feature Extraction**:
   - Compute TF-IDF scores using `scikit-learn`.
   - Extract keywords with `rake-nltk`.
4. **Analysis**:
   - Sentiment analysis using `textblob`.
   - Dependency parsing with `spacy`.
   - Semantic similarity computation between text and documents.
   - Readability metrics with `textstat`.
5. **Clustering and Topic Modeling**:
   - Cluster documents with KMeans.
   - Identify topics using LDA.
6. **Visualization**:
   - Word clouds with `wordcloud`.
   - Bar charts for TF-IDF results using `matplotlib`.
   - Co-occurrence networks with `networkx`.
   - Polarity heatmaps with `seaborn`.
7. **Interactive Results**:
   - Use Gradio for a web-based interface to display results dynamically.

---

## Key Functions

### Text Preprocessing
```python
def dependency_parsing(text):
    doc = nlp(text)
    for token in doc:
        print(f"{token.text} -> {token.dep_} -> {token.head.text}")
```

### Feature Extraction
```python
def compute_tfidf(documents, top_n=5):
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(documents)
    feature_names = vectorizer.get_feature_names_out()
    dense = tfidf_matrix.todense()
    scores = dense[0].tolist()[0]
    tfidf_scores = [(feature_names[i], scores[i]) for i in range(len(scores))]
    sorted_scores = sorted(tfidf_scores, key=lambda x: x[1], reverse=True)
    return sorted_scores[:top_n]
```

### Clustering
```python
def cluster_documents(documents, n_clusters=3):
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(documents)
    km = KMeans(n_clusters=n_clusters, random_state=42)
    km.fit(tfidf_matrix)
    return km.labels_
```

### Visualization
- **TF-IDF Chart**:
  ```python
  def visualize_tfidf_figure(tfidf_scores):
      fig, ax = plt.subplots()
      words, scores = zip(*tfidf_scores) if tfidf_scores else ([], [])
      ax.barh(words, scores)
      ax.set_xlabel("TF-IDF Score")
      ax.set_title("Top TF-IDF Keywords")
      plt.tight_layout()
      return fig
  ```
- **Word Cloud**:
  ```python
  def generate_wordcloud_figure(text):
      wordcloud = WordCloud(width=800, height=400, background_color="white").generate(text)
      fig, ax = plt.subplots(figsize=(10, 5))
      ax.imshow(wordcloud, interpolation="bilinear")
      ax.axis("off")
      ax.set_title("Word Cloud")
      plt.tight_layout()
      return fig
  ```

---

## Usage

### Running the Application
1. Start the application:
   ```bash
   python main.py
   ```
2. Open the Gradio interface at `http://127.0.0.1:7860`.

### Example Input
- **Text**: `"Artificial intelligence revolutionizes industries."`
- **Documents**:
  ```
  AI is transforming healthcare.
  Robotics drives automation.
  Machine learning enables new opportunities.
  ```

### Example Output
- **Named Entities**: `["Artificial intelligence", "industries"]`
- **Sentiment Analysis**: `Positive (Polarity: 0.85)`
- **Clusters**: `[0, 1, 2]`
- **TF-IDF Keywords**: `["artificial", "intelligence", "revolutionizes"]`
- **Readability Scores**:
  ```json
  {
      "flesch_reading_ease": 70.2,
      "gunning_fog_index": 8.3,
      "smog_index": 7.2
  }
  ```

---

## Gradio Panels
- **Inputs**:
  - **Text**: Multiline text for analysis.
  - **Documents**: Multiline input for additional context.
- **Outputs**:
  - JSON panels for analysis results.
  - Plots for visual insights (e.g., Word Cloud, Polarity Heatmap).

---

## Customization

### Adjust Number of Topics
```python
topic_modeling(documents, n_topics=5)
```

### Modify Clusters
```python
cluster_documents(documents, n_clusters=4)
```

---

## Troubleshooting

| **Issue**                 | **Solution**                                                      |
|----------------------------|------------------------------------------------------------------|
| Missing NLTK Data          | Run `nltk.download('punkt')` and `nltk.download('stopwords')`.  |
| SpaCy Model Missing        | Run `python -m spacy download en_core_web_sm`.                  |
| Backend Errors             | Uncomment `matplotlib.use('Agg')` for compatibility.           |

---

## Contribution

We welcome contributions! Fork the repository and submit pull requests to improve features or fix bugs.

---

## License

This project is licensed under the MIT License. See `LICENSE` for details.

---

## Screenshots

### Gradio Interface
![Gradio Interface](screenshot_ui.png)

### Visualizations
- **Word Cloud**:
  ![Word Cloud](screenshot_wordcloud.png)
- **Polarity Heatmap**:
  ![Polarity Heatmap](screenshot_heatmap.png)

---

## References
- [SpaCy Documentation](https://spacy.io/)
- [NLTK Documentation](https://www.nltk.org/)
- [TextBlob Documentation](https://textblob.readthedocs.io/)
- [Gradio Documentation](https://gradio.app/) 

Happy Analyzing! 🚀
