# English-and-Spanish-movie-reviews-

Multilingual Movie Reviews – NLP Pipeline (English & Spanish)
Full Pipeline: Data Cleaning → Tokenization → POS & Parsing → NER → Sentiment Classification → Ablation → Final Outputs
 Project Overview
This project implements a full NLP pipeline for 2000 English & Spanish movie reviews, including:
D2: Cleaning, normalization, tokenization, exploratory analysis
D3: POS tagging, syntactic parsing, chunking, n-gram language modeling, perplexity
D4: Named Entity Recognition (NER), Sentiment Classification, Ablation Studies, Error Analysis
Final Output: Trained sentiment classifier + vectorizer + processed dataset
The goal is to design a reproducible, multilingual NLP system consistent across all deliverables.

 Dataset Source
All deliverables load the dataset from GitHub to ensure reproducibility:
https://raw.githubusercontent.com/Alneaimi/English-and-Spanish-movie-reviews-/main/multilingual_reviews_2000_en_es_improved.csv
 Installation & Environment Setup
1. Clone the repo
git clone https://github.com/Alneaimi/multilingual-movie-reviews.git
cd multilingual-movie-reviews
2. Install dependencies
pip install -r requirements.txt
3. Download spaCy models
python -m spacy download en_core_web_sm
python -m spacy download es_core_news_sm
 Deliverable 2 — Data Cleaning & Tokenization
D2 performs foundation-level preprocessing:
 Cleaning
Lowercasing
Removing URLs
Removing punctuation
Removing digits
Whitespace normalization
 Tokenization
Two options implemented:
Regex tokenizer (supports EN/ES characters)
NLTK word_tokenize
 Exploratory Data Analysis
Token length distribution
Language distribution
Sample inspection
Output of D2:
Cleaned text column → clean
Tokenized text column → tokens
 Deliverable 3 — POS Tagging, Parsing, N-Grams
D3 extends preprocessing with linguistic tasks:
 POS Tagging
Performed using spaCy:
Identifies part-of-speech categories (NOUN, VERB, ADJ...)
POS used later in Ablation (D4)
 Syntactic Parsing & Chunking
Extracts:
noun phrases (NPs)
verb phrases
headwords of noun phrases
Visualization includes:
NP length distributions
Top NP heads
Top verbs
Ambiguous POS tokens
 N-Gram Language Model + Perplexity
Trigram MLE model used:
Training text: tokenized reviews
Computed perplexity per review
Fixed for invalid token sequences
Output of D3:
clean, tokens, pos, chunks, ner-ready data
🔍 Deliverable 4 — NER, Sentiment Classification, Ablation, Final Pipeline
 Named Entity Recognition (NER)
Using en_core_web_sm, the pipeline identifies:
PERSON
ORG
GPE
DATE
WORK_OF_ART
Outputs include:
entity text
entity label
entity frequency distribution
 Sentiment Classification
Baseline Model
Bag-of-Words (CountVectorizer)
Logistic Regression
Strong but simple start
Improved Model
TF-IDF Vectorizer
Multinomial Naive Bayes
Achieved significantly higher performance
Produces:
Accuracy
Precision
Recall
F1
Confusion Matrix
 Ablation Studies (Required for D4)
 Ablation 1 — No Cleaning
Trains classifier on raw text.
Ablation 2 — With POS tags
Append POS tags to text → evaluate benefit.
Produces a comparison:
Model	Accuracy
BoW (Baseline)	xx%
TF-IDF (Improved)	xx%
Ablation 1 (Raw)	xx%
Ablation 2 (With POS)	xx%
 Error Analysis
Lists misclassified examples:
True label
Predicted
Raw text
Observed patterns (short reviews, sarcasm, unusual entity-heavy text)
� Final Deliverables
Inside models/
sentiment_model_tfidf.pkl
tfidf_vectorizer.pkl
Inside outputs/
final_pipeline_output.csv
These allow direct inference without re-running the notebooks.
▶️ How to Run Sentiment Prediction
import joblib

model = joblib.load("models/sentiment_model_tfidf.pkl")
vectorizer = joblib.load("models/tfidf_vectorizer.pkl")

text = ["The movie was amazing and emotional."]
X = vectorizer.transform(text)

prediction = model.predict(X)
print("Prediction:", "Positive" if prediction[0] == 1 else "Negative")
📊 Project Summary
✔ Fully multilingual (English + Spanish)
✔ Reproducible across all deliverables
✔ Uses consistent cleaning/tokenization across D2 → D3 → D4
✔ Includes linguistic features (POS, NP/VP, NER)
✔ Strong sentiment classifier with detailed evaluation
✔ Ablation studies demonstrating model behavior
✔ Complete final pipeline with exported models
