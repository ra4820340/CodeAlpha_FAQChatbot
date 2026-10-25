from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st

st.set_page_config(page_title="FAQ Chatbot", page_icon="🤖", layout="centered")

st.title("🤖 CodeAlpha FAQ Chatbot")
st.write(
    "Ask any question about the internship program and get instant answers!"
)

# Predefined FAQs dataset
faqs = {
    "What is CodeAlpha?": (
        "CodeAlpha is a software development company providing real-world AI"
        " internship experiences."
    ),
    "How do I submit my tasks?": (
        "You are required to submit your completed tasks using the official"
        " submission form shared in your WhatsApp group."
    ),
    "What are the internship perks?": (
        "Perks include an internship offer letter, QR-verified completion"
        " certificate, unique ID, and LOR based on performance."
    ),
    "What is the naming convention for GitHub?": (
        "You must upload your source code to GitHub in a repository named"
        " CodeAlpha_ProjectName."
    ),
    "How many tasks must I complete?": (
        "You must complete a minimum of 2 or 3 tasks out of the given list to"
        " be eligible for certification."
    ),
}

questions = list(faqs.keys())
answers = list(faqs.values())

# User chat input
user_query = st.text_input(
    "Type your question:", placeholder="e.g., How do I submit tasks?"
)

if st.button("Ask Chatbot", type="primary"):
  if user_query.strip() != "":
    # Vectorize and compute cosine similarity
    vectorizer = TfidfVectorizer()
    all_texts = questions + [user_query]
    tfidf_matrix = vectorizer.fit_transform(all_texts)

    query_vector = tfidf_matrix[-1]
    faq_vectors = tfidf_matrix[:-1]
    similarities = cosine_similarity(query_vector, faq_vectors).flatten()

    best_match_idx = similarities.argmax()
    best_score = similarities[best_match_idx]

    # Threshold for matching
    if best_score > 0.15:
      st.success(f"**Answer:** {answers[best_match_idx]}")
    else:
      st.info(
          "I'm sorry, I couldn't find a matching FAQ. Please check the"
          " internship guidelines document."
      )
  else:
    st.warning("Please type a question first.")