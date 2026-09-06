import streamlit as st
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import re
import logging
import PyPDF2

# Suppress Streamlit warnings
logging.getLogger("streamlit").setLevel(logging.ERROR)

# Load model
@st.cache_resource
def load_model():
    try:
        return SentenceTransformer('all-MiniLM-L6-v2')
    except Exception as e:
        st.error(f"Failed to load model: {str(e)}")
        return None

MODEL = load_model()
if MODEL is None:
    st.stop()

# Sentence chunking
def sentence_chunking_basic(text):
    sentences = re.split(r'(?<!\.\.\.)(?<=\.|\?|\!)(?=\s)', text.strip())
    return [s.strip() for s in sentences if s.strip()]

# Embedding
def embed_text(text, model):
    return model.encode(text)

def embed_chunks(chunks, model):
    return [embed_text(chunk, model) for chunk in chunks]

# Coarse filtering (internal, not displayed)
def coarse_filter(question, text, model, percentile=95, window_size=7):
    chunks = sentence_chunking_basic(text)
    if not chunks:
        return "No sentences found.", []

    question_embedding = embed_text(question, model)
    chunk_embeddings = embed_chunks(chunks, model)
    similarities = [cosine_similarity([question_embedding], [ce])[0][0] for ce in chunk_embeddings]

    threshold = np.percentile(similarities, percentile)
    in_window = [False] * len(chunks)
    for i, sim in enumerate(similarities):
        if sim >= threshold:
            start = max(0, i - window_size // 2)
            end = min(len(chunks), i + window_size // 2 + 1)
            for j in range(start, end):
                in_window[j] = True

    result = " ".join([chunk for i, chunk in enumerate(chunks) if in_window[i]])
    return result, [(chunk, sim) for chunk, sim in zip(chunks, similarities)]

# Fine filtering with character limit
def fine_filter(text, question, model, window_size=3, char_limit=300):
    chunks = sentence_chunking_basic(text)
    if not chunks:
        return "No sentences found.", []

    question_embedding = embed_text(question, model)
    chunk_embeddings = embed_chunks(chunks, model)
    similarities = [cosine_similarity([question_embedding], [ce])[0][0] for ce in chunk_embeddings]

    # Sliding window to update similarities
    updated_similarities = []
    half_window = window_size // 2
    for i in range(len(similarities)):
        start = max(0, i - half_window)
        end = min(len(similarities), i + half_window + 1)
        updated_similarities.append(max(similarities[start:end]))

    # Rank chunks by similarity and select within character limit
    chunk_sim_pairs = sorted(zip(chunks, updated_similarities), key=lambda x: x[1], reverse=True)
    result = ""
    for chunk, _ in chunk_sim_pairs:
        if len(result) + len(chunk) + 1 <= char_limit:  # +1 for space
            result += chunk + " "
        else:
            break
    result = result.strip()
    
    if not result:
        result = chunk_sim_pairs[0][0][:char_limit].strip()  # Fallback to top sentence
    
    return result, [(chunk, sim) for chunk, sim in zip(chunks, updated_similarities)]

# Extract text from PDF
def extract_text_from_pdf(file):
    try:
        pdf_reader = PyPDF2.PdfReader(file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() or ""
        return text
    except Exception as e:
        st.error(f"Error reading PDF: {str(e)}")
        return None

# Streamlit UI
def main():
    st.title("Document Question Extractor")
    st.write("Extract relevant text from a document based on a question. Upload a .txt or .pdf file, or paste text below.")

    # File upload
    uploaded_file = st.file_uploader("Upload a file (.txt or .pdf)", type=["txt", "pdf"])
    document = None

    if uploaded_file is not None:
        file_type = uploaded_file.type
        if file_type == "text/plain":
            try:
                document = uploaded_file.read().decode("utf-8")
                st.success("Text file uploaded successfully!")
            except Exception as e:
                st.error(f"Error reading text file: {str(e)}")
        elif file_type == "application/pdf":
            document = extract_text_from_pdf(uploaded_file)
            if document:
                st.success("PDF file uploaded successfully!")
        else:
            st.error("Unsupported file type. Please upload a .txt or .pdf file.")

        if document:
            st.text_area("Uploaded Document Preview:", value=document, height=150, disabled=True)

    # Fallback text area
    if document is None:
        document_input = st.text_area("Or paste your document here:", height=200)
        if document_input:
            document = document_input

    # Question input
    question = st.text_input("Question:")

    # Parameters
    col1, col2 = st.columns(2)
    with col1:
        coarse_percentile = st.slider("Coarse Percentile", 50, 100, 95, help="Top % of sentences to include initially")
        coarse_window = st.slider("Coarse Window", 1, 15, 7, help="Context window size around top sentences")
    with col2:
        char_limit = st.selectbox("Max Characters in Output", [200, 300], index=1, help="Limit final text length")
        fine_window = st.slider("Fine Window", 1, 10, 3, help="Sliding window size for refinement")

    # Extract button
    if st.button("Extract"):
        if not document:
            st.error("Please upload a file or enter text in the document field.")
            return
        if not question:
            st.error("Please enter a question.")
            return

        with st.spinner("Processing..."):
            # Perform coarse filtering (not displayed)
            coarse_result, _ = coarse_filter(question, document, MODEL, coarse_percentile, coarse_window)
            # Directly show fine filtering result
            final_result, fine_data = fine_filter(coarse_result, question, MODEL, fine_window, char_limit=char_limit)

            st.subheader("Extracted Answer")
            st.write(final_result)
            st.write(f"Length: {len(final_result)} chars (limited to {char_limit})")

            with st.expander("Debug Info"):
                st.write("Fine Scores:", [(s, f"{sim:.3f}") for s, sim in fine_data])

if __name__ == "__main__":
    main()