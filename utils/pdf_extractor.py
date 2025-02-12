import PyPDF2
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def is_valid_pdf(file):
    """
    Validates if the uploaded file is a valid PDF.
    """
    try:
        reader = PyPDF2.PdfReader(file)
        if len(reader.pages) == 0:
            return False  # Empty PDF
        return True
    except Exception:
        return False


def extract_text_from_pdf(file):
    """
    Extracts text from a PDF file.
    Returns the extracted text as a string.
    """
    if not is_valid_pdf(file):
        raise ValueError("The uploaded file is not a valid PDF.")

    try:
        file.seek(0)  # Reset file pointer to the beginning
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text() or ""  # Handle pages with no text
        return text
    except Exception as e:
        raise ValueError(f"Failed to process PDF: {e}")


def split_text_into_chunks(text, chunk_size=500, overlap=100):
    """
    Splits the text into chunks of a specified size with overlap.
    Returns a list of chunks.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = splitter.split_text(text)
    return chunks


def get_relevant_chunks(question, chunks, top_n=3):
    """
    Selects the most relevant chunks based on the question using TF-IDF.
    Returns a list of top_n relevant chunks.
    """
    vectorizer = TfidfVectorizer()
    chunk_vectors = vectorizer.fit_transform(chunks)
    question_vector = vectorizer.transform([question])
    
    # Calculate cosine similarity between the question and each chunk
    similarities = cosine_similarity(question_vector, chunk_vectors).flatten()
    
    # Get indices of top_n most similar chunks
    top_indices = similarities.argsort()[-top_n:][::-1]
    relevant_chunks = [chunks[i] for i in top_indices]
    
    return relevant_chunks
