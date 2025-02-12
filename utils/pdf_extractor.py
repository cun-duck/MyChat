from langchain.text_splitter import RecursiveCharacterTextSplitter

def extract_text_from_pdf(file):
    """
    Extracts text from a PDF file.
    Returns the extracted text as a string.
    """
    reader = PyPDF2.PdfReader(file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text


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
