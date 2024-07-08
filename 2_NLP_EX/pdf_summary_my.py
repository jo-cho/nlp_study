import streamlit as st 
import PyPDF2
from langchain.vectorstores import FAISS
from transformers import pipeline
from transformers import AutoTokenizer, AutoModel
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings

summarizer = pipeline("summarization", model="t5-small", tokenizer="t5-small")

def extract_text_from_pdf(pdf_file):
    pdf_reader = PyPDF2.PdfFileReader(pdf_file)
    text = ""
    for page_num in range(pdf_reader.numPages):
        page = pdf_reader.getPage(page_num)
        text += page.extract_text()
    return text
    
def embed_sentences(sentences):
    tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

    inputs = tokenizer(sentences, return_tensors='pt', padding=True, truncation=True)
    with torch.no_grad():
        embeddings = model(**inputs).last_hidden_state.mean(dim=1).cpu().numpy()
    return embeddings

def chunk_text(text, chunk_size=512, overlap=50):
    sentences = text.split('. ')
    embeddings = embed_sentences(sentences)
    
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    
    chunks = []
    i = 0
    while i < len(sentences):
        chunk = ' '.join(sentences[i:i + chunk_size])
        chunks.append(chunk)
        i += chunk_size - overlap
    
    return chunks

def summarize_text(text):
    chunks = chunk_text(text)
    summaries = [summarizer(chunk, max_length=150, min_length=30, do_sample=False)[0]['summary_text'] for chunk in chunks]
    return " ".join(summaries)

import streamlit as st

def main():
    st.title("PDF 요약 웹페이지")

    pdf_file = st.file_uploader("PDF 파일을 업로드하세요", type=["pdf"])
    
    if pdf_file is not None:
        text = extract_text_from_pdf(pdf_file)
        st.write("PDF 내용:")
        st.write(text)
        
        if st.button("요약"):
            summary = summarize_text(text)
            st.write("요약 결과:")
            st.write(summary)

if __name__ == "__main__":
    main()