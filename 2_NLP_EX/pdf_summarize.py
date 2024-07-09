import os
from PyPDF2 import PdfReader
import streamlit as st
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import requests
from huggingface_hub import login

def process_text(text):
    # CharacterTextSplitter를 사용하여 텍스트를 청크로 분할
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)

    # 임베딩 처리(벡터 변환), 임베딩은 HuggingFaceEmbeddings 모델을 사용합니다.
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    documents = FAISS.from_texts(chunks, embeddings)
    return documents

def summarize_with_llama2(chunks, hf_token):
    model_name = "meta-llama/Meta-Llama-3-8B"
    headers = {"Authorization": f"Bearer {hf_token}"}
    
    # Check if the model is accessible
    response = requests.get(f"https://huggingface.co/{model_name}", headers=headers)
    if response.status_code != 200:
        raise ValueError("Access to the model is restricted or the token is invalid.")

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=hf_token)
    model = AutoModelForCausalLM.from_pretrained(model_name, use_auth_token=hf_token)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    inputs = tokenizer(" ".join(chunks), return_tensors="pt", truncation=True, padding="longest").to(device)
    summary_ids = model.generate(inputs["input_ids"], max_length=200, num_beams=4, early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    return summary

def main():
    st.title("📄PDF 요약하기")
    st.divider()
    pdf = st.file_uploader('PDF파일을 업로드해주세요', type='pdf')
    login("my_token")
    hf_token = st.text_input("Hugging Face Token", type="password")
    if not hf_token:
        st.error("Please enter your Hugging Face token.")
        return

    if pdf is not None:
        pdf_reader = PdfReader(pdf)
        text = ""   # 텍스트 변수에 PDF 내용을 저장
        for page in pdf_reader.pages:
            text += page.extract_text()

        documents = process_text(text)
        chunks = [doc.page_content for doc in documents.similarity_search("summary")]
        summary = summarize_with_llama2(chunks, hf_token)

        st.subheader('--요약 결과--:')
        st.write(summary)

if __name__ == '__main__':
    main()