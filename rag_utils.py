from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import chromadb
import google.generativeai as genai
import re
import os
from dotenv import load_dotenv
#load Gemini API key
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel("gemini-3.1-flash-lite")
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
# extract text from PDF and create chunks
def process_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    chunk_size = 800
    chunk_overlap = 150
    page_chunks = []

    for page_num, page in enumerate(reader.pages, start=1):
        text = page.extract_text()
        if not text:
            continue
        text = re.sub(r"\s+", " ", text).strip()

        start = 0
        while start < len(text):
            end = start + chunk_size
            page_chunks.append({
                "text": text[start:end],
                "page": page_num,
                "location": f"Page {page_num}"
            })
            start = end - chunk_overlap

    chunk_texts = [item["text"] for item in page_chunks]
#generate embeddings
    chunk_embeddings = embedding_model.encode(chunk_texts, show_progress_bar=False)

    client = chromadb.Client()
    try:
        client.delete_collection("contract_collection")
    except NotFoundError:
        pass
# store chunk vectors in ChromaDB
    collection = client.create_collection(name="contract_collection")
    collection.add(
        embeddings=chunk_embeddings.tolist(),
        documents=chunk_texts,
        metadatas=[{"page": item["page"], "location": item["location"]} for item in page_chunks],
        ids=[f"chunk_{i}" for i in range(len(page_chunks))]
    )
    return collection

# retrieve top 5 matching chunks for a question
def retrieve_chunks(question, collection, top_k=5):
    question_embedding = embedding_model.encode(question)
    results = collection.query(
        query_embeddings=[question_embedding.tolist()],
        n_results=top_k
    )
    return results
# combine retrieved chunks into a single context
def build_context(results):
    context = ""
    for chunk in results["documents"][0]:
        context += chunk + "\n\n"
    return context

def generate_answer(question, context):
    prompt = f"""
    You are a contract analysis assistant.
    Answer only using the provided context.
    Do not make up information.
    If a clause or penalty is not mentioned anywhere in the context,
    explicitly state that no such clause exists in this contract.
    Do not say "answer not found" — instead explain what is absent.
    Context:
    {context}
    Question:
    {question}
    """
    response = model.generate_content(prompt,generation_config=genai.types.GenerationConfig(temperature=0))
    return response.text
#use Gemini as a judge
def judge_answer(expected_answer, system_answer):
    judge_prompt = f"""
You are evaluating a RAG system's answer against a ground truth answer extracted from a contract document.
Compare the two answers and classify the result as exactly one of:
Match
Partial Match
No Match
Then provide a single sentence explaining your classification.
Do not add any other commentary.
Ground Truth Answer:
{expected_answer}
System Answer:
{system_answer}
"""
    response = model.generate_content(judge_prompt,generation_config=genai.types.GenerationConfig(temperature=0))
    return response.text
