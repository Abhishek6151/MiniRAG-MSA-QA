
#### **MiniRAG – MSA Contract Q\&A System**



A RAG-based Q\&A system built on the Salesforce Master Services Agreement. Upload the contract, ask questions in plain English, and get answers backed by retrieved evidence from the document.


#### **Setup**
Clone or download the repository.
Install the required dependencies:
pip install -r requirements.txt
Create a .env file in the project root directory.

#### **Example:**

GEMINI_API_KEY=your_gemini_api_key
Run the Streamlit application:
streamlit run app.py

Note: The Gemini API key is not included in this repository for security reasons. A valid API key must be added to the .env file before running the application.


###### **MSA Used**



Salesforce Main Services Agreement (publicly available at salesforce.com/legal), 17 pages. Covers payment terms, termination, liability, IP ownership, confidentiality, governing law, and dispute resolution — good coverage of all 10 benchmark categories.







###### **Chunk Size: 800 | Overlap: 150**



I tried 700, 800, and 1000 during development. 1000 hurt retrieval because chunks started mixing multiple clauses together — the embedding became an average of 3 topics instead of one. 700 was too small for longer clauses like Section 10.1 (limitation of liability) which runs 900 characters.

800 matched the average clause length in this document. 150 overlap is roughly one sentence  enough to rescue a clause that splits across a chunk boundary without creating too much duplication.







###### **Embedding Model: all-MiniLM-L6-v2**



Lightweight, runs on CPU, no GPU needed. Works well for short legal text and is fast enough to embed a 17-page contract in seconds. The same model is used for both chunks and questions so both live in the same vector space.







###### **Vector Store: ChromaDB**



No external setup needed — just pip install and it runs locally. Good enough for a 17-page document. Would switch to Pinecone at production scale.







###### **LLM: Gemini 3.1 Flash Lite**



Used for both answer generation and the judge step. Temperature set to 0 for consistent deterministic outputs across runs. Free tier requires a 4 second sleep between API calls during evaluation to stay under the 15 requests per minute limit.







###### **Working**



1\. Upload the Salesforce MSA PDF

2\. Text extracted page by page using pypdf

3\. Split into 800 character chunks with 150 overlap

4\. Each chunk embedded with all-MiniLM-L6-v2 and stored in ChromaDB

5\. User question embedded with same model

6\. Top 5 most relevant chunks retrieved from ChromaDB similarity search

7\. Similarity Report displayed with rank, location, score, signal, preview

8\. Top 5 chunks passed as context to Gemini with instruction to answer only from context

9\. Answer displayed below the Similarity Report







###### **Evaluation**



10 ground truth questions prepared manually before running the system — one per category (payment terms, late payment, delivery penalty, termination conditions, notice period, liability cap, IP ownership, confidentiality duration, governing law, dispute resolution).

Each system answer is judged by a second Gemini call using the exact judge prompt specified in the brief. Results show Match / Partial Match / No Match  with a one line reason. Final score: 7 Match, 2 Partial, 1 No Match — 80% accuracy.





###### **Known Failures**



Q7 – IP Ownership (No Match): Retrieval pulls Section 6.2 licensing chunks instead of Section 6.1 reservation of rights. Fix: prepend section headers to chunk text before embedding so ownership queries score higher for 6.1.



Q6 – Limitation of Liability (Partial Match): System finds the right clause but also pulls Germany-specific unlimited liability exceptions from pages 12-13. Fix: filter country-specific local law override sections during chunking.



Q8 – Confidentiality Duration (Partial Match): System finds that confidentiality survives termination but lists all surviving sections instead of focusing on no fixed end date. Fix: more targeted generation prompt.





###### **Stack**



Python

Streamlit

pypdf

sentence-transformers

ChromaDB

Google Gemini API

pandas


 Project Files

- `MiniRAG.ipynb` – Development notebook used for PDF extraction, chunking experiments, embedding generation, retrieval testing, and evaluation analysis.The notebook was used to build and test the RAG pipeline step by step. Once the workflow was working, the reusable functions were moved to rag_pipeline.py and connected to the Streamlit application.
- `rag_pipeline.py` – Core RAG pipeline functions including chunking, retrieval, answer generation, and evaluation.
- `app.py` – Streamlit user interface for contract question answering and automated evaluation.
- `ground_truth.json` – Benchmark questions and expected answers used for evaluation.
- `requirements.txt` – Python dependencies required to run the project.
  
###### Notes

For security reasons, the Gemini API key is not included in this repository. Create a .env file and add your own API key before running the project.
