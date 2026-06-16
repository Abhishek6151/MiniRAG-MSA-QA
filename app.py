import streamlit as st
import pandas as pd
import json
import time
from rag_utils import (
    process_pdf,
    retrieve_chunks,
    build_context,
    generate_answer,
    judge_answer,
    )
with open("ground_truth.json", "r") as f:
    ground_truth = json.load(f)
st.title("MiniRAG - MSA Contract Q&A System")
if "answer" not in st.session_state:
    st.session_state.answer = ""
if "report_df" not in st.session_state:
    st.session_state.report_df = None
if "eval_df" not in st.session_state:
    st.session_state.eval_df = None
if "metrics" not in st.session_state:
    st.session_state.metrics = None
uploaded_file = st.file_uploader("Upload MSA Contract", type=["pdf"])
question = st.text_input("Ask a question about the contract")
collection = None
if uploaded_file is not None:
    pdf_path = uploaded_file.name
    with open(pdf_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    collection =process_pdf(pdf_path)
submit_btn = st.button("Submit Question")
evaluate_btn = st.button("Run Evaluation")
if uploaded_file is not None and submit_btn:
    with st.spinner("Processing question..."):
        results = retrieve_chunks(question, collection, top_k=5)
    distances = results["distances"][0]
    metadatas = results["metadatas"][0]
    documents = results["documents"][0]
    max_dist = max(distances)
    min_dist = min(distances)
    report_data = []
    for i in range(len(documents)):
        # normalize similarity score between 0 and 1
        if max_dist == min_dist:
            similarity = 1.0
        else:
            similarity = round(1 - ((distances[i] - min_dist) / (max_dist - min_dist)), 2)
        if similarity >= 0.85:
            signal = "Strong"
        elif similarity >= 0.70:
            signal = "Good"
        elif similarity >= 0.50:
            signal = "Weak"
        else:
            signal = "Poor"
        report_data.append({
            "Rank": i + 1,
            "Location": metadatas[i]["location"],
            "Similarity Score": similarity,
            "Distance": round(distances[i], 4),
            "Score Signal": signal,
            "Chunk Preview": documents[i][:120]
        })
    report_df = pd.DataFrame(report_data)
    report_df.index = range(1, len(report_df) + 1)
    # generate answer using retrieved context
    context = build_context(results)
    answer = generate_answer(question, context)
    st.session_state.answer = answer
    st.session_state.report_df = report_df
# run evaluation on all benchmark questions    
if uploaded_file is not None and evaluate_btn:
    st.info("Running automated evaluation on 10 benchmark questions...")
    evaluation_results = []
    for item in ground_truth:
        eval_results = retrieve_chunks(item["question"], collection, top_k=5)
        eval_context = build_context(eval_results)
        system_answer = generate_answer(item["question"], eval_context)
        time.sleep(4)  
        evaluation_results.append({
            "category": item["category"],
            "question": item["question"],
            "expected_answer": item["expected_answer"],
            "system_answer": system_answer
        })
    judged_results = []
    # compare system answers against ground truth
    for item in evaluation_results:
        judge_output = judge_answer(item["expected_answer"], item["system_answer"]).strip()
        time.sleep(4)  
        if judge_output.startswith("Partial Match"):
            judgement = "Partial Match"
            reason = judge_output[len("Partial Match"):].strip(" .:-")
        elif judge_output.startswith("No Match"):
            judgement = "No Match"
            reason = judge_output[len("No Match"):].strip(" .:-")
        elif judge_output.startswith("Match"):
            judgement = "Match"
            reason = judge_output[len("Match"):].strip(" .:-")
        else:
            judgement = judge_output
            reason = ""
        judged_results.append({
            "category": item["category"],
            "judgement": judgement,
            "reason": reason
        })
    eval_df = pd.DataFrame(judged_results)
    eval_df = eval_df.rename(columns={
        "category": "Category",
        "judgement": "Judgement",
        "reason": "Reason"
    })
    eval_df = eval_df[["Category", "Judgement", "Reason"]]
    eval_df.index = range(1, len(eval_df) + 1)
    match_count = sum(item["judgement"] == "Match" for item in judged_results)
    partial_count = sum(item["judgement"] == "Partial Match" for item in judged_results)
    no_match_count = sum(item["judgement"] == "No Match" for item in judged_results)
    accuracy = ((match_count + (0.5 * partial_count)) / len(judged_results)) * 100
    st.session_state.eval_df = eval_df
    st.session_state.metrics = {
        "match": match_count,
        "partial": partial_count,
        "no_match": no_match_count,
        "accuracy": accuracy
    }
if uploaded_file is None:
    st.info("Please upload a PDF contract first.")
if st.session_state.report_df is not None:
    st.subheader("Similarity Report")
    st.dataframe(
        st.session_state.report_df,
        use_container_width=True
    )
if st.session_state.answer:
    st.subheader("Answer")
    st.success(st.session_state.answer)
if st.session_state.eval_df is not None:
    st.subheader("Evaluation Summary")
    st.table(st.session_state.eval_df)
    st.subheader("Overall Evaluation")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Match", st.session_state.metrics["match"])
    col2.metric("Partial Match", st.session_state.metrics["partial"])
    col3.metric("No Match", st.session_state.metrics["no_match"])
    col4.metric(
        "Accuracy",
        f"{st.session_state.metrics['accuracy']:.1f}%"
    )

    
