from transformers import pipeline


qa_model = None


def load_qa():
    global qa_model
    if qa_model is None:
        qa_model = pipeline("question-answering", model="deepset/minilm-uncased-squad2", device=-1)
    return qa_model


def answer_question(context, question):
    if not isinstance(context, str) or not context.strip():
        return {"answer": "Please provide a context passage.", "confidence": 0.0}
    if not isinstance(question, str) or not question.strip():
        return {"answer": "Please provide a question.", "confidence": 0.0}
    qa = load_qa()
    result = qa(context=context, question=question, handle_impossible_answer=True)
    answer = result["answer"].strip()
    score = float(result["score"])
    if not answer or score < 0.2:
        answer = "No reliable answer found in the passage."
    return {"answer": answer, "confidence": round(score, 3)}
