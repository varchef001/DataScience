import re
from transformers import pipeline
from qa_system import answer_question


generator = None


def load_generator():
    global generator
    if generator is None:
        generator = pipeline("text2text-generation", model="google/flan-t5-base", device=-1)
    return generator


def generate_questions(text, num_questions=5):
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Please enter a non-empty passage.")
    if not isinstance(num_questions, int) or num_questions < 1:
        raise ValueError("num_questions must be a positive integer.")
    if len(text.split()) < 10:
        raise ValueError("Please provide a longer passage containing several facts.")
    # Try questions from different sentences first.
    sentences = re.split(r"(?<=[.!?])\s+|\n+", text.strip())
    sentences = [s for s in sentences if len(s.split()) >= 4]
    questions = []
    seen = set()
    generator = load_generator()
    for attempt in range(2):
        for sentence in sentences:
            instruction = "Write one specific factual question answered by this text."
            if attempt:
                instruction = "Write a different question about another fact in this text."
            prompt = (
                f"{instruction}\nReturn only a question ending in ?. "
                f"If there is no clear fact, return NONE.\nText: {sentence}"
            )
            output = generator(prompt, max_new_tokens=64, num_beams=4,
                               do_sample=False, truncation=True)
            question = output[0]["generated_text"].strip()
            question = re.sub(r"^(question\s*:|\d+[.)])\s*", "", question,
                              flags=re.IGNORECASE)
            if "?" not in question:
                continue
            question = question.split("?")[0].strip() + "?"
            key = re.sub(r"\W+", " ", question.lower()).strip()
            if key and key not in seen:
                # Keep the question only if an answer can be found.
                checked = answer_question(text, question)
                if checked["answer"] == "No reliable answer found in the passage.":
                    continue
                seen.add(key)
                questions.append(question)
            if len(questions) >= num_questions:
                return questions
    print(f"Generated {len(questions)} of {num_questions} requested questions. "
                  "Add more factual details if needed.")
    return questions
