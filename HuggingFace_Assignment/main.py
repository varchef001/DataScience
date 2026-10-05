from pathlib import Path
from test_generator import generate_questions
from qa_system import answer_question

# Replace this paragraph to try a different topic.
text = (
    "Green Valley Library opened in 2010. "
    "The library is located in Brookfield. "
    "The library contains 12000 books. "
    "Maria Chen is the library director. "
    "The library opens at nine in the morning on weekdays. "
    "The library offers free computer classes on Saturdays. "
    "Students can borrow up to four books at a time. "
    "The borrowing period is fourteen days."
)


def build_quiz_with_answers(text, num_questions=5):
    questions = generate_questions(text, num_questions)
    report = []
    for question in questions:
        result = answer_question(text, question)
        report.append({
            "question": question,
            "answer": result["answer"],
            "confidence": result["confidence"]
        })
    return report


def main():
    try:
        print("Generating the quiz. Please wait...")
        quiz = build_quiz_with_answers(text)
        lines = ["Input passage:", text, "", "Quiz:"]
        for number, item in enumerate(quiz, start=1):
            lines.append(f"\nQuestion {number}: {item['question']}")
            lines.append(f"Answer: {item['answer']}")
            lines.append(f"Confidence: {item['confidence']}")
        if len(quiz) < 5:
            lines.append("Not enough questions were found. Try adding more facts.")
        output = "\n".join(lines)
        print(output)
        file = Path(__file__).with_name("sample_output.txt")
        file.write_text(output + "\n", encoding="utf-8")
        print("\nThe output was saved in sample_output.txt.")
    except (ValueError, OSError, RuntimeError) as error:
        print("Error:", error)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
