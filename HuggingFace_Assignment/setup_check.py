from transformers import pipeline

classifier = pipeline(
    "sentiment-analysis",
    model="distilbert/distilbert-base-uncased-finetuned-sst-2-english"
)
result = classifier("Hugging Face makes NLP accessible.")
print(result)
