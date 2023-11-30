from transformers import pipeline

classifier = pipeline("sentiment-analysis", model="s-nlp/russian_toxicity_classifier")

classifier("Меня всё бесит")
