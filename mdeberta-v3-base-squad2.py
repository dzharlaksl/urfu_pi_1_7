from transformers import pipeline

qa_model = pipeline("question-answering", "timpal0l/mdeberta-v3-base-squad2")
question = "Where do I live?"
context = "My name is Tim and I live in Sweden."
qa_model(question = question, context = context)
# {'score': 0.975547730922699, 'start': 28, 'end': 36, 'answer': ' Sweden.'}
