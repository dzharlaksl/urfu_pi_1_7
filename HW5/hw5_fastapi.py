from transformers import pipeline
from fastapi import FastAPI
from pydantic import BaseModel

class Item(BaseModel):
    context: str
    question: str

# Markdown не применяется форматирование жирным шрифтом, если с краю стоит "." или ",". 
# Модель ошибочно может выводить их в ответ, поэтому такие символы надо удалить.
def strip_symbol(text, symbols):

    # Проходим текст 2 раза: с конца строки и с начала
    for step in range(2):

        text = text[::-1]
        start = 0
        for i in text:
            if i in symbols:
                start += 1
            else:
                break

        text = text[start:]

    return text


app = FastAPI()

model = pipeline('question-answering', 'timpal0l/mdeberta-v3-base-squad2')

# GET метод для проверки работоспособности

@app.get('/')
def root():
    return {'message': "It's a live (by group 1.7)"}


@app.post("/predict/")
def predict(item: Item):
    """
    API позволяет найти ответ на заданный вопрос в тексте.

    Для работы необходимо сделать POST запрос со следуюими параметрами в теле:
    context - строка, содержащая текст
    queston - строка, содержащая вопрос, ответ на который необходимо найти.

    За реализацию функционала отвечает модель timpal0l/mdeberta-v3-base-squad2
    с сайта huggingface.co. Модель поддерживает ru, en и множество других языков.
    """
    result = model(question = item.question, context = item.context)
    result["answer"] = strip_symbol(result["answer"], ' ".<>:,=*')
    return result
