from transformers import pipeline
import streamlit as st


@st.cache_resource
def load_model():
    model = pipeline('question-answering', 'timpal0l/mdeberta-v3-base-squad2')
    return model


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

# Загрузка модели
model = load_model()

st.title('Поиск ответа на вопрос в тексте')

context = st.text_input('Введите текст, в котором необходимо искать ответы:', value='My name is Tim and I live in Sweden.')

question = st.text_input('Введите вопрос:', value='Where do I live?')

result = st.button('Получить ответ')

#  Вывод результата
if result:
    response = model(question = question, context = context)
    print(response)
    text = 'Ответ **' + strip_symbol(response['answer'], ' .<>:,=*') + '** с вероятностью **' + str(response['score']) + '**.'
    st.write(text)
