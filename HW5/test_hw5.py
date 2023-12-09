from fastapi.testclient import TestClient
from hw5_fastapi import app

client = TestClient(app)


def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "It's a live (by group 1.7)"}

def test_predict():
    """
    Функция проверяет работособоность предсказаний. Для добавления тестов используется список test_list.
    Каждый элемент списка должен содержать словарь со следующими полями:

    context - текст в котором ищем ответ.
    question - вопрос, на который ищем ответ.
    answer - ответ, который должны получить, задавая вопрос.
    score - число, минимальная уверенность, которая допустима при ответе на вопрос.
    """
    
    test_list = [{"context": "My name is Tim and I live in Sweden.", "question": "Where do I live?", "answer": "Sweden", "score": 0.8},
                 {"context": "My name is Tim and I live in Sweden.", "question": "What is my name?", "answer": "Tim", "score": 0.8}
    ]

    for valid_rule in test_list:

        # Получаем параметры, необходимые для запроса из правила проверки
        params = dict(context = valid_rule["context"], question = valid_rule["question"])

        response = client.post("/predict/", json = params)
        json_data = response.json() 

        assert response.status_code == 200
        assert json_data["answer"] == valid_rule["answer"]
        assert float(json_data["score"]) >= valid_rule["score"]