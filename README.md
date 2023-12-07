# urfu_pi_1_7
URFU master's degree for group 1.7 - start sept 2023 
### В репозитории в общей папке личные тесты и для обсуждения выполнения ДЗ в команде
### Далее выполнены ДЗ 2-3-4 описание указано в пунктах HW2\HW3\HW4

## by Monakhov
Файл model-mt5-small.py содержит пример использования модели lmqg/mt5-small-ruquad-qg с сайта huggingface.co
Модель строит вопросы на основе переданного текста. Для корректного запуска программы, необходимы установленные
библиотеки pytorch и transformers.


### Igor Eroshin 
	HW1 - Тест модели из примера на платформе SF 

	Будем анализировать текст на эмоциональную окраску
	Для этого напишем скрипт на Python и будем использовать готовую библиотеку Transformers для создания pipeline и использования готовой модели
	1) на виртуальной машине\пк linux установите Python,pip, библиотеку Transformers
	2) сделайте копию репозитория
	3) запустите скрипт sentiment_igor.py 
	4) Анализируемую фразу вы можете указать изменив всего 1 строку в скрипте
	Готово
	
	HW2+HW3+HW4 
	Общая работа над домашними заданиями, комментарии ниже в разделах Home Work *

## Klim Kolchin (@synrocka)
<p>Мной была выбрана модель **timpal0l/mdeberta-v3-base-squad2** с сайта hugginface.co<br>
Модель отвечает на простые вопросы по заданному тексту. Для запуска кода необходима библиотека transformers.<br>
Текст вопроса находится в переменной 'question'; текст, по которому задается вопрос, нужно указать в переменной 'context'.<p>


## Home Work 2
Папка HW2, содержит файл test_streamlit.py, который демонстрирует работоспособность модели **timpal0l/mdeberta-v3-base-squad2**, 
используя WEB-интерфейс библиотеки streamlit. Для использования, необходимо ввести текст и задать вопрос по этому тексту. 
Модель должна ответить на вопрос, используя полученные знания из заданного текста. 

## Home Work 3
Папка HW3, содержит файл test_fastapi.py, который демонстрирует работоспособность модели **timpal0l/mdeberta-v3-base-squad2**, 
используя HTTP-сервисы, построенные на библиотеке FastAPI. Для использования, необходимо запустить web-сервер с приложением
командой vicorn test_fastapi:app из директории HW3. Подробное описание по работе с API будет содержаться по адресу http://127.0.0.1:8000/docs.
Все необходимые библиотеки для работы приложения описаны в файле requirements.txt из директории HW3. Пример работы приложения с использованием
curl содержится в файле screenshot.png в той же директории.

## Home Work 4
Папка HW4 содержит файлы для запуска модели на платформе яндекс-облако при использовании библиотеки streamlit. С работой приложения
можно ознакомиться по адресу http://158.160.51.20:8501 
