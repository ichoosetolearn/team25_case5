import requests

try:
    url = "http://127.0.0.1:8000/search"
    response = requests.post(url, params={'entity': 'мельников андрей'})
    print(response.json())
except Exception as e:
    print(f"Ошибка: {e}")

try:
    url = "http://127.0.0.1:8000/search"
    response = requests.post(url, params={'entity': 'sdfsd'})
    print(response.json())
except Exception as e:
    print(f"Ошибка: {e}")