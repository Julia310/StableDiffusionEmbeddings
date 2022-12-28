import requests

URL = "https://lexica.art/"
page = requests.get(URL)

print(page.text)
