from bs4 import BeautifulSoup
import requests
from urllib.parse import urljoin
def extract_links(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    base_url = url
    links = []
    for link in soup.find_all('a'):
        href = link.get('href')
        if href and href.endswith('.html'):
            full_url = urljoin(base_url, href)
            links.append(full_url)
    return links