import os
import requests
from bs4 import BeautifulSoup
import sqlite3
import urllib.robotparser as urobot
from urllib.parse import urljoin, urlparse
from dotenv import load_dotenv

MAX_RECURSION_DEPTH = 5
MAX_URLS = 1000
MAX_THREADS = 10
HEADERS = {
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7",
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36",
}

conn = sqlite3.connect("crawler.db")
cursor = conn.cursor()

# Initialization
cursor.execute(
    """
CREATE TABLE IF NOT EXISTS url_list (
    url TEXT PRIMARY KEY,
    fetched_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    visited BOOLEAN,
    parent_url TEXT,
    child_url_count INTEGER
)
"""
)
conn.commit()


def fetch_url(url, headers=None):
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        return response.text, response.url
    except requests.RequestException as e:
        print(f"Error fetching {url}: {e}")
        return None, url


def extract_links(html, base_url):
    soup = BeautifulSoup(html, "html.parser")
    links = set()
    for a_tag in soup.find_all("a", href=True):
        href = a_tag["href"]
        full_url = urljoin(base_url, href)
        if urlparse(full_url).netloc == urlparse(base_url).netloc:
            links.add(full_url)
    return links


def fetch_sitemap(sitemap_url):
    html, _ = fetch_url(sitemap_url)
    if html:
        soup = BeautifulSoup(html, "xml")
        urls = {loc.text for loc in soup.find_all("loc")}
        return urls
    return set()


def save_url(url, parent_url=None):
    cursor = conn.cursor()
    cursor.execute(
        """
    INSERT OR IGNORE INTO url_list (url, visited, parent_url, child_url_count)
    VALUES (?, ?, ?, ?)
    """,
        (url, False, parent_url, 0),
    )
    conn.commit()


def update_url(url, child_url_count):
    cursor.execute(
        """
    UPDATE url_list SET child_url_count = ? WHERE url = ?
    """,
        (child_url_count, url),
    )
    conn.commit()


def crawl(url, rp=None, depth=0):
    if depth > MAX_RECURSION_DEPTH:
        return

    if (
        rp
        and rp.can_fetch("*", url) == False
        and rp.can_fetch("Googlebot", url) == False
        and rp.can_fetch("Baiduspider", url) == False
    ):
        return
    save_url(url)
    html, fetched_url = fetch_url(url, HEADERS)
    if not html:
        return

    cursor.execute(
        """
    UPDATE url_list SET visited = TRUE, fetched_time = CURRENT_TIMESTAMP WHERE url = ?
    """,
        (fetched_url,),
    )
    conn.commit()

    links = extract_links(html, fetched_url)
    for link in links:
        save_url(link, fetched_url)

    update_url(fetched_url, len(links))

    for link in links:
        crawl(link, depth=depth + 1)


def main(seed_url, rp, sitemap=None):
    if sitemap:
        sitemap_urls = fetch_sitemap(sitemap)
        for sitemap_url in sitemap_urls:
            save_url(sitemap_url)
    crawl(seed_url, rp=rp)

def get_config(key):
    return os.getenv(key)

# Example usage
if __name__ == "__main__":
    load_dotenv()
    seed_url = get_config("SEED_URL")
    rp = urobot.RobotFileParser()
    rp.set_url(get_config("ROBOTS_URL"))
    rp.read()
    main(seed_url, rp, get_config("SITEMAP_URL"))
    conn.close()
