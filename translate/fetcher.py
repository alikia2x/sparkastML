import sqlite3
import trafilatura
import hashlib
import re
import os
from trafilatura.readability_lxml import is_probably_readerable
from concurrent.futures import ThreadPoolExecutor, as_completed

# 常量定义
MAX_FETCH_LIMIT = 300  # 每次运行时获取的最大任务数量

# 数据库连接
def connect_db(db_path):
    return sqlite3.connect(db_path)

# 创建fetch_list表
def create_fetch_list_table(conn):
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS fetch_list (
            url TEXT PRIMARY KEY,
            fetched_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()

# 获取未爬取的URL列表
def get_unfetched_urls(conn, limit):
    cursor = conn.cursor()
    cursor.execute("""
        SELECT url FROM url_list
        WHERE url NOT IN (SELECT url FROM fetch_list)
        LIMIT ?
    """, (limit,))
    return [row[0] for row in cursor.fetchall()]

# 下载并提取网页内容
def fetch_and_extract_content(url):
    downloaded = trafilatura.fetch_url(url)
    if not downloaded:
        return None
    
    html_string = downloaded
    if not is_probably_readerable(html_string):
        return None
    
    content = trafilatura.extract(html_string, output_format="txt", url=url, favor_precision=True)
    return content

# 计算URL的MD5
def md5_hash(url):
    return hashlib.md5(url.encode()).hexdigest()

# 分段规则
def split_content(content):
    sentences = re.split(r'[。！？；.!?;]', content)
    segments = []
    current_segment = []
    current_length = 0
    
    for sentence in sentences:
        sentence_length = len(sentence)
        if (len(current_segment) >= 12 or current_length + sentence_length > 1800):
            segments.append(''.join(current_segment))
            current_segment = []
            current_length = 0
        
        current_segment.append(sentence)
        current_length += sentence_length
    
    if current_segment:
        segments.append(''.join(current_segment))
    
    return segments

# 保存分段内容到文件
def save_segments(url, segments, path):
    url_hash = md5_hash(url)
    for idx, segment in enumerate(segments):
        save_path = os.path.join(path, f"{url_hash}_{idx}.txt")
        with open(save_path, "w", encoding="utf-8") as f:
            f.write(segment)

# 记录已爬取的URL
def record_fetched_url(conn, url):
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO fetch_list (url, fetched_time)
        VALUES (?, CURRENT_TIMESTAMP)
    """, (url,))
    conn.commit()

# 处理单个URL的任务
def process_url(url, db_path, save_path):
    conn = connect_db(db_path)
    content = fetch_and_extract_content(url)
    if content:
        segments = split_content(content)
        save_segments(url, segments, save_path)
        record_fetched_url(conn, url)
    conn.close()

# 主函数
def main():
    db_path = "crawler.db"
    save_path = "./source"
    conn = connect_db(db_path)
    
    # 创建fetch_list表
    create_fetch_list_table(conn)
    
    unfetched_urls = get_unfetched_urls(conn, MAX_FETCH_LIMIT)
    conn.close()
    
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(process_url, url, db_path, save_path) for url in unfetched_urls]
        
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()