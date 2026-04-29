import feedparser
import httpx
from bs4 import BeautifulSoup
from datetime import datetime
from typing import List, Optional
import re
import logging

from src.config import settings
from src.models.schemas import NewsItem

logger = logging.getLogger(__name__)

def clean_html(html_content: str) -> str:
    """Удаляет HTML-теги и нормализует текст"""
    if not html_content:
        return ""
    soup = BeautifulSoup(html_content, 'lxml')
    # Удаляем скрипты, стили, навигацию
    for tag in soup(['script', 'style', 'nav', 'header', 'footer']):
        tag.decompose()
    text = soup.get_text(separator=' ', strip=True)
    # Нормализуем пробелы
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def parse_rss_feed(feed_url: str, limit: Optional[int] = None) -> List[NewsItem]:
    """Парсит RSS-фид и возвращает список новостей"""
    try:
        response = httpx.get(feed_url, timeout=30)
        response.raise_for_status()
        feed = feedparser.parse(response.content)
        
        items = []
        source = "hardware" if "hardware" in feed_url else "software"
        
        for entry in feed.entries[:limit]:
            # Извлекаем контент (приоритет: content > summary > description)
            content = ""
            if hasattr(entry, 'content') and entry.content:
                content = entry.content[0].value
            elif hasattr(entry, 'summary'):
                content = entry.summary
            elif hasattr(entry, 'description'):
                content = entry.description
            
            # Парсим дату публикации
            published = None
            if hasattr(entry, 'published_parsed') and entry.published_parsed:
                published = datetime(*entry.published_parsed[:6])
            elif hasattr(entry, 'updated_parsed') and entry.updated_parsed:
                published = datetime(*entry.updated_parsed[:6])
            else:
                published = datetime.now()
            
            item = NewsItem(
                title=getattr(entry, 'title', 'Без заголовка'),
                content=clean_html(content),
                url=getattr(entry, 'link', ''),
                published_at=published,
                source=source
            )
            items.append(item)
            logger.info(f"Parsed: {item.title[:50]}...")
        
        logger.info(f"Loaded {len(items)} items from {feed_url}")
        return items
        
    except Exception as e:
        logger.error(f"Error parsing {feed_url}: {e}")
        return []

def load_all_feeds(limit_per_feed: Optional[int] = None) -> List[NewsItem]:
    """Загружает новости из всех фидов"""
    all_items = []
    for feed_url in settings.rss_feeds:
        items = parse_rss_feed(feed_url, limit=limit_per_feed)
        all_items.extend(items)
    return all_items