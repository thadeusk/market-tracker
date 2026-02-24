import os
import time
import json
from datetime import datetime, timezone
from dateutil import tz

import feedparser
import requests
from openai import OpenAI

MAX_ARTICLES = int(os.getenv("MAX_ARTICLES", "12"))
MAX_THEMES = int(os.getenv("MAX_THEMES", "4"))
MODEL = os.getenv("OPENAI_MODEL", "gpt-5-mini")
NOTION_VERSION = os.getenv("NOTION_VERSION", "2022-06-28")

NOTION_TOKEN = os.getenv("NOTION_TOKEN")
NOTION_DATABASE_ID = os.getenv("NOTION_DATABASE_ID")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not NOTION_TOKEN or not NOTION_DATABASE_ID or not OPENAI_API_KEY:
    raise SystemExit("Missing one of: NOTION_TOKEN, NOTION_DATABASE_ID, OPENAI_API_KEY")

client = OpenAI(api_key=OPENAI_API_KEY)

def read_sources(path: str = "config_sources.txt") -> list[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip() and not line.strip().startswith("#")]

def fetch_rss_items(feed_urls: list[str]) -> list[dict]:
    items = []
    for url in feed_urls:
        parsed = feedparser.parse(url)
        for e in parsed.entries[:25]:
            link = getattr(e, "link", None)
            title = getattr(e, "title", "").strip()
            summary = (getattr(e, "summary", "") or "").strip()
            published = getattr(e, "published", "") or getattr(e, "updated", "")
            if not link or not title:
                continue
            items.append({
                "source_feed": url,
                "title": title,
                "link": link,
                "summary": summary[:700],
                "published": published[:120],
            })

    # Deduplicate by link
    seen = set()
    deduped = []
    for it in items:
        if it["link"] in seen:
            continue
        seen.add(it["link"])
        deduped.append(it)
    return deduped

def pick_top_items(items: list[dict], max_n: int) -> list[dict]:
    # Minimal: just take the first N deduped items
    return items[:max_n]

def call_openai_to_extract_themes(as_of_date_local: str, items: list[dict]) -> dict:
    schema = {
        "name": "market_digest",
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "as_of": {"type": "string"},
                "themes": {
                    "type": "array",
                    "minItems": 1,
                    "maxItems": MAX_THEMES,
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "theme": {"type": "string"},
                            "what_happened": {"type": "string"},
                            "why_it_matters": {"type": "string"},
                            "market_impact": {"type": "string"},
                            "watch_next": {"type": "string"},
                            "confidence": {"type": "string", "enum": ["High", "Medium", "Low"]},
                            "best_source_url": {"type": "string"},
                        },
                        "required": [
                            "theme", "what_happened", "why_it_matters",
                            "market_impact", "watch_next", "confidence", "best_source_url"
                        ],
                    },
                },
            },
            "required": ["as_of", "themes"],
        },
    }

    prompt = f"""
You are a global markets interview-prep assistant.
Given the news items below, produce up to {MAX_THEMES} major market THEMES for {as_of_date_local}.

Rules:
- Base claims ONLY on the provided items. If unsure, set confidence to Low.
- Keep each field concise (2–5 sentences).
- "Market impact" should mention rates, equities, FX, credit, commodities as relevant.
- best_source_url must be one of the provided links.

News items:
{json.dumps(items, ensure_ascii=False)}
""".strip()

    # Structured Outputs: enforce JSON schema. See OpenAI docs. 
    resp = client.responses.create(
        model=MODEL,
        input=prompt,
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": schema["name"],
                "schema": schema["schema"],
                "strict": True,
            },
        },
    )

    return json.loads(resp.output_text)

def notion_create_page(theme_obj: dict, as_of_iso: str):
    url = "https://api.notion.com/v1/pages"
    headers = {
        "Authorization": f"Bearer {NOTION_TOKEN}",
        "Notion-Version": NOTION_VERSION,
        "Content-Type": "application/json",
    }

    payload = {
        "parent": {"database_id": NOTION_DATABASE_ID},
        "properties": {
            "Theme": {"title": [{"text": {"content": theme_obj["theme"]}}]},
            "As of": {"date": {"start": as_of_iso}},
            "What happened": {"rich_text": [{"text": {"content": theme_obj["what_happened"]}}]},
            "Why it matters": {"rich_text": [{"text": {"content": theme_obj["why_it_matters"]}}]},
            "Market impact": {"rich_text": [{"text": {"content": theme_obj["market_impact"]}}]},
            "Watch next": {"rich_text": [{"text": {"content": theme_obj["watch_next"]}}]},
            "Confidence": {"select": {"name": theme_obj["confidence"]}},
            "Sources": {"url": theme_obj["best_source_url"]},
        },
    }

    r = requests.post(url, headers=headers, data=json.dumps(payload).encode("utf-8"), timeout=30)
    if r.status_code >= 300:
        raise RuntimeError(f"Notion error {r.status_code}: {r.text}")
    return r.json()

def main():
    zurich = tz.gettz("Europe/Zurich")
    now_local = datetime.now(tz=timezone.utc).astimezone(zurich)
    as_of_date_local = now_local.strftime("%Y-%m-%d")
    as_of_iso = now_local.date().isoformat()

    feeds = read_sources()
    items = fetch_rss_items(feeds)
    if not items:
        print("No RSS items found.")
        return

    items = pick_top_items(items, MAX_ARTICLES)
    digest = call_openai_to_extract_themes(as_of_date_local, items)

    created = 0
    for t in digest["themes"]:
        notion_create_page(t, as_of_iso)
        created += 1
        time.sleep(0.4)

    print(f"Created {created} Notion pages for {digest['as_of']}")

if __name__ == "__main__":
    main()
