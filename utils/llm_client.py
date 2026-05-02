import httpx
import os
import json
import asyncio
from dotenv import load_dotenv

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
BASE_URL = "https://openrouter.ai/api/v1/chat/completions"

PRODUCER_MODEL = os.getenv("PRODUCER_MODEL", "meta-llama/llama-3.1-8b-instruct:free")
CRITIC_MODEL   = os.getenv("CRITIC_MODEL",   "mistralai/mistral-7b-instruct:free")

CALL_DELAY  = 4   
MAX_RETRIES = 5


async def _post(model: str, messages: list, json_mode: bool = False) -> str:
    if not OPENROUTER_API_KEY:
        raise ValueError("OPENROUTER_API_KEY not set in .env")

    payload = {"model": model, "messages": messages}
    if json_mode:
        payload["response_format"] = {"type": "json_object"}

    await asyncio.sleep(CALL_DELAY)

    for attempt in range(MAX_RETRIES):
        async with httpx.AsyncClient(timeout=120) as client:
            resp = await client.post(
                BASE_URL,
                headers={
                    "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                    "Content-Type": "application/json",
                },
                json=payload,
            )
            if resp.status_code == 429:
                wait = 15 * (attempt + 1)
                print(f"      [rate limit] waiting {wait}s (attempt {attempt+1}/{MAX_RETRIES})...")
                await asyncio.sleep(wait)
                continue
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"]

    raise RuntimeError(f"Failed after {MAX_RETRIES} retries (rate limit).")


async def call_producer(prompt: str, system: str = "", json_mode: bool = False) -> str:
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    return await _post(PRODUCER_MODEL, messages, json_mode)


async def call_critic(prompt: str, system: str = "", json_mode: bool = False) -> str:
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    return await _post(CRITIC_MODEL, messages, json_mode)


async def call_json(prompt: str, system: str = "", use_critic: bool = False) -> dict:
    """Call LLM and parse JSON. Strips markdown fences if present."""
    fn = call_critic if use_critic else call_producer
    raw = await fn(prompt, system=system, json_mode=True)
    clean = raw.strip()
    if clean.startswith("```"):
        parts = clean.split("```")
        clean = parts[1]
        if clean.startswith("json"):
            clean = clean[4:]
    return json.loads(clean.strip())