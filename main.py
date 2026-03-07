import asyncio
import base64
import os
import logging
import aiohttp
from openai import AsyncOpenAI
from aiogram import Bot, Dispatcher, F
from aiogram.types import Message
from aiogram.filters import CommandStart, Command
from aiogram.fsm.storage.memory import MemoryStorage

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BOT_TOKEN = os.environ["BOT_TOKEN"]
OPENROUTER_API_KEY = os.environ["OPENROUTER_API_KEY"]

bot = Bot(token=BOT_TOKEN)
dp = Dispatcher(storage=MemoryStorage())

client = AsyncOpenAI(
      base_url="https://openrouter.ai/api/v1",
      api_key=OPENROUTER_API_KEY,
)

MODEL = "google/gemini-2.0-flash-001"

# User profiles stored compactly
user_profiles = {}

def get_profile(uid):
      if uid not in user_profiles:
                user_profiles[uid] = {
                              "name": "",
                              "skin_type": "",
                              "week": 1,
                              "symptoms": "",
                              "photo_summary": "",
                              "triggers": [],
                              "care_plan": "",
                }
            return user_profiles[uid]

def profile_context(p):
      parts = []
    if p["name"]: parts.append(f"Имя: {p['name']}")
          if p["skin_type"]: parts.append(f"Тип кожи: {p['skin_type']}")
                parts.append(f"Неделя программы: {p['week']}")
    if p["symptoms"]: parts.append(f"Симптомы: {p['symptoms']}")
          if p["photo_summary"]: parts.append(f"Последнее фото: {p['photo_summary']}")
                if p["triggers"]: parts.append(f"Триггеры: {', '.join(p['triggers'])}")
                      if p["care_plan"]: parts.append(f"План ухода: {p['care_plan']}")
                            return "\n".join(parts)

ANALYST_PROMPT = """Ты дерматолог-аналитик SkinCoach. Анализируй симптомы и фото.
Выдай ТОЛЬКО внутренний анализ (до 80 слов): ключевые проблемы, динамика, что нужно уточнить.
Без вступлений."""

STRATEGIST_PROMPT = """Ты стратег по уходу SkinCoach. На основе симптомов составь план.
Выдай ТОЛЬКО внутренний план (до 80 слов): схема ухода, питание, триггеры, действие на сегодня.
Без вступлений."""

JUDGE_PROMPT = """Ты финальный редактор SkinCoach. Получаешь два внутренних анализа.
Составь ФИНАЛЬНЫЙ ответ пользователю на русском. СТРОГО такой формат (макс 120 слов):

Понял: [1 предложение]
Сделай сейчас: [1-2 действия]
Следи за: [1 вещь]
Вопрос: [1 вопрос]

Без лишних слов, без вступлений."""

async def call_ai(system, user_content, max_tokens=200):
      resp = await client.chat.completions.create(
          model=MODEL,
import base64
import os
import logging
import aiohttp
from openai import AsyncOpenAI
from aiogram import Bot, Dispatcher, F
from aiogram.types import Message
from aiogram.filters import CommandStart, Command
from aiogram.fsm.storage.memory import MemoryStorage

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BOT_TOKEN = os.environ["BOT_TOKEN"]
OPENROUTER_API_KEY = os.environ["OPENROUTER_API_KEY"]

bot = Bot(token=BOT_TOKEN)
dp = Dispatcher(storage=MemoryStorage())
client = AsyncOpenAI(base_url="https://openrouter.ai/api/v1", api_key=OPENROUTER_API_KEY)
MODEL = "google/gemini-2.0-flash-001"

user_profiles = {}

def get_profile(uid):
    if uid not in user_profiles:
        user_profiles[uid] = {"name":"","skin_type":"","week":1,"symptoms":"","photo_summary":"","triggers":[],"care_plan":""}
    return user_profiles[uid]

def profile_ctx(p):
    parts = []
    if p["name"]: parts.append(f"Имя: {p['name']}")
    if p["skin_type"]: parts.append(f"Кожа: {p['skin_type']}")
    parts.append(f"Неделя: {p['week']}")
    if p["symptoms"]: parts.append(f"Симптомы: {p['symptoms']}")
    if p["photo_summary"]: parts.append(f"Фото: {p['photo_summary']}")
    if p["triggers"]: parts.append(f"Триггеры: {', '.join(p['triggers'])}")
    if p["care_plan"]: parts.append(f"Уход: {p['care_plan']}")
    return "
".join(parts)

async def call_ai(system, content, max_tokens=200):
    r = await client.chat.completions.create(
        model=MODEL,
        messages=[{"role":"system","content":system},{"role":"user","content":content}],
        max_tokens=max_tokens
    )
    return r.choices[0].message.content.strip()

async def multi_agent(uid, user_input, image_b64=None):
    p = get_profile(uid)
    ctx = profile_ctx(p)
    text_ctx = f"Профиль:
{ctx}
Запрос: {user_input}"

    if image_b64:
        vision_content = [
            {"type":"text","text":f"Профиль:
{ctx}
Запрос: {user_input}
Опиши кожу на фото."},
            {"type":"image_url","image_url":{"url":f"data:image/jpeg;base64,{image_b64}"}}
        ]
        r1 = await client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role":"system","content":"Ты Skin Analyst. Опиши состояние кожи на фото. Что видишь, что важно уточнить. Макс 80 слов."},
                {"role":"user","content":vision_content}
            ],
            max_tokens=150
        )
        a1 = r1.choices[0].message.content.strip()
    else:
        a1 = await call_ai("Ты Skin Analyst. Выдели ключевые симптомы и что нужно уточнить. Макс 80 слов.", text_ctx, 150)

    a2 = await call_ai("Ты Care Strategist. Предложи конкретный план ухода на сегодня. Макс 80 слов.", text_ctx, 150)

    judge_prompt = f"Analyst:
{a1}

Strategist:
{a2}

Профиль:
{ctx}"
    final = await call_ai(
        """Ты финальный редактор SkinCoach. Напиши ответ пользователю строго по формату:
1. Что я понял — 1 предложение
2. Что делать сейчас — 1-2 действия  
3. Что отслеживать — 1 пункт
4. Вопрос — один вопрос
Максимум 120 слов. Без вступлений.""",
        judge_prompt, 250
    )

    p["symptoms"] = await call_ai("Кратко опиши симптомы (макс 30 слов).", f"Симптомы: {p['symptoms']}
Новое: {user_input}", 50)
    return final

async def download_photo(file_path):
    url = f"https://api.telegram.org/file/bot{BOT_TOKEN}/{file_path}"
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as resp:
            data = await resp.read()
    return base64.b64encode(data).decode("utf-8")

@dp.message(CommandStart())
async def cmd_start(message: Message):
    user_profiles.pop(message.from_user.id, None)
    await message.answer("Привет. Я SkinCoach — помощник по программе Чистая кожа.

Помогу найти триггеры, выстроить уход и отслеживать динамику.

Как тебя зовут и что беспокоит кожа?")

@dp.message(Command("reset"))
async def cmd_reset(message: Message):
    user_profiles.pop(message.from_user.id, None)
    await message.answer("Профиль сброшен. /start чтобы начать.")

@dp.message(Command("help"))
async def cmd_help(message: Message):
    await message.answer("/start — начать
/reset — сброс
/profile — твой профиль

Пиши или присылай фото кожи.")

@dp.message(Command("profile"))
async def cmd_profile(message: Message):
    p = get_profile(message.from_user.id)
    ctx = profile_ctx(p)
    await message.answer(f"Профиль:
{ctx}" if ctx.strip() else "Профиль пуст. Напиши /start")

@dp.message(F.photo)
async def photo_handler(message: Message):
    uid = message.from_user.id
    await message.answer("Анализирую фото...")
    try:
        file = await bot.get_file(message.photo[-1].file_id)
        b64 = await download_photo(file.file_path)
        logger.info(f"Photo b64 len: {len(b64)}")
        caption = message.caption or "Проанализируй фото кожи"
        response = await multi_agent(uid, caption, image_b64=b64)
        get_profile(uid)["photo_summary"] = response[:80]
        await message.answer(response)
    except Exception as e:
        logger.error(f"Photo error: {e}")
        await message.answer("Ошибка при обработке фото. Попробуй ещё раз.")

@dp.message(F.text)
async def text_handler(message: Message):
    uid = message.from_user.id
    p = get_profile(uid)
    if not p["name"]:
        words = message.text.strip().split()
        if words:
            p["name"] = words[0].capitalize()
    response = await multi_agent(uid, message.text)
    await message.answer(response)

async def main():
    logger.info("SkinCoach bot starting...")
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())
