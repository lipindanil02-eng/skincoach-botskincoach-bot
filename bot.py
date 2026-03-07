"""
SkinCoach v6 — 2 reasoner + judge + 28-дневная программа "Чистая кожа"
Ready for Railway.app deployment
"""
import asyncio, json, os, sys, base64, logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List
import httpx
from dotenv import load_dotenv
from telegram import Update
from telegram.constants import ChatAction
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, ContextTypes, filters

load_dotenv()

# ── Config ──
TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
API_KEY = os.getenv("OPENROUTER_API_KEY", "").strip()
VISION = os.getenv("VISION_MODEL", "nvidia/nemotron-nano-12b-v2-vl:free").strip()
REASON_A = os.getenv("REASONER_A_MODEL", "arcee-ai/trinity-large-preview:free").strip()
REASON_B = os.getenv("REASONER_B_MODEL", "stepfun/step-3.5-flash:free").strip()
JUDGE = os.getenv("JUDGE_MODEL", "mistralai/mistral-small-3.1-24b-instruct:free").strip()
CHAT = os.getenv("CHAT_MODEL", "arcee-ai/trinity-large-preview:free").strip()
VIS_FB = [m.strip() for m in os.getenv("VISION_FALLBACKS", "google/gemma-3-27b-it:free,google/gemma-3-12b-it:free").split(",") if m.strip()]
TXT_FB = [m.strip() for m in os.getenv("TEXT_FALLBACKS", "meta-llama/llama-3.3-70b-instruct:free,google/gemma-3-4b-it:free").split(",") if m.strip()]
MAX_TOK = int(os.getenv("MAX_TOKENS", "800"))
TEMP = float(os.getenv("TEMPERATURE", "0.3"))
TOUT = int(os.getenv("TIMEOUT", "120"))
HIST_FILE = "history.json"

logging.basicConfig(level=logging.INFO, format="%(asctime)s|%(levelname)s|%(message)s")
log = logging.getLogger("skincoach")

# ── States ──
S_NAME, S_DUR, S_TRIED, S_PHOTO, S_ACTIVE = "name", "dur", "tried", "photo", "active"

WEEKS = {1: "ПИТАНИЕ — убираем провокаторы", 2: "НАРУЖНЫЙ УХОД — мыло, масло, крем",
         3: "ЭМОЦИИ — стресс-протокол, кинезиология", 4: "АНАЛИЗЫ — контроль и коррекция"}
W_EMOJI = {1: "🥗", 2: "🧴", 3: "🧠", 4: "🔬"}

FOCUSES = {
    1: {1: "Составь список что ел за 3 дня", 2: "Исключи молочку", 3: "Убери сахар и белый хлеб",
        4: "Добавь куркуму с чёрным перцем", 5: "8 стаканов воды", 6: "Смузи: шпинат+банан+имбирь+куркума",
        7: "Итог недели питания"},
    2: {1: "Серное мыло НЕ при остром воспалении", 2: "Только тёплая вода до 37C",
        3: "Крем с мочевиной на влажную кожу", 4: "Масло точечно на бляшки после крема",
        5: "Если этап позволяет — серное мыло 1 раз", 6: "Полная схема утро+вечер",
        7: "Сравни кожу с 1м днём. Отправь фото"},
    3: {1: "Дыхание 4-7-8, 5 минут утром", 2: "Запиши 3 ситуации обострения",
        3: "Точечный массаж виски+между бровями", 4: "Аффирмация 10 раз утром",
        5: "Мышечное расслабление 10 мин вечером", 6: "Напиши письмо своей коже",
        7: "Итог: связь стресс и кожа?"},
    4: {1: "Запишись: ОАК, витамин D, ферритин, ТТГ", 2: "Копрограмма",
        3: "Проверь цинк и селен", 4: "Результаты? Отправь фото",
        5: "Скорректируй добавки", 6: "Составь персональный протокол",
        7: "Финальное фото для сравнения"},
}

# ── Utils ──
def rp(f, d=""): p=Path(f); return p.read_text("utf-8").strip() if p.exists() else d
def cm(t):
    t=t.replace("**","").replace("__","").replace("```","").replace("`","")
    return "\n".join(l.lstrip("#").strip() if l.lstrip().startswith("#") else l for l in t.split("\n"))

def xj(t):
    t=t.strip()
    if t.startswith("```"): t=t.split("\n",1)[-1]
    if t.endswith("```"): t=t.rsplit("```",1)[0]
    t=t.strip()
    try: return json.loads(t)
    except: pass
    s,e=t.find("{"),t.rfind("}")
    if s!=-1 and e>s:
        try: return json.loads(t[s:e+1])
        except: pass
    raise ValueError(f"No JSON: {t[:200]}")

# ── History ──
def lh():
    if os.path.exists(HIST_FILE):
        try:
            with open(HIST_FILE,"r",encoding="utf-8") as f: return json.load(f)
        except: return {}
    return {}
def sh(h):
    try:
        with open(HIST_FILE,"w",encoding="utf-8") as f: json.dump(h,f,ensure_ascii=False,indent=2)
    except Exception as e: log.error(f"Save:{e}")
def gu(h,uid):
    u=str(uid)
    if u not in h: h[u]={"state":S_NAME,"name":None,"duration":None,"tried":None,
        "vision":None,"analysis":None,"day":0,"week":1,"msgs":[],"created":datetime.now().isoformat()}
    return h[u]
def tm(m): return m[-30:] if len(m)>30 else m
def gc(u):
    l=u["msgs"][-4:] if u["msgs"] else []
    if not l: return ""
    return "Последние:\n"+"".join(f"{'Человек' if m['role']=='user' else 'Коуч'}: {(m['content'] if isinstance(m['content'],str) else str(m['content']))[:150]}\n" for m in l)

# ── API ──
def hdr(): return {"Authorization":f"Bearer {API_KEY}","Content-Type":"application/json",
    "HTTP-Referer":"https://t.me/skincoach_bot","X-Title":"SkinCoach"}

async def call(msgs, mdl, fb, mt=MAX_TOK):
    last_e=None
    async with httpx.AsyncClient(timeout=TOUT) as c:
        for m in [mdl]+fb:
            try:
                log.info(f"  -> {m}")
                r=await c.post("https://openrouter.ai/api/v1/chat/completions",headers=hdr(),
                    json={"model":m,"messages":msgs,"temperature":TEMP,"max_tokens":mt})
                if r.status_code==200:
                    d=r.json()
                    if "choices" in d and d["choices"]:
                        ct=d["choices"][0]["message"]["content"]
                        if isinstance(ct,list): ct="".join(p.get("text","") for p in ct if isinstance(p,dict))
                        log.info(f"  OK: {m}")
                        return ct
                log.warning(f"  {m}: {r.status_code}")
                last_e=f"{m}:{r.status_code}"
            except httpx.TimeoutException: log.warning(f"  {m}: timeout"); last_e=f"{m}:timeout"
            except Exception as e: log.warning(f"  {m}: {e}"); last_e=str(e)
    raise Exception(f"All models down. {last_e}")

async def call_json(msgs,mdl,fb,mt=MAX_TOK): return xj(await call(msgs,mdl,fb,mt))
async def call_text(msgs,mdl,fb,mt=MAX_TOK): return cm(await call(msgs,mdl,fb,mt))

# ── Consilium ──
async def consilium(b64, cap, u):
    nm,dur,tri=u.get("name","друг"),u.get("duration","?"),u.get("tried","?")
    dy,wk=u.get("day",1),u.get("week",1)
    wt=WEEKS.get(wk,"Программа"); diw=((dy-1)%7)+1; df=FOCUSES.get(wk,{}).get(diw,"Следуй программе")
    uc=f"Имя:{nm}, давность:{dur}, пробовали:{tri}, день:{dy}/28, неделя:{wk}"

    # 1. Vision
    log.info("👁 1/4 Vision...")
    vp=rp("vision_prompt.txt","Проанализируй фото кожи. JSON.")
    vm=[{"role":"system","content":vp+f"\nДанные: {uc}"},
        {"role":"user","content":[{"type":"text","text":cap or "Анализ фото"},
            {"type":"image_url","image_url":{"url":f"data:image/jpeg;base64,{b64}"}}]}]
    try: vr=await call_json(vm,VISION,VIS_FB,400)
    except Exception as e: log.error(f"Vision fail:{e}"); return "Не удалось проанализировать фото. Попробуй ещё раз."
    u["vision"]=vr

    # Check quality
    if vr.get("quality")=="poor":
        return f"{nm}, фото получилось нечётким. Переснимай при дневном свете, крупным планом, чтобы кожа была в фокусе."

    # 2. Reasoner A
    log.info("🧴 2/4 Reasoner A...")
    rap=rp("reasoner_a_prompt.txt","План ухода. JSON.")
    rac=json.dumps({"user":uc,"vision":vr,"day_focus":df},ensure_ascii=False)
    try: ra=await call_json([{"role":"system","content":rap},{"role":"user","content":rac}],REASON_A,TXT_FB,500)
    except: ra={"summary":"недоступно","care_steps":[],"confidence":0}

    # 3. Reasoner B
    log.info("🧠 3/4 Reasoner B...")
    rbp=rp("reasoner_b_prompt.txt","Психосоматика. JSON.")
    rbc=json.dumps({"user":uc,"vision":vr,"day_focus":df},ensure_ascii=False)
    try: rb=await call_json([{"role":"system","content":rbp},{"role":"user","content":rbc}],REASON_B,TXT_FB,500)
    except: rb={"psycho_summary":"недоступно","affirmation":"","confidence":0}

    # 4. Judge
    log.info("📋 4/4 Judge...")
    jp=rp("judge_prompt.txt","Объедини ответы. JSON.")
    jc=json.dumps({"user":uc,"vision":vr,"answer_a":ra,"answer_b":rb,
        "day":dy,"week":wk,"week_theme":wt,"day_focus":df,"name":nm},ensure_ascii=False)
    try:
        jr=await call_json([{"role":"system","content":jp},{"role":"user","content":jc}],JUDGE,TXT_FB,800)
        final=jr.get("final_answer","")
        fq=jr.get("follow_up_question","")
        if final:
            if fq and fq not in final: final+=f"\n\n{fq}"
        else: final=fallback(vr,ra,rb,u)
    except: final=fallback(vr,ra,rb,u)

    u["analysis"]=final
    return cm(final)

def fallback(v,ra,rb,u):
    nm,dy,wk=u.get("name",""),u.get("day",1),u.get("week",1)
    p=[f"День {dy}/28 — Неделя {wk} {W_EMOJI.get(wk,'📋')}\n"]
    s=ra.get("summary",""); 
    if s: p.append(f"🔍 {nm}, {s}\n")
    mc=ra.get("morning_care",[])
    if mc: p.append("🧴 Утро:"); p.extend(f"  {x}" for x in mc[:3]); p.append("")
    ec=ra.get("evening_care",[])
    if ec: p.append("🌙 Вечер:"); p.extend(f"  {x}" for x in ec[:3]); p.append("")
    nr=ra.get("nutrition_remove",[]); na=ra.get("nutrition_add",[])
    if nr or na:
        p.append("🥗 Питание:")
        if nr: p.append(f"  Убрать: {', '.join(nr[:3])}")
        if na: p.append(f"  Добавить: {', '.join(na[:3])}")
        p.append("")
    af=rb.get("affirmation","")
    if af: p.append(f"💫 {af}\n")
    mp=rb.get("morning_practice",""); ep=rb.get("evening_practice","")
    if mp: p.append(f"🧠 Утро: {mp}")
    if ep: p.append(f"🧠 Вечер: {ep}")
    p.append("\n📝 Вечером напиши: что сделал, как кожа, ощущения.")
    return "\n".join(p)

# ── Send ──
async def send(msg, txt):
    if len(txt)<=4000: await msg.reply_text(txt); return
    parts,cur=[],""
    for l in txt.split("\n"):
        if len(cur)+len(l)+1>4000:
            if cur: parts.append(cur)
            cur=l
        else: cur=cur+"\n"+l if cur else l
    if cur: parts.append(cur)
    for p in parts: await msg.reply_text(p)

# ── Handlers ──
async def cmd_start(upd: Update, ctx: ContextTypes.DEFAULT_TYPE):
    h=lh(); uid=str(upd.effective_user.id)
    h[uid]={"state":S_NAME,"name":None,"duration":None,"tried":None,"vision":None,
        "analysis":None,"day":0,"week":1,"msgs":[],"created":datetime.now().isoformat()}
    sh(h)
    await upd.message.reply_text(
        "Привет.\nЯ твой персональный помощник по программе 'Чистая кожа'.\n\n"
        "Я помогу тебе понять, что влияет на состояние кожи, выстроить уход "
        "и пройти персональный маршрут шаг за шагом.\n\n"
        "Каждое фото анализирует консилиум из 4 AI-экспертов.\n\n"
        "Для начала скажи: как тебя зовут?")

async def handle_text(upd: Update, ctx: ContextTypes.DEFAULT_TYPE):
    uid=upd.effective_user.id; txt=upd.message.text; h=lh(); u=gu(h,uid)
    await upd.message.chat.send_action(ChatAction.TYPING)

    if u["state"]==S_NAME:
        u["name"]=txt.strip(); u["state"]=S_DUR; sh(h)
        await upd.message.reply_text(f"{u['name']}, какая у тебя проблема с кожей и как давно беспокоит?")
        return
    if u["state"]==S_DUR:
        u["duration"]=txt.strip(); u["state"]=S_TRIED; sh(h)
        await upd.message.reply_text("Что уже пробовал(а)? Мази, диеты, народные средства, фототерапия?")
        return
    if u["state"]==S_TRIED:
        u["tried"]=txt.strip(); u["state"]=S_PHOTO; sh(h)
        await upd.message.reply_text(
            f"Отлично, {u['name']}.\n\n📸 Теперь отправь фото проблемного участка кожи.\n"
            "Дневной свет, крупный план.\n\nКонсилиум определит тип, стадию и составит план на День 1.")
        return
    if u["state"]==S_PHOTO:
        await upd.message.reply_text(f"{u.get('name','')}, мне нужно фото чтобы запустить анализ. 📸")
        return

    if u["state"]==S_ACTIVE:
        u["msgs"].append({"role":"user","content":txt}); u["msgs"]=tm(u["msgs"])
        wt=WEEKS.get(u["week"],"Программа"); an=(u.get("analysis") or "нет")[:300]
        cp=rp("chat_prompt.txt","Ты SkinCoach.").format(
            name=u.get("name","друг"),duration=u.get("duration","?"),
            tried=u.get("tried","?"),analysis=an,
            day=u["day"],week=u["week"],week_theme=wt)
        msgs=[{"role":"system","content":cp}]+u["msgs"]
        try: reply=await call_text(msgs,CHAT,TXT_FB,600)
        except: reply="Модели заняты. Напиши через пару минут."
        u["msgs"].append({"role":"assistant","content":reply}); u["msgs"]=tm(u["msgs"]); sh(h)
        await send(upd.message, reply)
        return

    u["state"]=S_NAME; sh(h)
    await upd.message.reply_text("Давай начнём. Как тебя зовут?")

async def handle_photo(upd: Update, ctx: ContextTypes.DEFAULT_TYPE):
    uid=upd.effective_user.id; h=lh(); u=gu(h,uid)
    if u["state"] in (S_NAME,S_DUR,S_TRIED):
        await upd.message.reply_text("Сначала познакомимся. /start"); return

    st=await upd.message.reply_text(
        "📸 Фото получено. Консилиум работает...\n\n"
        "👁 Дерматолог анализирует...\n🧴 Подбирает схему ухода...\n"
        "🧠 Оценивает психосоматику...\n📋 Собирает план...\n\n30-90 сек ⏳")
    await upd.message.chat.send_action(ChatAction.TYPING)

    ph=upd.message.photo[-1]; f=await ctx.bot.get_file(ph.file_id)
    b=await f.download_as_bytearray(); b64=base64.b64encode(b).decode()
    cap=(upd.message.caption or "").strip()

    if u["state"]==S_PHOTO or u["day"]==0:
        u["state"]=S_ACTIVE; u["day"]=1; u["week"]=1

    try: reply=await consilium(b64,cap,u)
    except Exception as e: reply="Ошибка анализа. Попробуй через минуту."; log.error(f"Cons:{e}")

    u["msgs"].append({"role":"user","content":f"[фото] {cap}"}); u["msgs"].append({"role":"assistant","content":reply})
    u["msgs"]=tm(u["msgs"]); sh(h)
    try: await st.delete()
    except: pass
    await send(upd.message, reply)

async def cmd_next(upd: Update, ctx: ContextTypes.DEFAULT_TYPE):
    uid=upd.effective_user.id; h=lh(); u=gu(h,uid)
    if u["state"]!=S_ACTIVE: await upd.message.reply_text("/start чтобы начать"); return
    await upd.message.chat.send_action(ChatAction.TYPING)
    u["day"]+=1
    if u["day"]>28:
        await upd.message.reply_text(f"🎉 {u.get('name','')}, программа пройдена! Отправь фото для сравнения.")
        sh(h); return
    u["week"]=((u["day"]-1)//7)+1
    wt=WEEKS.get(u["week"],"Программа"); diw=((u["day"]-1)%7)+1
    df=FOCUSES.get(u["week"],{}).get(diw,"Следуй программе")
    an=(u.get("analysis") or "нет")[:300]
    pr=rp("next_day_prompt.txt","План на день {day}.").format(
        day=u["day"],week=u["week"],week_theme=wt,week_emoji=W_EMOJI.get(u["week"],"📋"),
        name=u.get("name","друг"),duration=u.get("duration","?"),
        tried=u.get("tried","?"),analysis=an,day_focus=df,context=gc(u))
    msgs=[{"role":"system","content":pr},{"role":"user","content":f"План на день {u['day']}."}]
    try: plan=await call_text(msgs,CHAT,TXT_FB,600)
    except: plan="Не удалось. /next через минуту."
    u["msgs"].append({"role":"assistant","content":plan}); u["msgs"]=tm(u["msgs"]); sh(h)
    await send(upd.message, plan)

async def cmd_status(upd: Update, ctx: ContextTypes.DEFAULT_TYPE):
    h=lh(); u=gu(h,upd.effective_user.id)
    if u["state"]!=S_ACTIVE: await upd.message.reply_text("/start"); return
    wt=WEEKS.get(u["week"],"Программа"); pct=int((u["day"]/28)*100)
    bar="▓"*(pct//10)+"░"*(10-pct//10)
    await upd.message.reply_text(
        f"📊 {u.get('name','')}\n\nДень {u['day']}/28\nНеделя {u['week']}/4 — {wt}\n[{bar}] {pct}%\n\n"
        f"/next — следующий день\n📸 Фото — повторный анализ")

async def cmd_help(upd: Update, ctx: ContextTypes.DEFAULT_TYPE):
    await upd.message.reply_text(
        "SkinCoach:\n\n📸 Фото — консилиум + план\n💬 Текст — вопросы, отчёты\n\n"
        "/next — следующий день\n/status — прогресс\n/start — заново\n/help — справка")

def main():
    if not TOKEN: raise RuntimeError("TELEGRAM_BOT_TOKEN not set")
    if not API_KEY: raise RuntimeError("OPENROUTER_API_KEY not set")
    if sys.platform=="win32": asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    app=ApplicationBuilder().token(TOKEN).build()
    app.add_handler(CommandHandler("start",cmd_start))
    app.add_handler(CommandHandler("help",cmd_help))
    app.add_handler(CommandHandler("next",cmd_next))
    app.add_handler(CommandHandler("status",cmd_status))
    app.add_handler(MessageHandler(filters.PHOTO,handle_photo))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND,handle_text))
    log.info("="*50); log.info("  SkinCoach v6"); log.info("="*50)
    app.run_polling()

if __name__=="__main__": main()
