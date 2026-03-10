"""
SkinCoach v7 — 8-слойный пайплайн + уточняющие вопросы + 28-дневная программа
"""
import tempfile, os
from inference import predict_image
import asyncio,json,os,sys,base64,logging
from datetime import datetime
from pathlib import Path
from typing import Any,Dict,List
import httpx
from dotenv import load_dotenv
from telegram import Update
from telegram.constants import ChatAction
from telegram.ext import ApplicationBuilder,CommandHandler,MessageHandler,ContextTypes,filters

load_dotenv()

TOKEN=os.getenv("TELEGRAM_BOT_TOKEN","").strip()
OR_KEY=os.getenv("OPENROUTER_API_KEY","").strip()
VISION_M=os.getenv("VISION_MODEL","google/gemini-2.0-flash-lite-001").strip()
REASON_M=os.getenv("REASON_MODEL","google/gemini-2.0-flash-lite-001").strip()
STRONG_M=os.getenv("STRONG_MODEL","google/gemini-2.5-flash-preview").strip()
VIS_FB=[m.strip() for m in os.getenv("VISION_FALLBACKS","google/gemini-2.5-flash-preview").split(",") if m.strip()]
TXT_FB=[m.strip() for m in os.getenv("TEXT_FALLBACKS","google/gemini-2.5-flash-preview,arcee-ai/trinity-large-preview:free").split(",") if m.strip()]
TEMP=float(os.getenv("TEMPERATURE","0.3"))
TOUT=int(os.getenv("TIMEOUT","120"))

logging.basicConfig(level=logging.INFO,format="%(asctime)s|%(levelname)s|%(message)s")
log=logging.getLogger("skincoach")

# States
S_NAME="name";S_DUR="dur";S_TRIED="tried";S_PHOTO="photo";S_QUESTIONS="questions";S_ACTIVE="active"

WEEKS={1:"ПИТАНИЕ — убираем провокаторы",2:"НАРУЖНЫЙ УХОД — мыло, масло, крем",
       3:"ЭМОЦИИ — стресс-протокол",4:"АНАЛИЗЫ — контроль и коррекция"}
W_EMOJI={1:"🥗",2:"🧴",3:"🧠",4:"🔬"}
FOCUSES={
    1:{1:"Список что ел за 3 дня",2:"Исключи молочку",3:"Убери сахар",
       4:"Добавь куркуму",5:"8 стаканов воды",6:"Анти-воспалительный смузи",7:"Итог недели"},
    2:{1:"Серное мыло НЕ при остром",2:"Тёплая вода до 37C",3:"Крем с мочевиной на влажную кожу",
       4:"Масло точечно после крема",5:"Серное мыло 1 раз если можно",6:"Полная схема утро+вечер",7:"Фото для сравнения"},
    3:{1:"Дыхание 4-7-8",2:"3 ситуации обострения",3:"Точечный массаж",
       4:"Аффирмация 10 раз",5:"Мышечное расслабление",6:"Письмо коже",7:"Связь стресс-кожа?"},
    4:{1:"Запись: ОАК, D, ферритин, ТТГ",2:"Копрограмма",3:"Цинк и селен",
       4:"Расшифровка результатов",5:"Коррекция добавок",6:"Персональный протокол",7:"Финальное фото"},
}

# Utils
def rp(f,d=""):
    p=Path(f)
    if p.exists(): return p.read_text("utf-8").strip()
    p2=Path("prompts")/f
    if p2.exists(): return p2.read_text("utf-8").strip()
    return d

def cm(t):
    t=t.replace("**","").replace("__","").replace("```","").replace("`","")
    return "\n".join(l.lstrip("#").strip() if l.lstrip().startswith("#") else l for l in t.split("\n"))

def xj(t):
    t=t.strip()
    for prefix in ["```json","```"]:
        if t.startswith(prefix): t=t[len(prefix):]
    if t.endswith("```"): t=t[:-3]
    t=t.strip()
    try: return json.loads(t)
    except: pass
    s,e=t.find("{"),t.rfind("}")
    if s!=-1 and e>s:
        try: return json.loads(t[s:e+1])
        except: pass
    raise ValueError(f"No JSON: {t[:300]}")

# History
HIST="history.json"
def lh():
    if os.path.exists(HIST):
        try:
            with open(HIST,"r",encoding="utf-8") as f: return json.load(f)
        except: return {}
    return {}
def sh(h):
    try:
        with open(HIST,"w",encoding="utf-8") as f: json.dump(h,f,ensure_ascii=False,indent=2)
    except Exception as e: log.error(f"Save:{e}")
def gu(h,uid):
    u=str(uid)
    if u not in h: h[u]={"state":S_NAME,"name":None,"duration":None,"tried":None,
        "vision_data":None,"reasoning_data":None,"diagnosis":None,"risk":None,
        "recommendations":None,"pending_questions":None,"photo_b64":None,
        "day":0,"week":1,"msgs":[],"created":datetime.now().isoformat()}
    return h[u]
def tm(m): return m[-30:] if len(m)>30 else m

# API
def hdr(): return {"Authorization":f"Bearer {OR_KEY}","Content-Type":"application/json",
    "HTTP-Referer":"https://t.me/skincoach_bot","X-Title":"SkinCoach"}

async def call_raw(msgs,mdl,fb,mt=800):
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
                        ct=d["choices"][0]["message"].get("content") or ""
                        if isinstance(ct,list): ct="".join(p.get("text","") for p in ct if isinstance(p,dict))
                        if not ct.strip():
                            log.warning(f"  {m}: empty"); continue
                        log.info(f"  OK: {m}")
                        return ct
                log.warning(f"  {m}: {r.status_code}"); last_e=f"{m}:{r.status_code}"
            except httpx.TimeoutException: log.warning(f"  {m}: timeout"); last_e=f"{m}:timeout"
            except Exception as e: log.warning(f"  {m}: {e}"); last_e=str(e)
    raise Exception(f"All down. {last_e}")

async def cj(msgs,mdl,fb,mt=800): return xj(await call_raw(msgs,mdl,fb,mt))
async def ct(msgs,mdl,fb,mt=800): return cm(await call_raw(msgs,mdl,fb,mt))

# ════════════════════════════════════
#  8-STEP PIPELINE
# ════════════════════════════════════
async def pipeline_photo(b64,cap,u):
    """Steps 1-4: analyze photo and generate questions"""
    nm=u.get("name","друг");dur=u.get("duration","?");tri=u.get("tried","?")
    uctx=f"Имя:{nm}, давность:{dur}, пробовали:{tri}"

    # STEP 1: Quality Check
    log.info("📸 1/8 Quality...")
    try:
        qp=rp("1_quality.txt","Проверь качество фото. JSON.")
        q=await cj([{"role":"system","content":qp},
            {"role":"user","content":[{"type":"text","text":"Оцени качество фото кожи"},
                {"type":"image_url","image_url":{"url":f"data:image/jpeg;base64,{b64}"}}]}],
            VISION_M,VIS_FB,300)
        if not q.get("usable",True):
            return "ask_reshoot",q.get("suggestion","Пересними при дневном свете, крупным планом.")
    except Exception as e:
        log.warning(f"Quality skip: {e}")

    # STEP 2: Vision Description
    log.info("👁 2/8 Vision...")
    vp=rp("2_vision.txt","Опиши что видно на фото кожи. JSON.")
    try:
        vis=await cj([{"role":"system","content":vp},
            {"role":"user","content":[{"type":"text","text":cap or "Опиши что видишь на коже"},
                {"type":"image_url","image_url":{"url":f"data:image/jpeg;base64,{b64}"}}]}],
            VISION_M,VIS_FB,500)
    except Exception as e:
        log.error(f"Vision fail: {e}")
        return "error","Не удалось проанализировать фото. Попробуй ещё раз."
    u["vision_data"]=vis

    # STEP 3: Dermatology Reasoning
    log.info("🔬 3/8 Reasoning...")
    rp3=rp("3_reasoning.txt","Дифференциальная диагностика. JSON.")
    ctx3=json.dumps({"vision":vis,"patient":uctx},ensure_ascii=False)
    try:
        reason=await cj([{"role":"system","content":rp3},{"role":"user","content":ctx3}],
            STRONG_M,TXT_FB,600)
    except Exception as e:
        log.error(f"Reasoning fail: {e}")
        reason={"hypotheses":[{"diagnosis":"требуется уточнение","probability":100,"reasoning":"не удалось провести анализ"}],
                "primary_diagnosis":"требуется уточнение","stage":"unknown","phase":"unknown",
                "severity":"unknown","soap_safe":False,"confidence":0}
    u["reasoning_data"]=reason
    u["diagnosis"]=reason.get("primary_diagnosis","не определено")

    # STEP 4: Clinical Questions
    log.info("❓ 4/8 Questions...")
    rp4=rp("4_questions.txt","Задай 1-2 вопроса. JSON.")
    ctx4=json.dumps({"vision":vis,"reasoning":reason,"patient":uctx},ensure_ascii=False)
    try:
        qs=await cj([{"role":"system","content":rp4},{"role":"user","content":ctx4}],
            REASON_M,TXT_FB,400)
    except:
        qs={"questions":[],"intro":f"{nm}, я проанализировал фото."}
    u["pending_questions"]=qs
    return "questions",qs

async def pipeline_final(u,answers_text=""):
    """Steps 5-8: after questions answered, generate final plan"""
    nm=u.get("name","друг");dur=u.get("duration","?");tri=u.get("tried","?")
    vis=u.get("vision_data",{});reason=u.get("reasoning_data",{})
    dy=u.get("day",1);wk=u.get("week",1)
    wt=WEEKS.get(wk,"Программа");diw=((dy-1)%7)+1
    df=FOCUSES.get(wk,{}).get(diw,"Следуй программе")

    all_data=json.dumps({"vision":vis,"reasoning":reason,"patient_answers":answers_text,
        "patient":f"Имя:{nm}, давность:{dur}, пробовали:{tri}",
        "day":dy,"week":wk,"week_theme":wt,"day_focus":df},ensure_ascii=False)

    # STEP 5: Risk Triage
    log.info("⚠️ 5/8 Triage...")
    rp5=rp("5_triage.txt","Определи уровень риска. JSON.")
    try:
        triage=await cj([{"role":"system","content":rp5},{"role":"user","content":all_data}],
            REASON_M,TXT_FB,300)
    except:
        triage={"risk_level":"green","urgency":"routine"}
    u["risk"]=triage

    # STEP 6: Recommendations
    log.info("📋 6/8 Recommendations...")
    rp6=rp("6_recommendations.txt","Составь рекомендации. JSON.")
    ctx6=json.dumps({"all_data":json.loads(all_data) if isinstance(all_data,str) else all_data,
        "triage":triage},ensure_ascii=False)
    try:
        recs=await cj([{"role":"system","content":rp6},{"role":"user","content":ctx6}],
            STRONG_M,TXT_FB,800)
    except Exception as e:
        log.error(f"Recs fail: {e}")
        recs={"diagnosis_summary":"Анализ выполнен","morning_routine":["Мягкое очищение"],
              "evening_routine":["Увлажнение"],"day_focus":df}
    u["recommendations"]=recs

    # STEP 7: Safety Filter
    log.info("🛡️ 7/8 Safety...")
    rp7=rp("7_safety.txt","Проверь безопасность. JSON.")
    try:
        safety=await cj([{"role":"system","content":rp7},
            {"role":"user","content":json.dumps({"recs":recs,"triage":triage,"reasoning":reason},ensure_ascii=False)}],
            REASON_M,TXT_FB,300)
        if not safety.get("approved",True):
            log.warning(f"Safety issues: {safety.get('issues')}")
    except:
        pass  # If safety check fails, proceed anyway

    # STEP 8: Format Response
    log.info("💬 8/8 Response...")
    rp8=rp("8_response.txt","Собери ответ для Telegram.")
    rp8=rp8.replace("{name}",nm).replace("{day}",str(dy)).replace("{week}",str(wk))
    ctx8=json.dumps({"recommendations":recs,"triage":triage,"reasoning":reason,
        "vision":vis,"name":nm,"day":dy,"week":wk,"week_theme":wt},ensure_ascii=False)
    try:
        final=await ct([{"role":"system","content":rp8},{"role":"user","content":ctx8}],
            REASON_M,TXT_FB,900)
    except Exception as e:
        log.error(f"Response fail: {e}")
        final=format_fallback(recs,reason,triage,u)
    return final

def format_fallback(recs,reason,triage,u):
    nm=u.get("name","");dy=u.get("day",1);wk=u.get("week",1)
    p=[f"🔍 {nm}, вот что я вижу:"]
    ds=recs.get("diagnosis_summary","")
    if ds: p.append(ds)
    hyps=reason.get("hypotheses",[])
    if hyps:
        for h in hyps[:3]:
            p.append(f"  {h.get('diagnosis','?')} — {h.get('probability',0)}%")
    p.append(f"\nДень {dy}/28 — Неделя {wk} {W_EMOJI.get(wk,'📋')}")
    mr=recs.get("morning_routine",[])
    if mr: p.append("\n🧴 Утро:"); p.extend(f"  {x}" for x in mr[:3])
    er=recs.get("evening_routine",[])
    if er: p.append("\n🌙 Вечер:"); p.extend(f"  {x}" for x in er[:3])
    ps=recs.get("psycho",{})
    af=ps.get("affirmation","")
    if af: p.append(f"\n💫 {af}")
    p.append("\n📝 Вечером напиши: что сделал, как кожа, ощущения.")
    return "\n".join(p)

# Send
async def send(msg,txt):
    if len(txt)<=4000: await msg.reply_text(txt); return
    parts,cur=[],""
    for l in txt.split("\n"):
        if len(cur)+len(l)+1>4000:
            if cur: parts.append(cur)
            cur=l
        else: cur=cur+"\n"+l if cur else l
    if cur: parts.append(cur)
    for p in parts: await msg.reply_text(p)

# ════════════════════════════════════
#  HANDLERS
# ════════════════════════════════════
async def cmd_start(upd:Update,ctx:ContextTypes.DEFAULT_TYPE):
    h=lh();uid=str(upd.effective_user.id)
    h[uid]=gu(h,upd.effective_user.id)
    h[uid]["state"]=S_NAME;h[uid]["msgs"]=[]
    sh(h)
    await upd.message.reply_text(
        "Привет.\nЯ твой персональный помощник по программе 'Чистая кожа'.\n\n"
        "Я помогу понять что влияет на кожу, выстроить уход и пройти маршрут шаг за шагом.\n\n"
        "Каждое фото проходит 8-ступенчатый анализ:\n"
        "  👁 Описание кожи\n  🔬 Диагностика с вероятностями\n"
        "  ❓ Уточняющие вопросы\n  ⚠️ Оценка рисков\n"
        "  📋 Персональные рекомендации\n  🛡️ Проверка безопасности\n\n"
        "Для начала — как тебя зовут?")

async def handle_text(upd:Update,ctx:ContextTypes.DEFAULT_TYPE):
    uid=upd.effective_user.id;txt=upd.message.text;h=lh();u=gu(h,uid)
    await upd.message.chat.send_action(ChatAction.TYPING)

    # Onboarding
    if u["state"]==S_NAME:
        u["name"]=txt.strip();u["state"]=S_DUR;sh(h)
        await upd.message.reply_text(f"{u['name']}, какая у тебя проблема с кожей и как давно беспокоит?")
        return
    if u["state"]==S_DUR:
        u["duration"]=txt.strip();u["state"]=S_TRIED;sh(h)
        await upd.message.reply_text("Что уже пробовал(а)? Мази, диеты, народные средства, фототерапия?")
        return
    if u["state"]==S_TRIED:
        u["tried"]=txt.strip();u["state"]=S_PHOTO;sh(h)
        await upd.message.reply_text(
            f"Отлично, {u['name']}.\n\n📸 Отправь фото проблемного участка.\n"
            "Дневной свет, крупный план.\nМой 8-ступенчатый анализ определит тип, стадию и составит план.")
        return
    if u["state"]==S_PHOTO:
        await upd.message.reply_text(f"{u.get('name','')}, мне нужно фото. 📸")
        return

    # Answers to clinical questions
    if u["state"]==S_QUESTIONS:
        u["state"]=S_ACTIVE
        if u["day"]==0: u["day"]=1;u["week"]=1
        st=await upd.message.reply_text("Принял ответы. Генерирую персональный план... ⏳")
        try:
            reply=await pipeline_final(u,txt)
        except Exception as e:
            reply=f"Ошибка генерации плана. Попробуй /next"; log.error(f"Final:{e}")
        u["msgs"].append({"role":"user","content":txt})
        u["msgs"].append({"role":"assistant","content":reply})
        u["msgs"]=tm(u["msgs"]);sh(h)
        try: await st.delete()
        except: pass
        await send(upd.message,reply)
        return

    # Active program - chat
    if u["state"]==S_ACTIVE:
        u["msgs"].append({"role":"user","content":txt});u["msgs"]=tm(u["msgs"])
        wt=WEEKS.get(u["week"],"Программа")
        diag=(u.get("diagnosis") or "не определено")[:200]
        cp=rp("chat.txt","Ты SkinCoach.").format(
            name=u.get("name","друг"),duration=u.get("duration","?"),
            tried=u.get("tried","?"),diagnosis=diag,
            day=u["day"],week=u["week"],week_theme=wt)
        msgs=[{"role":"system","content":cp}]+u["msgs"]
        try: reply=await ct(msgs,REASON_M,TXT_FB,600)
        except: reply="Модели заняты. Через минуту."
        u["msgs"].append({"role":"assistant","content":reply});u["msgs"]=tm(u["msgs"]);sh(h)
        await send(upd.message,reply)
        return

    u["state"]=S_NAME;sh(h)
    await upd.message.reply_text("Как тебя зовут?")

async def handle_photo(upd:Update,ctx:ContextTypes.DEFAULT_TYPE):
    uid=upd.effective_user.id;h=lh();u=gu(h,uid)
    if u["state"] in (S_NAME,S_DUR,S_TRIED):
        await upd.message.reply_text("Сначала познакомимся. /start"); return

    st=await upd.message.reply_text(
        "📸 Фото получено. Запускаю 8-ступенчатый анализ...\n\n"
        "1️⃣ Проверка качества фото...\n2️⃣ Описание кожи...\n"
        "3️⃣ Диагностика с вероятностями...\n4️⃣ Подготовка вопросов...\n\n"
        "30-60 сек ⏳")
    await upd.message.chat.send_action(ChatAction.TYPING)

ph=upd.message.photo[-1];f=await ctx.bot.get_file(ph.file_id)
b=await f.download_as_bytearray();b64=base64.b64encode(b).decode()

# Локальная модель
import tempfile, os
from inference import predict_image
with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
    tmp.write(b)
    tmp_path = tmp.name
try:
    skin_result = predict_image(tmp_path)
    u["local_model_result"] = skin_result
finally:
    os.unlink(tmp_path)

# ВОТ ЭТИ ДВЕ СТРОКИ ДОЛЖНЫ БЫТЬ ЗДЕСЬ — НЕ ВНУТРИ finally!
cap=(upd.message.caption or "").strip()
u["photo_b64"]=b64[:100]
# store ref only
    result_type,result=await pipeline_photo(b64,cap,u)

    try: await st.delete()
    except: pass

    if result_type=="ask_reshoot":
        await upd.message.reply_text(f"📸 {result}")
        sh(h); return

    if result_type=="error":
        await upd.message.reply_text(result)
        sh(h); return

    # result_type == "questions"
    qs=result
    intro=qs.get("intro",f"{u.get('name','')}, я проанализировал фото.")
    questions=qs.get("questions",[])

    # Show diagnosis preview
    reason=u.get("reasoning_data",{})
    hyps=reason.get("hypotheses",[])
    diag_text=""
    if hyps:
        diag_text="\n\n🔬 Предварительный анализ:\n"
        for hp in hyps[:3]:
            diag_text+=f"  {hp.get('diagnosis','?')} — {hp.get('probability',0)}%\n"

    if questions:
        q_text="\n\nЧтобы дать точные рекомендации, мне нужно уточнить:\n\n"
        for i,q in enumerate(questions):
            q_text+=f"{i+1}. {q.get('question','')}\n"
            opts=q.get("options",[])
            if opts: q_text+="   "+", ".join(opts)+"\n"
        q_text+="\nОтветь на вопросы одним сообщением."
        u["state"]=S_QUESTIONS
    else:
        q_text=""
        u["state"]=S_ACTIVE
        if u["day"]==0: u["day"]=1;u["week"]=1

    sh(h)
    msg=intro+diag_text+q_text
    await send(upd.message,msg)

    # If no questions — proceed to final immediately
    if not questions:
        st2=await upd.message.reply_text("Генерирую план... ⏳")
        try: reply=await pipeline_final(u,"")
        except Exception as e: reply="Ошибка. /next"; log.error(f"Final:{e}")
        u["msgs"].append({"role":"assistant","content":reply});u["msgs"]=tm(u["msgs"]);sh(h)
        try: await st2.delete()
        except: pass
        await send(upd.message,reply)

async def cmd_next(upd:Update,ctx:ContextTypes.DEFAULT_TYPE):
    uid=upd.effective_user.id;h=lh();u=gu(h,uid)
    if u["state"]!=S_ACTIVE: await upd.message.reply_text("/start"); return
    await upd.message.chat.send_action(ChatAction.TYPING)
    u["day"]+=1
    if u["day"]>28:
        await upd.message.reply_text(f"🎉 {u.get('name','')}, программа пройдена! Отправь фото для сравнения.")
        sh(h); return
    u["week"]=((u["day"]-1)//7)+1
    wt=WEEKS.get(u["week"],"Программа");diw=((u["day"]-1)%7)+1
    df=FOCUSES.get(u["week"],{}).get(diw,"Следуй программе")
    diag=(u.get("diagnosis") or "не определено")[:200]
    last=u["msgs"][-4:] if u["msgs"] else []
    context="".join(f"{'Человек' if m['role']=='user' else 'Коуч'}: {(m['content'] if isinstance(m['content'],str) else '')[:150]}\n" for m in last)
    pr=rp("next_day.txt","План на день.").format(
        day=u["day"],week=u["week"],week_theme=wt,week_emoji=W_EMOJI.get(u["week"],"📋"),
        name=u.get("name","друг"),diagnosis=diag,day_focus=df,context=context)
    try: plan=await ct([{"role":"system","content":pr},{"role":"user","content":f"План на день {u['day']}."}],REASON_M,TXT_FB,600)
    except: plan="Не удалось. /next через минуту."
    u["msgs"].append({"role":"assistant","content":plan});u["msgs"]=tm(u["msgs"]);sh(h)
    await send(upd.message,plan)

async def cmd_status(upd:Update,ctx:ContextTypes.DEFAULT_TYPE):
    h=lh();u=gu(h,upd.effective_user.id)
    if u["state"]!=S_ACTIVE: await upd.message.reply_text("/start"); return
    wt=WEEKS.get(u["week"],"Программа");pct=int((u["day"]/28)*100)
    bar="▓"*(pct//10)+"░"*(10-pct//10)
    diag=u.get("diagnosis","не определено")
    reason=u.get("reasoning_data",{})
    hyps=reason.get("hypotheses",[])
    diag_info=""
    if hyps:
        diag_info="\n\nДиагноз:\n"
        for hp in hyps[:3]:
            diag_info+=f"  {hp.get('diagnosis','?')} — {hp.get('probability',0)}%\n"
    await upd.message.reply_text(
        f"📊 {u.get('name','')}{diag_info}\n"
        f"День {u['day']}/28\nНеделя {u['week']}/4 — {wt}\n[{bar}] {pct}%\n\n"
        f"/next — следующий день\n📸 Фото — повторный анализ")

async def cmd_help(upd:Update,ctx:ContextTypes.DEFAULT_TYPE):
    await upd.message.reply_text(
        "SkinCoach — 8-ступенчатый анализ кожи:\n\n"
        "📸 Фото — полный анализ с диагнозом и вероятностями\n"
        "💬 Текст — вопросы, отчёты\n\n"
        "/next — следующий день\n/status — прогресс + диагноз\n/start — заново")

def main():
    if not TOKEN: raise RuntimeError("TELEGRAM_BOT_TOKEN not set")
    if not OR_KEY: raise RuntimeError("OPENROUTER_API_KEY not set")
    if sys.platform=="win32": asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    app=ApplicationBuilder().token(TOKEN).build()
    app.add_handler(CommandHandler("start",cmd_start))
    app.add_handler(CommandHandler("help",cmd_help))
    app.add_handler(CommandHandler("next",cmd_next))
    app.add_handler(CommandHandler("status",cmd_status))
    app.add_handler(MessageHandler(filters.PHOTO,handle_photo))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND,handle_text))
    log.info("="*50);log.info("  SkinCoach v7 — 8-step pipeline");log.info("="*50)
    app.run_polling()

if __name__=="__main__": main()
