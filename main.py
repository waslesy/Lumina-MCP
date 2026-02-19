import os
import time
import asyncio
import re
import webbrowser
import base64
import json
from threading import Timer
from datetime import datetime
from typing import Optional, List, Dict, Any
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Depends, Security
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from sqlmodel import SQLModel, Field, Session, select, create_engine
import httpx
from mcp.server.fastmcp import FastMCP

# --- 1. 配置 ---
PORT = 26212
UPLOAD_DIR = "static/uploads"
DB_FILE = "lumina.db"
MASTER_KEY = os.getenv("LUMINA_MCP_KEY", "lumina-mcp-key")

# --- 2. 数据库 ---
class Provider(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    name: str
    provider_type: str 
    base_url: str
    api_key: str
    quota_limit: int = 100
    quota_used: int = 0
    is_active: bool = True
    models: str = Field(default="[]") 

class ImageMeta(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    filename: str
    url: str
    prompt: str
    provider_type: str
    created_at: datetime = Field(default_factory=datetime.now)

# --- 3. 初始化 ---
engine = create_engine(f"sqlite:///{DB_FILE}", connect_args={"check_same_thread": False})
security = HTTPBearer()

def init_db():
    SQLModel.metadata.create_all(engine)
    os.makedirs(UPLOAD_DIR, exist_ok=True)

@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    print(f"🚀 Lumina-Temple Running | Master Key: {MASTER_KEY}")
    yield

app = FastAPI(lifespan=lifespan)
mcp = FastMCP("Lumina-Temple")

# --- 4. 鉴权与工具 ---
def verify_key(credentials: HTTPAuthorizationCredentials = Security(security)):
    if credentials.credentials != MASTER_KEY:
        raise HTTPException(status_code=401, detail="密钥错误")
    return credentials.credentials

def sanitize_url_str(url: str, p_type: str) -> str:
    url = url.strip().rstrip('/')
    if not url:
        if p_type == "gitee": url = "https://ai.gitee.com/v1"
        elif p_type == "volcengine": url = "https://ark.cn-beijing.volces.com/api/v3"
        else: url = "https://api.openai.com/v1"
    
    if not url.startswith("http"): url = f"https://{url}"
    if p_type == "gitee" and "api.gitee.com" in url:
        url = url.replace("api.gitee.com", "ai.gitee.com")
        if "/v1" not in url: url += "/v1"
    return url

def sanitize_url(p: Provider) -> str:
    return sanitize_url_str(p.base_url, p.provider_type)

def get_default_models(p_type: str) -> List[str]:
    if p_type == "gitee": return ["FLUX.1-dev", "Kolors", "RMBG-2.0", "Qwen-Image"]
    if p_type == "volcengine": return ["doubao-seedream-4-5-251128"]
    return ["dall-e-3"]

# [核心逻辑] 严格筛选器
def is_image_model(model_id: str, p_type: str = "") -> bool:
    mid = model_id.lower()
    
    # 针对火山引擎 (volcengine) 的绝对严格过滤
    if p_type == "volcengine":
        return "seedream" in mid

    # 其他渠道的白名单
    if "qwen-image" in mid: return True
    if "glm-image" in mid: return True
    if "seedream" in mid: return True
    
    # 明显的文本模型特征 -> 排除
    text_keywords = ['instruct', 'chat', 'embedding', 'coder', 'gpt', 'llm', 'text-generation', 'deepseek', 'qwen2', 'qwen1', 'glm-4']
    for k in text_keywords:
        if k in mid: return False

    # 明显的图片模型特征 -> 保留
    img_keywords = [
        'image', 'flux', 'diffusion', 'sdxl', 'sd3', 'kolors', 'hunyuan', 
        'midjourney', 'dall-e', 'drawing', 'painting', 'sketch', 'matting', 'face', 'edit',
        'upscale', 'rmbg', 'controlnet', 'lora', 'vision', 'stable-diffusion'
    ]
    for k in img_keywords:
        if k in mid: return True
        
    return False

# --- 通用拉取逻辑 ---
async def fetch_remote_models(base_url: str, api_key: str, p_type: str) -> List[str]:
    target_url = f"{base_url}/models"
    headers = {"Authorization": f"Bearer {api_key}"}
    print(f"☁️ Fetching models from: {target_url}")
    
    fetched = []
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(target_url, headers=headers)
            if resp.status_code == 200:
                data = resp.json()
                raw_list = []
                if "data" in data and isinstance(data["data"], list):
                    raw_list = [item["id"] for item in data["data"] if "id" in item]
                
                fetched = [m for m in raw_list if is_image_model(m, p_type)]
    except Exception as e:
        print(f"Fetch failed: {e}")
        
    if not fetched:
        if p_type == "volcengine":
            fetched = get_default_models("volcengine")
            
    return fetched

async def poll_task(client, url, headers):
    print(f"🔄 轮询任务: {url}")
    for _ in range(30):
        await asyncio.sleep(2)
        try:
            r = await client.get(url, headers=headers)
            if r.status_code != 200: continue
            d = r.json()
            status = d.get("status") or d.get("task", {}).get("status")
            if status in ["SUCCESS", "TASK_STATUS_SUCCEED", "succeeded"]:
                if "results" in d and d["results"]: return d["results"][0]["url"]
                if "images" in d and d["images"]: return d["images"][0].get("image_url") or d["images"][0].get("url")
                if "output" in d: return d["output"].get("url")
            if status in ["FAILED", "TASK_STATUS_FAILED"]:
                raise ValueError(f"生成失败: {d}")
        except Exception as e: print(f"Polling error: {e}")
    raise TimeoutError("任务超时")

# --- 5. 核心生成 (含智能纠错) ---
async def core_generate(prompt: str, size: str, model_hint: str = None, provider_id: int = None) -> Dict[str, Any]:
    # 1. 基础清洗
    prompt = prompt.strip()
    size = size.strip()
    
    # 2. 过滤无效模型词汇 (防止 LLM 传入 "Auto", "None" 等)
    if model_hint:
        model_hint = model_hint.strip()
        if model_hint.lower() in ["auto", "default", "none", "", "null"]: 
            model_hint = None

    with Session(engine) as session:
        provider = None
        # 策略A: 指定ID
        if provider_id:
            provider = session.get(Provider, provider_id)
        
        # 策略B: 遍历查找包含该模型的渠道 (支持忽略大小写查找)
        if not provider and model_hint:
            all_active = session.exec(select(Provider).where(Provider.is_active == True)).all()
            for p in all_active:
                try:
                    p_models = json.loads(p.models)
                    # 🔍 智能匹配：不区分大小写查找
                    for m in p_models:
                        if m.lower() == model_hint.lower():
                            provider = p
                            model_hint = m # 修正为正确的大小写 (例如 flux.1-dev -> FLUX.1-dev)
                            break
                    if provider: break
                except: pass

        # 策略C: 默认渠道
        if not provider:
            provider = session.exec(select(Provider).where(Provider.is_active == True)).first()
        
        if not provider: raise HTTPException(500, "无可用渠道 (No active provider)")
        
        # 3. 最终确定模型
        real_model = model_hint
        p_models = []
        try:
            p_models = json.loads(provider.models)
        except: pass

        # 如果指定了模型但大小写不对，或者不在列表中，尝试再次匹配
        if real_model:
            matched = False
            for m in p_models:
                if m.lower() == real_model.lower():
                    real_model = m
                    matched = True
                    break
            # 如果没匹配上，且该渠道有模型，默认用第一个，防止发给 Gitee 报错
            if not matched and p_models:
                print(f"⚠️ 模型 '{model_hint}' 不在渠道 '{provider.name}' 支持列表中，自动回退到 '{p_models[0]}'")
                real_model = p_models[0]
        
        # 最后的兜底
        if not real_model:
            real_model = p_models[0] if p_models else ("FLUX.1-dev" if provider.provider_type == "gitee" else "dall-e-3")

        # Volcengine 修正
        if provider.provider_type == "volcengine" and "4-5" in real_model and size == "1024x1024":
            size = "2K"

        base_url = sanitize_url(provider)
        headers = {"Authorization": f"Bearer {provider.api_key}", "Content-Type": "application/json"}

        print(f"🚀 Generating: Provider=[{provider.name}] Model=[{real_model}] Size=[{size}]")

        async with httpx.AsyncClient(timeout=120) as client:
            final_url = ""
            try:
                if provider.provider_type == "gitee":
                    api_url = f"{base_url}/images/generations"
                    payload = {"model": real_model, "prompt": prompt, "size": size, "n": 1, "response_format": "url"}
                    resp = await client.post(api_url, json=payload, headers=headers)
                    if resp.status_code != 200: 
                        print(f"❌ Gitee Payload: {payload}") 
                        raise HTTPException(400, f"Gitee Error ({resp.status_code}): {resp.text}")
                    
                    data = resp.json()
                    if "task_id" in data:
                        q_url = f"{base_url}/async/task-result?task_id={data['task_id']}"
                        final_url = await poll_task(client, q_url, headers)
                    elif "data" in data and data["data"]:
                        item = data["data"][0]
                        final_url = item.get("url") or item.get("image_url") or item.get("b64_json")
                    else: raise ValueError(f"Gitee无法解析: {data}")

                else:
                    api_url = f"{base_url}/images/generations"
                    payload = {"model": real_model, "prompt": prompt, "size": size}
                    if provider.provider_type == "volcengine": payload["watermark"] = False

                    resp = await client.post(api_url, json=payload, headers=headers)
                    if resp.status_code != 200: raise HTTPException(400, f"Upstream Error: {resp.text}")
                    d = resp.json()
                    
                    if "task_id" in d:
                        q_url = f"{base_url}/async/task-result?task_id={d['task_id']}"
                        final_url = await poll_task(client, q_url, headers)
                    else:
                        final_url = d.get("data", [{}])[0].get("url") or d.get("url")

                if not final_url: raise ValueError("未能获取图片URL")
                
                img = ImageMeta(filename=f"gen_{int(time.time())}.png", url=final_url, prompt=prompt, provider_type=provider.name)
                provider.quota_used += 1
                session.add(provider); session.add(img); session.commit()
                
                return {
                    "url": final_url,
                    "meta": { "model": real_model, "size": size, "provider": provider.name }
                }
            except HTTPException as he: raise he
            except Exception as e: raise HTTPException(500, f"Core Generate Error: {str(e)}")

# --- 6. API 路由 ---
@app.get("/api/providers/", dependencies=[Depends(verify_key)])
def get_p():
    with Session(engine) as session: return session.exec(select(Provider)).all()

@app.post("/api/providers/", dependencies=[Depends(verify_key)])
def add_p(p: Provider):
    p.base_url = sanitize_url(p)
    if not p.models or p.models == "[]":
        p.models = json.dumps(get_default_models(p.provider_type))
    with Session(engine) as session: session.add(p); session.commit(); session.refresh(p); return p

@app.put("/api/providers/{id}", dependencies=[Depends(verify_key)])
def update_p(id: int, p: Provider):
    p.base_url = sanitize_url(p)
    with Session(engine) as session:
        db_p = session.get(Provider, id)
        if db_p:
            for k,v in p.model_dump(exclude_unset=True).items(): setattr(db_p, k, v)
            session.add(db_p); session.commit(); return db_p
        raise HTTPException(404)

@app.post("/api/providers/{id}/refresh_models", dependencies=[Depends(verify_key)])
async def refresh_models(id: int):
    with Session(engine) as session:
        p = session.get(Provider, id)
        if not p: raise HTTPException(404)
        base = sanitize_url(p)
        fetched_models = await fetch_remote_models(base, p.api_key, p.provider_type)
        if not fetched_models:
            current = json.loads(p.models) if p.models else []
            fetched_models = current if current else get_default_models(p.provider_type)
        p.models = json.dumps(fetched_models)
        session.add(p); session.commit(); session.refresh(p)
        return {"message": f"Updated models: {len(fetched_models)}", "models": fetched_models}

class TempFetchReq(BaseModel):
    base_url: str
    api_key: str
    provider_type: str

@app.post("/api/providers/temp_fetch_models", dependencies=[Depends(verify_key)])
async def temp_fetch_models(req: TempFetchReq):
    real_url = sanitize_url_str(req.base_url, req.provider_type)
    fetched_models = await fetch_remote_models(real_url, req.api_key, req.provider_type)
    if not fetched_models:
        if req.provider_type == "volcengine": fetched_models = get_default_models("volcengine")
        else: return {"message": "⚠️ 未检测到图片模型，请检查 Key 或 URL", "models": []}
    return {"message": f"成功检测到 {len(fetched_models)} 个模型", "models": fetched_models}

@app.delete("/api/providers/{id}", dependencies=[Depends(verify_key)])
def del_p(id: int):
    with Session(engine) as session:
        p = session.get(Provider, id)
        if p: session.delete(p); session.commit(); return {"ok": True}
        raise HTTPException(404)

class TestReq(BaseModel):
    model: Optional[str] = None

@app.post("/api/providers/{id}/test", dependencies=[Depends(verify_key)])
async def test_prov(id: int, req: TestReq):
    with Session(engine) as session:
        p = session.get(Provider, id)
        if not p: raise HTTPException(404)
        base = sanitize_url(p)
        headers = {"Authorization": f"Bearer {p.api_key}", "Content-Type": "application/json"}
        async with httpx.AsyncClient(timeout=8) as client:
            try:
                test_url = f"{base}/images/generations"
                resp = await client.post(test_url, json={}, headers=headers)
                if resp.status_code == 400: return {"message": f"✅ {p.name} 连接成功 (鉴权通过)"}
                if resp.status_code == 401 or resp.status_code == 403: return {"message": f"❌ 鉴权失败"}
                url = f"{base}/models"
                resp = await client.get(url, headers=headers)
                if resp.status_code == 200: return {"message": f"✅ {p.name} 连接正常"}
                if resp.status_code == 401: return {"message": "❌ 鉴权失败: API Key 无效"}
                return {"message": f"⚠️ 连接异常 (HTTP {resp.status_code})"}
            except Exception as e: return {"message": f"⚠️ 网络错误: {str(e)}"}

@app.post("/api/images/generate_task", dependencies=[Depends(verify_key)])
async def api_gen(req: dict):
    try:
        result = await core_generate(
            req['prompt'], 
            req.get('size', '1024x1024'), 
            model_hint=req.get('model'), 
            provider_id=req.get('provider_id')
        )
        return result
    except Exception as e: raise HTTPException(500, str(e))

@app.get("/api/images/", dependencies=[Depends(verify_key)])
def get_i(limit: int = 50, offset: int = 0):
    with Session(engine) as session: 
        return session.exec(select(ImageMeta).order_by(ImageMeta.created_at.desc()).offset(offset).limit(limit)).all()

@app.delete("/api/images/{id}", dependencies=[Depends(verify_key)])
def del_i(id: int):
    with Session(engine) as session:
        img = session.get(ImageMeta, id)
        if img: session.delete(img); session.commit()
        return {"ok": True}

@app.post("/api/auth/verify")
async def verify_auth(token: str = Depends(verify_key)):
    return {"status": "ok"}

app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/mcp", mcp.sse_app())

@app.get("/")
def root(): return RedirectResponse("/static/login.html")

if __name__ == "__main__":
    import uvicorn
    Timer(1.5, lambda: webbrowser.open(f"http://127.0.0.1:{PORT}")).start()
    uvicorn.run(app, host="0.0.0.0", port=PORT)