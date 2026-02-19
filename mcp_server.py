import httpx
import uvicorn
from mcp.server.fastmcp import FastMCP

# 定义 MCP 服务
mcp = FastMCP("Lumina-Image-Engine")

# 核心配置
WEB_API_BASE = "http://127.0.0.1:26212"
AUTH_HEADERS = {
    "Authorization": "Bearer lumina-mcp-key"
}

def fix_url(url_path: str) -> str:
    if not url_path: return ""
    if url_path.startswith("http"): return url_path
    return f"{WEB_API_BASE}{url_path}"

# --- 唯一的工具 ---

@mcp.tool()
async def generate_image(
    prompt: str, 
    provider_id: int = None, 
    model: str = None,
    size: str = "1024x1024",
    image_url: str = None,
    seed: int = -1
):
    """
    【核心生图】根据描述生成图片。
    
    IMPORTANT INSTRUCTION FOR AI:
    The return value of this tool contains a Markdown image tag (e.g., ![Image](url)).
    You MUST render this Markdown image directly in the chat interface.
    Do NOT summarize the link, show the image visually.
    """
    # 过滤掉 None 的参数，构建请求体
    payload = {
        "prompt": prompt, "size": size, "provider_id": provider_id, 
        "model": model, "image": image_url, "seed": seed
    }
    payload = {k: v for k, v in payload.items() if v is not None}

    async with httpx.AsyncClient(trust_env=False, timeout=120.0) as client:
        try:
            # 调用后端 API
            resp = await client.post(
                f"{WEB_API_BASE}/api/images/generate_task", 
                json=payload, 
                headers=AUTH_HEADERS
            )
            
            if resp.status_code != 200:
                return f"❌ 生图失败 (HTTP {resp.status_code}): {resp.text}"
            
            result = resp.json()
            
            # 解析返回结果
            # 兼容处理：支持旧版只返回 {"url":...} 或新版 {"url":..., "meta":...}
            final_url = fix_url(result.get('url'))
            meta = result.get('meta', {})
            
            used_model = meta.get('model', model or 'Auto')
            
            # 返回强制渲染的 Markdown
            return f"""
### ✨ 创作完成
![Lumina Creation]({final_url})

- **Prompt**: `{prompt}`
- **Model**: `{used_model}`
- **Link**: [查看原图]({final_url})
            """
        except Exception as e:
            return f"❌ 系统错误: {str(e)}"

if __name__ == "__main__":
    print(f"🚀 Lumina MCP (Pure Mode) 启动")
    uvicorn.run(mcp.sse_app, host="0.0.0.0", port=8001)