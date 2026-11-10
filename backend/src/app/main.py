# AudioTranscriber 本地 API 入口（V3.0 §11 / §13.1）
# 单进程、单 Uvicorn Worker，监听 127.0.0.1；API 不加载模型（推理在子进程）
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .api.v1.router import health_payload
from .api.v1.router import init as router_init, router
from .core.config import load_config
from .core.database import Database
from .core.security import (
    LocalSecurityMiddleware, allowed_hosts, attach_session_cookie, generate_session_token,
)
from .services.project_service import ProjectService, ensure_ffmpeg_on_path
from .services.task_service import TaskService

# 启动时加载配置并生成本次进程的会话令牌（§13.2：重启即失效）
BOOT_CFG = load_config()
SESSION_TOKEN = generate_session_token()

# 只允许本地 Origin（§13.2：禁 0.0.0.0 / CORS 不用 *）
ALLOWED_ORIGINS = ["http://127.0.0.1:8730", "http://localhost:8730"]

app = FastAPI(title="AudioTranscriber", version="0.1.0")

# Host 白名单 + 写接口令牌（§13.2）；先于 CORS 注册即可，二者独立
app.add_middleware(
    LocalSecurityMiddleware,
    hosts=allowed_hosts(BOOT_CFG.port),
    token=SESSION_TOKEN,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)


@app.on_event("startup")
def startup() -> None:
    cfg = BOOT_CFG
    ensure_ffmpeg_on_path(cfg)  # 捆绑 FFmpeg 注入进程 PATH
    db = Database(cfg.paths.database / "app.db")
    projects = ProjectService(cfg, db)
    tasks = TaskService(cfg, db)
    recovered = tasks.recover_interrupted()  # §8.4 遗留任务标记中断
    router_init(cfg, projects, tasks)
    app.state.cfg, app.state.db, app.state.tasks = cfg, db, tasks
    if recovered:
        print(f"[启动] 已将 {recovered} 个遗留 PROCESSING 任务标记为 APP_INTERRUPTED")


@app.on_event("shutdown")
def shutdown() -> None:
    """停止常驻 ASR worker，释放其占用的 GPU 显存。"""
    tasks = getattr(app.state, "tasks", None)
    if tasks is not None:
        tasks.close()


@app.get("/health")
def health() -> dict:
    """健康检查：start.bat 等待此接口成功后再打开浏览器。

    与 `/api/v1/health` 共用同一负载；`ready=false` 表示服务层尚未初始化。
    """
    return health_payload()


# ---- 前端静态托管（§13.1：start.bat 打开浏览器直接落到工作台页面） ----
# 优先用 dist 生产构建；无构建产物时返回提示（开发期走 vite dev server）
_FRONTEND_DIST = Path(__file__).resolve().parents[3] / "frontend" / "dist"

# index.html 指向带 hash 的 assets：HTML 本身不能缓存，否则前端更新后
# 浏览器仍用旧 chunk（用户已遇到「页面逻辑不生效」）。assets 文件名带 hash，可长缓存。
_NO_CACHE = {"Cache-Control": "no-cache, no-store, must-revalidate"}

if (_FRONTEND_DIST / "index.html").is_file():
    # 静态资源（js/css/图标）走缓存友好的 StaticFiles
    app.mount("/assets", StaticFiles(directory=_FRONTEND_DIST / "assets"), name="assets")

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        # 页面响应下发会话 Cookie（§13.2）：浏览器同源请求自动携带，保护写接口
        return attach_session_cookie(
            FileResponse(_FRONTEND_DIST / "index.html", headers=_NO_CACHE), SESSION_TOKEN)

    @app.get("/{full_path:path}", include_in_schema=False)
    def spa_fallback(full_path: str) -> FileResponse:
        """SPA 路由回退：非 /api、非静态文件的路径统一返回 index.html（前端路由接管）"""
        candidate = _FRONTEND_DIST / full_path
        if full_path and candidate.is_file():
            return FileResponse(candidate)
        return attach_session_cookie(
            FileResponse(_FRONTEND_DIST / "index.html", headers=_NO_CACHE), SESSION_TOKEN)
else:
    @app.get("/", include_in_schema=False)
    def index_hint() -> dict:
        return {
            "hint": "前端尚未构建：请先在 frontend/ 执行 npm run build，或使用 vite dev server 开发",
        }
