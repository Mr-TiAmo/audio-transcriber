"""本地安全中间件（V3.0 §13.2）。

两道防线，均针对「本机服务被外部网页滥用」这一威胁模型：

1. **Host 白名单**：防 DNS rebinding。攻击者域名解析到 127.0.0.1 后，
   浏览器发来的请求 ``Host`` 头是攻击者域名；只放行本机实际监听的
   ``127.0.0.1:{port}`` / ``localhost:{port}``，其余一律 403。
2. **本地会话令牌**：启动时生成随机 token，通过 ``SameSite=Strict`` 的
   HttpOnly Cookie 下发给页面；写接口（POST/PUT/PATCH/DELETE）校验该
   Cookie。跨站请求（攻击者页面诱导浏览器发起）不会携带本 Cookie，
   即使绕过 CORS 预检也拿不到写权限。读接口（含 SSE / 音频 Range）不校验，
   保证播放器与事件流不受影响。

令牌只存内存：重启即失效，页面刷新时 ``GET /`` 会重新下发，符合
「启动时生成随机本地会话令牌」的设计语义。
"""

from __future__ import annotations

import secrets
from http.cookies import SimpleCookie

from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.types import ASGIApp, Receive, Scope, Send

# 需要会话令牌保护的写方法
WRITE_METHODS = frozenset({"POST", "PUT", "PATCH", "DELETE"})

# Cookie 名（前端无需读取：HttpOnly + 浏览器同源自动携带）
SESSION_COOKIE = "at_session"


def generate_session_token() -> str:
    """启动时调用：256-bit 随机令牌，urlsafe 编码。"""
    return secrets.token_urlsafe(32)


def allowed_hosts(port: int) -> frozenset[str]:
    """本机实际监听的 Host 白名单（含显式端口，防默认端口混淆）。"""
    return frozenset({f"127.0.0.1:{port}", f"localhost:{port}"})


def parse_cookie(header: str, name: str) -> str | None:
    """从 Cookie 头取指定键；坏头返回 None 而非抛异常。"""
    cookie = SimpleCookie()
    try:
        cookie.load(header)
    except Exception:
        return None
    morsel = cookie.get(name)
    return morsel.value if morsel else None


def session_cookie_value(token: str) -> str:
    """Set-Cookie 值：Strict 防跨站携带；HttpOnly 防页面脚本读取转发。"""
    return f"{SESSION_COOKIE}={token}; Path=/; HttpOnly; SameSite=Strict"


class LocalSecurityMiddleware:
    """纯 ASGI 中间件：Host 校验 + 写接口令牌校验（§13.2）。"""

    def __init__(self, app: ASGIApp, hosts: frozenset[str], token: str) -> None:
        self.app = app
        self.hosts = hosts
        self.token = token

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        headers = {k.decode("latin-1"): v.decode("latin-1") for k, v in scope["headers"]}

        # 防线 1：Host 白名单（DNS rebinding）
        host = headers.get("host", "")
        if host not in self.hosts:
            response = JSONResponse(
                {"detail": "Host 不在本机白名单内"}, status_code=403,
            )
            await response(scope, receive, send)
            return

        # 防线 2：写接口必须携带本次会话令牌
        if scope["method"] in WRITE_METHODS:
            cookie_header = headers.get("cookie", "")
            if parse_cookie(cookie_header, SESSION_COOKIE) != self.token:
                response = JSONResponse(
                    {"detail": "缺少本地会话令牌，请从应用页面发起操作"},
                    status_code=401,
                )
                await response(scope, receive, send)
                return

        await self.app(scope, receive, send)


def attach_session_cookie(response: Response, token: str) -> Response:
    """页面响应下发会话 Cookie（GET / 与 SPA 回退共用）。"""
    response.headers.append("Set-Cookie", session_cookie_value(token))
    return response


def request_has_token(request: Request, token: str) -> bool:
    """供路由层显式校验（中间件已覆盖写方法；保留给 SSE 等可选加固）。"""
    return parse_cookie(request.headers.get("cookie", ""), SESSION_COOKIE) == token
