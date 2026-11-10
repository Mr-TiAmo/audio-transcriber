"""本地安全中间件测试（V3.0 §13.2：Host 白名单 + 写接口令牌）。

不依赖真实应用实例：自建最小 FastAPI app 挂载中间件，
用 starlette TestClient 验证两道防线的放行/拒绝行为。
"""

from __future__ import annotations

from fastapi import FastAPI
from starlette.testclient import TestClient

from app.core.security import (
    SESSION_COOKIE,
    LocalSecurityMiddleware,
    allowed_hosts,
    generate_session_token,
    parse_cookie,
    session_cookie_value,
)

TOKEN = "test-token-abc"
HOSTS = allowed_hosts(8730)


def make_app() -> FastAPI:
    app = FastAPI()
    app.add_middleware(LocalSecurityMiddleware, hosts=HOSTS, token=TOKEN)

    @app.get("/read")
    def read() -> dict:
        return {"ok": True}

    @app.post("/write")
    def write() -> dict:
        return {"saved": True}

    @app.delete("/write")
    def delete() -> dict:
        return {"deleted": True}

    return app


def client() -> TestClient:
    # raise_server_exceptions=False：让 403/401 作为响应返回而非抛异常
    # base_url 与白名单端口一致：TestClient 默认 Host 为 testserver 会被防线 1 拒绝
    return TestClient(
        make_app(), base_url="http://127.0.0.1:8730", raise_server_exceptions=False)


# ---------------------------------------------------------------------------
# 防线 1：Host 白名单（DNS rebinding）
# ---------------------------------------------------------------------------

def test_rejects_foreign_host_on_read():
    resp = client().get("/read", headers={"Host": "evil.example.com"})
    assert resp.status_code == 403


def test_rejects_foreign_host_on_write():
    resp = client().post(
        "/write", headers={"Host": "evil.example.com"},
        cookies={SESSION_COOKIE: TOKEN},
    )
    assert resp.status_code == 403


def test_rejects_host_without_port():
    # 无端口 Host 不在白名单（白名单含显式端口，防默认端口混淆）
    resp = client().get("/read", headers={"Host": "127.0.0.1"})
    assert resp.status_code == 403


def test_allows_whitelisted_hosts():
    c = client()
    for host in ("127.0.0.1:8730", "localhost:8730"):
        assert c.get("/read", headers={"Host": host}).status_code == 200


# ---------------------------------------------------------------------------
# 防线 2：写接口会话令牌
# ---------------------------------------------------------------------------

def test_write_without_cookie_rejected():
    resp = client().post("/write")
    assert resp.status_code == 401


def test_write_with_wrong_cookie_rejected():
    resp = client().post("/write", cookies={SESSION_COOKIE: "wrong"})
    assert resp.status_code == 401


def test_write_with_valid_cookie_allowed():
    resp = client().post("/write", cookies={SESSION_COOKIE: TOKEN})
    assert resp.status_code == 200
    assert resp.json() == {"saved": True}


def test_delete_also_protected():
    c = client()
    assert c.delete("/write").status_code == 401
    assert c.delete("/write", cookies={SESSION_COOKIE: TOKEN}).status_code == 200


def test_read_needs_no_token():
    # 读接口（SSE / 音频 Range 同类）不校验令牌
    assert client().get("/read").status_code == 200


# ---------------------------------------------------------------------------
# 令牌与 Cookie 工具
# ---------------------------------------------------------------------------

def test_token_is_random_and_urlsafe():
    a, b = generate_session_token(), generate_session_token()
    assert a != b and len(a) >= 43  # 256-bit -> 43+ urlsafe 字符


def test_cookie_flags_strict_and_httponly():
    value = session_cookie_value(TOKEN)
    assert "HttpOnly" in value and "SameSite=Strict" in value
    assert value.startswith(f"{SESSION_COOKIE}={TOKEN}")


def test_parse_cookie_tolerates_garbage():
    assert parse_cookie(";;; =broken", SESSION_COOKIE) is None
    assert parse_cookie(f"{SESSION_COOKIE}={TOKEN}; other=1", SESSION_COOKIE) == TOKEN
    assert parse_cookie("", SESSION_COOKIE) is None
