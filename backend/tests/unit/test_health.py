"""健康检查接口测试。

约束（§13.2）：探活是读接口，不需要会话令牌；
`/api/v1/health` 与 `/health` 必须返回同一负载，避免两处实现漂移。
"""

from __future__ import annotations

from fastapi import FastAPI
from starlette.testclient import TestClient

from app.api.v1 import router as router_mod


def client() -> TestClient:
    app = FastAPI()
    app.include_router(router_mod.router)
    return TestClient(app)


def test_health_ready_after_init(monkeypatch):
    # 只需非 None 即可代表服务层已初始化
    monkeypatch.setattr(router_mod, "_tasks", object())
    resp = client().get("/api/v1/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["ready"] is True
    assert body["app"] == "AudioTranscriber"


def test_health_starting_before_init(monkeypatch):
    # startup 回调未跑完时路由已可访问，用字段而非状态码区分
    monkeypatch.setattr(router_mod, "_tasks", None)
    body = client().get("/api/v1/health").json()
    assert body == {
        "status": "starting",
        "ready": False,
        "app": "AudioTranscriber",
        "version": "0.1.0",
    }


def test_health_needs_no_session_token():
    # 读接口不校验令牌（与 SSE / 音频 Range 同类）
    assert client().get("/api/v1/health").status_code == 200


def test_root_health_shares_payload():
    """main.app 的 /health 必须与 /api/v1/health 同源同值。"""
    from app import main as main_mod

    app = main_mod.app
    # 不走 with：不触发 startup，保持「未初始化」状态以便比较
    c = TestClient(app)
    assert c.get("/health").json() == c.get("/api/v1/health").json()