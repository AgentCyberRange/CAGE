from __future__ import annotations

from pathlib import Path

import pytest

from cage.config import WebInspectorAuthConfig
from cage.web.app import create_app


def test_web_inspector_auth_rejects_missing_token(tmp_path: Path) -> None:
    app = create_app(
        tmp_path,
        auth=WebInspectorAuthConfig(enabled=True, token="local-secret"),
    )

    response = app.test_client().get("/")

    assert response.status_code == 401
    assert "authentication required" in response.get_data(as_text=True).lower()


def test_web_inspector_auth_accepts_bearer_token(tmp_path: Path) -> None:
    app = create_app(
        tmp_path,
        auth=WebInspectorAuthConfig(enabled=True, token="local-secret"),
    )

    response = app.test_client().get(
        "/",
        headers={"Authorization": "Bearer local-secret"},
    )

    assert response.status_code == 200


def test_web_inspector_auth_query_token_sets_cookie(tmp_path: Path) -> None:
    app = create_app(
        tmp_path,
        auth=WebInspectorAuthConfig(enabled=True, token="local-secret"),
    )
    client = app.test_client()

    first = client.get("/?token=local-secret")
    second = client.get("/")

    assert first.status_code == 200
    assert "cage_inspector_token=" in first.headers.get("Set-Cookie", "")
    assert second.status_code == 200


def test_web_inspector_auth_requires_non_empty_token(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="token"):
        create_app(
            tmp_path,
            auth=WebInspectorAuthConfig(enabled=True, token=""),
        )
