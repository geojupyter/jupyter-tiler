from typing import Any

import pytest


@pytest.fixture
def jp_server_config(jp_server_config: Any) -> dict[str, Any]:
    return {
        "ServerApp": {
            "jpserver_extensions": {
                "jupyter_tiler": True,
            },
        },
    }
