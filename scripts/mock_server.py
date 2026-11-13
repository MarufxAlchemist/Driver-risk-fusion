"""Run the mock API: `make serve` (docs/05_api_contract.md section 5).

REST needs header `X-API-Key: <token>`, the WebSocket needs `?token=<token>`.
The token is $DRIVERISK_API_KEY, default "dev-token". Docs UI: http://127.0.0.1:8000/docs
"""

from __future__ import annotations

import argparse

import uvicorn

from driverisk.serving.mock_app import app


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")  # local only: the mock has demo auth
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
