from __future__ import annotations

import argparse

import uvicorn

from .server import create_app


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the kolibrify RL data server")
    parser.add_argument("config", help="Path to rl_data_config.yaml")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind", type=str)
    parser.add_argument("--port", default=9000, help="Port to bind", type=int)
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print a short preview of samples and rewards for each /grade request",
    )
    return parser


def run(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    app = create_app(args.config, verbose=args.verbose)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    run()
