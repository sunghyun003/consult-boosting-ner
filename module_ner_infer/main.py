
import argparse
import asyncio
import uvicorn

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from module_ner_infer.core import env
from module_ner_infer.services.inference import (
    NERInferenceService,
    MeCabInferenceService,
)
from module_ner_infer.api import api_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        # Ensure all required files are present
        print("Ensuring required model ...")
        required_model_path = f"{env.PROJECT_ROOT_DIR}/models/roberta-klue-ner"
        dic_path = f"{env.DIC_PATH}"

        # Load the model
        app.state.ner_service = NERInferenceService(model_path=required_model_path)
        app.state.mecab_service = MeCabInferenceService(dic_path=dic_path)

        print(f"Model loaded from {required_model_path}")
    except Exception as exc:
        print(f"Failed to load model: {exc}")
        raise RuntimeError("Model loading failed during startup.")
    yield
    app.state.ner_service = None
    print("Cleaned up the models and released resources.")


app = FastAPI(lifespan=lifespan)
origins = env.ORIGINS_PROD if env.CURRENT_ENV == "prod" else ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(api_router)


def make_arg_parser() -> argparse.ArgumentParser:
    """
    :return: argparse.ArgumentParser
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--host",
        "-H",
        required=True,
        type=str,
        default="127.0.0.1",
        help="IP Address of the server"
    )
    parser.add_argument(
        "--port",
        "-P",
        required=True,
        type=int,
        default=8000,
        help="Port number of the server",
    )
    return parser


async def main() -> None:
    """
    :return: None
    """
    try:
        params = {
            'app_name': "main:app",
            'args': make_arg_parser().parse_args(),
            'log_level': "info",
        }
        config = uvicorn.Config(
            app=params["app_name"],
            host=params["args"].host,
            port=params["args"].port,
            log_level=params["log_level"],
            # ssl_keyfile=ssl_key_path,
            # ssl_certfile=ssl_cert_path,
        )
        print(
            f"[Server] Starting FastAPI server with app {params['app_name']} "
            f"on {params['args'].host}:{params['args'].port}"
        )
        server = uvicorn.Server(config=config)
        await server.serve()
        print("[Server] Server started successfully and is now running.")
    except Exception as e:
        print(f"[Exception][main] An unexpected exception occurred: {e}")


if __name__ == "__main__":
    asyncio.run(main())
