import tomllib
from pathlib import Path

def load_config():
    # 定义文件路径（使用当前文件所在目录作为基准）
    file_path = Path(__file__).parent / "config" / "app.toml"

    # 读取并解析 TOML 文件
    with open(file_path, "rb") as f:
        config = tomllib.load(f)
    
    print(config)

    # 将配置存储为变量
    # [app]
    APP_NAME = config["app"]["name"]
    APP_DEBUG = config["app"]["debug"]
    APP_VERSION = config["app"]["version"]

    # [frontend]
    FRONTEND_IP = config["frontend"]["ip"]
    FRONTEND_PORT = config["frontend"]["port"]
    FRONTEND_TOKEN = config["frontend"]["token"]

    # [volume]
    VOLUME_GLOBAL = config["volume"]["global"]
    VOLUME_CHARACTER = config["volume"]["character"]
    VOLUME_BACKGROUND = config["volume"]["background"]

    # [ui]
    UI_TEXT_SPEED = config["ui"]["text_speed"]

    # [game]
    GAME_MODE = config["game"]["mode"]
    GAME_CHARACTER = config["game"]["character"]

    # [model]
    MODEL_CHAT = config["model"]["chat"]
    MODEL_TTS = config["model"]["tts"]
    MODEL_EMBEDDING = config["model"]["embedding"]
    MODEL_STT = config["model"]["stt"]

    return()

def load_api_config():
    file_path = Path(__file__).parent / "config" / "api.toml"

    # 读取并解析 TOML 文件
    with open(file_path, "rb") as f:
        config = tomllib.load(f)
    
    print(config)

    # 将配置存储为变量
    # [app]
    CHAT_URL = config["chat"]["base_url"]
    CHAT_MODELID = config["chat"]["model_id"]
    CHAT_APIKEY = config["chat"]["api_key"]
    CHAT_VISION = config["chat"]["vision"]

    CHAT_TTS_URL = config["tts"]["base_url"]
    CHAT_TTS_APIKEY = config["tts"]["api_key"]

    RAG_URL = config["rag"]["base_url"]
    RAG_APIKEY = config["rag"]["api_key"]

    SEARCH_PLATFORM = config["search"]["platform"]
    SEARCH_APIKEY = config["search"]["api_key"]

    return("{"+"CHAT_URL"+"CHAT_MODELID"+"CHAT_APIKEY"+"CHAT_VISION"+"CHAT_TTS_URL"+"CHAT_TTS_APIKEY"+"RAG_URL"+"RAG_APIKEY"+"SEARCH_PLATFORM"+"SEARCH_APIKEY"+"}")