"""后端的主程序"""
import backend.app.load_config as load_config

load_config.load_config()
load_config.load_api()