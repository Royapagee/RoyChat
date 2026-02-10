"""后端的主程序"""
import backend.app.load_config as load_config

config = load_config.load_config()
api_config = load_config.load_api_config()

print (api_config)