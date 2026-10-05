query = 'What does the picture show?'
image_paths = ['xxx/path']

from HuatuoGPT_Vision.cli import HuatuoChatbot
bot = HuatuoChatbot("xxx/path") # loads the model
output = bot.inference(query, image_paths) # generates
print(output)