import discord
from discord.ext import commands
from gradio_client import Client

intents = discord.Intents.default()
intents.message_content = True  
bot = commands.Bot(command_prefix='!', intents=intents)

@bot.event
async def on_ready():
    print(f'Logged in as {bot.user.name}')

@bot.command()
async def ping(ctx):
    await ctx.send('Pong!')

def get_response_from_gradio(input_text):
    client = Client("ableian/scgpt")
    
    response = client.predict(input_text, api_name="/generate_recipe")
    
    return response

@bot.command()
async def ask(ctx, *, question):
    response = get_response_from_gradio(question)
    await ctx.send(response)

bot.run('YOUR_BOT_TOKEN')