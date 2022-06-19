from os import getenv
from typing import Optional

from dotenv import load_dotenv
from discord.ext import commands

from rnnLSTM import generate_text, model

load_dotenv()
bot = commands.Bot(command_prefix='!')

@bot.event
async def on_ready():
    print("beep boop beep beep boop")

@bot.command()
async def generate(ctx, temp: Optional[float] = 0.5, *text: str):
    msg = await ctx.send("wait a sec...")
    result = await generate_text(model, "".join(text), temp=temp)
    await msg.edit(content=result)

@bot.command()
async def hello(ctx):
    await ctx.send("hello")

bot.run(getenv('TOKEN'))