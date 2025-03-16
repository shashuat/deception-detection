import os
from dotenv import load_dotenv
import wandb

print('code/init')

load_dotenv()

wandb_token = os.getenv("WANDB_TOKEN")

if wandb_token:
    wandb.login(key=wandb_token)
    print("Logged in to Weights & Biases successfully!")
else:
    print("WANDB_TOKEN not found in .env file.")
