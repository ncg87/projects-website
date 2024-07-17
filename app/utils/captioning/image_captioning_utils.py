# Nessesary imports
import torch
import os
# Import the model
from utils.captioning.image_captioning import EncoderDecoder

# Relative path to the model
MODEL_PATH = './models/captioning_model.pth'

# Get device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Get the checkpoint
checkpoint = torch.load(MODEL_PATH, map_location=device)

# Get hyperparameters from the checkpoint
args = checkpoint['args']

embed_size = args['embed_size']
decoder_dim = args['decoder_dim']
attention_dim = args['attention_dim']
vocab_size = args['vocab_size']
p = args['p']

# Initalize the model
model = EncoderDecoder(
    embed_size=embed_size,
    decoder_dim=decoder_dim,
    attention_dim=attention_dim,
    vocab_size=vocab_size,
    p=p
).to(device)

# Load the trained model
model.load_state_dict(checkpoint['state_dict'])

def predict_caption(image):
    return None