# Nessesary imports
import torch
import os
import torchvision.transforms as transforms
import PIL
# Import the model
from utils.captioning.image_captioning import EncoderDecoder
from dataset import Vocabulary

# Relative path to the model
MODEL_PATH = './models/best_captioning_model.pt'
VOCAB_PATH = './utils/captioning/vocab.pt'

# Get device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Get the checkpoint
checkpoint = torch.load(MODEL_PATH, map_location=device)
# Get hyperparameters from the checkpoint
args = checkpoint['args']
# Get the hyperparameters
embed_size = args['embed_size']
decoder_dim = args['decoder_dim']
attention_dim = args['attention_dim']
vocab_size = args['vocab_size']
p = args['p']
# Get the vocabulary
vocab = torch.load(VOCAB_PATH)['vocabulary']

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

# Define the inference transformation
transform = transforms.Compose(
    [
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        # For some reason the images given are already normalized
        #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

def predict_caption(image):
    # Read the image using PIL and transform to PyTorch
    image = PIL.Image.open(image).convert('RGB')
    image = transform(image).to(device)
    # Transform the image for inference and predict the caption
    caption = model.caption_image(image, vocab)
    # Convert the list of words to a string
    caption = ' '.join(caption[:-1])
    # Return caption
    return caption