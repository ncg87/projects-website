import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Language token mappings
LANG_TOKEN_MAPPING = {
    'en' : '<en>',
    'fil' : '<fil>',
    'hi' : '<hi>',
    'id' : '<id>',
    'ja' : '<ja>', 
}

# Base model repository and model path
model_repo = "google/mt5-base"
MODEL_PATH = "./models/mt5_translator_best.pt"

##-- Initialize Tokenizer --##

# Download mt5 tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_repo, legacy=True)
# create a dict of the dict
special_tokens = { 'additional_special_tokens': list(LANG_TOKEN_MAPPING.values()) }
# add special tokens to the tokenizer
tokenizer.add_special_tokens(special_tokens)

##-- Intialize Model --##

# Download model
model= AutoModelForSeq2SeqLM.from_pretrained(model_repo)
# Check the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Resize the base model to so state dict will load correctly
model.resize_token_embeddings(len(tokenizer))
# Get the SSD and load trained model
model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device(device)))
model.eval()

## -- Nessary Functions --##

# Tokenizes and numericalizes input string
def encode_input_str(text, target_lang, tokenizer, seq_len,
                     lang_token_map=LANG_TOKEN_MAPPING):
  target_lang_token = lang_token_map[target_lang]

  # Tokenize and add special token for target language
  input_ids = tokenizer.encode(
      text = target_lang_token + text,
      return_tensors = 'pt',
      padding = 'max_length',
      truncation = True,
      max_length = seq_len)

  return input_ids

# Predict translation given input text
def predict(text, target_lang, tokenizer):
    # Encode the input string
    input_ids = encode_input_str(text, target_lang, tokenizer, 20)
    # Generate the output
    output = model.generate(input_ids.to(device), num_beams = 10, max_new_tokens = 30)
    # Decode the output
    translated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    # Returns a string of the translated text
    return translated_text