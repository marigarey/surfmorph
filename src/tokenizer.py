from sentence_transformers import SentenceTransformer

from transformers import AutoTokenizer, AutoModel
import torch

try:
    model_name = "sentence-transformers/LaBSE"
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    
    print(f"✓ Loaded model: {model_name}\n")
    
    # Example usage
    text = "unhappiness"
    inputs = tokenizer(text, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Get embeddings
    embeddings = outputs.last_hidden_state  # [batch_size, seq_len, hidden_size]
    
    # Mean pooling to get sentence embedding
    sentence_embedding = embeddings.mean(dim=1)
    
    print(f"Input text: '{text}'")
    print(f"Tokens: {tokenizer.tokenize(text)}")
    print(f"Embedding shape: {sentence_embedding.shape}")
    print(f"Embedding (first 10 dims): {sentence_embedding[0, :10].tolist()}\n")
    
except Exception as e:
    print(f"Could not load model: {e}\n")

model = SentenceTransformer('all-MiniLM-L6-v2')

tokenizer = model.tokenizer

words = []

for word in words:
  tokens = tokenizer.tokenize(word)
  token_ids = tokenizer.encode(word, add_special_tokens=False)
  decoded_tokens = [tokenizer.decode([tid]) for tid in token_ids]
