from sentence_transformers import SentenceTransformer, InputExample, losses


model = SentenceTransformer('all-MiniLM-L6-v2')

# Access the tokenizer
tokenizer = model.tokenizer

# Examples of words to tokenize
words = [
    "running",           # Common word
    "run",
    "####ing",
    "nning",
    "unhappiness",       # Word with prefix/suffix
    "antidisestablishmentarianism",  # Very long word
    "COVID-19",          # Mixed alphanumeric
    "machine-learning",  # Hyphenated
    "tokenization",      # Technical term
    "preprocessing",     # Compound word 5218 364 
    "biochemistry"
]

print("How the tokenizer splits words into sub-word tokens:\n")
print("="*70)

for word in words:
    # Tokenize the word
    tokens = tokenizer.tokenize(word)
    
    # Get token IDs
    token_ids = tokenizer.encode(word, add_special_tokens=False)
    
    # Convert back to see what tokens represent
    decoded_tokens = [tokenizer.decode([tid]) for tid in token_ids]
    
    print(f"\nWord: '{word}'")
    print(f"  Tokens: {tokens}")
    print(f"  Token IDs: {token_ids}")
    print(f"  Decoded: {decoded_tokens}")

# Compare: words not in vocabulary vs. common words
# print("\n\n" + "="*70)
# print("Rare vs. Common words:\n")

# test_cases = [
#     ("hello", "Common word"),
#     ("antidisestablishmentarianism", "Rare long word"),
#     ("biochemistry", "Technical compound word"),
#     ("supercalifragilisticexpialidocious", "Made-up word"),
# ]

# for word, description in test_cases:
#     tokens = tokenizer.tokenize(word)
#     token_ids = tokenizer.encode(word, add_special_tokens=False)
#     decoded_tokens = [tokenizer.decode([tid]) for tid in token_ids]
#     print(f"{description}: '{word}'")
#     print(f"  → Split into {len(decoded_tokens)} tokens: {decoded_tokens}\n")

# # Show vocabulary lookup
# print("="*70)
# print("Vocabulary information:\n")
# print(f"Total vocabulary size: {tokenizer.vocab_size:,}")
# print(f"Max sequence length: {tokenizer.model_max_length}")

# # Check if specific words are in vocabulary
# check_words = ["machine", "learning", "antidisestablishmentarianism"]
# print("\nVocabulary lookup:")
# for word in check_words:
#     token_id = tokenizer.convert_tokens_to_ids(word)
#     in_vocab = token_id != tokenizer.unk_token_id
#     print(f"  '{word}': {'✓ in vocab' if in_vocab else '✗ not in vocab'}")