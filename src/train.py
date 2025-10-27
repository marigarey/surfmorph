import os
import requests
import pandas as pd
import random
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader

url = "https://raw.githubusercontent.com/unimorph/eng/master/eng"
response = requests.get(url)
lines = response.text.strip().split('\n')

data = []
for line in lines:
    parts = line.split('\t')
    if len(parts) == 3:
        lemma, inflected, tags = parts
        data.append([lemma, inflected, tags])

df = pd.DataFrame(data, columns=['lemma', 'inflected_form', 'tags'])
print(f"Loaded {len(df)} examples")

train_examples = []

# pos pairs
for _, row in df.iterrows():
    train_examples.append(
        InputExample(texts=[row['lemma'], row['inflected_form']], label=1.0)
    )

# neg pairs
num_negatives = min(len(df), 5000)  # Limit negatives for speed
for _ in range(num_negatives):
    i, j = random.sample(range(len(df)), 2)
    if df.iloc[i]['lemma'] != df.iloc[j]['lemma']:
        train_examples.append(
            InputExample(texts=[df.iloc[i]['inflected_form'], df.iloc[j]['inflected_form']], label=0.0)
        )

random.shuffle(train_examples)
print(f"Created {len(train_examples)} training examples")

# persistence
os.makedirs("checkpoints", exist_ok=True)
examples_csv_path = os.path.join("checkpoints", "train_examples.csv")

with open(examples_csv_path, "w", encoding="utf-8") as f:
    f.write("text1\ttext2\tlabel\n")
    for ex in train_examples:
        t1, t2 = ex.texts
        f.write(f"{t1}\t{t2}\t{ex.label}\n")
print(f"Saved training examples to {examples_csv_path}")

model = SentenceTransformer('all-MiniLM-L6-v2')
train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)
train_loss = losses.CosineSimilarityLoss(model)

checkpoint_dir = os.path.join("checkpoints", "model_checkpoints")
os.makedirs(checkpoint_dir, exist_ok=True)

checkpoint_save_steps = 1000
checkpoint_save_total_limit = 5  # keep last N checkpoints

model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=4,
    warmup_steps=100,
    output_path='./output/unimorph-english-model',
    checkpoint_path=checkpoint_dir,
    checkpoint_save_steps=checkpoint_save_steps,
    checkpoint_save_total_limit=checkpoint_save_total_limit
)

print("Training complete! Model saved to ./output/unimorph-english-model")
print(f"Checkpoints saved under {checkpoint_dir}")