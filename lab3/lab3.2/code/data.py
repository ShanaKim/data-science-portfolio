import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import random
import pickle
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizerFast
from train_encoder import train_bert
from encoder import Encoder
from gensim.models import KeyedVectors

# --- DATASET ---
class TextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_len=32):
        self.encodings = tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=max_len,
            return_tensors="pt"
        )

    def __len__(self):
        return len(self.encodings["input_ids"])

    def __getitem__(self, idx):
        return {
            "input_ids": self.encodings["input_ids"][idx],
            "token_type_ids": self.encodings["token_type_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx]
        }

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
    path_to_data = "/ocean/projects/mth240012p/shared/data"
    raw_path = os.path.join(path_to_data, "raw_text.pkl")

    with open(raw_path, "rb") as f:
        raw_text = pickle.load(f)

    print(f"opened {raw_path!r}")

    subj2 = set(os.path.splitext(f)[0] for f in os.listdir(os.path.join(path_to_data, "subject2")) if f.endswith(".npy"))
    subj3 = set(os.path.splitext(f)[0] for f in os.listdir(os.path.join(path_to_data, "subject3")) if f.endswith(".npy"))
    valid_stories = sorted(list(set(raw_text.keys()) & subj2 & subj3))

    random.seed(42)
    random.shuffle(valid_stories)
    split_idx = int(0.7 * len(valid_stories))
    train_stories = sorted(valid_stories[:split_idx])
    val_stories = sorted(valid_stories[split_idx:])

    texts = [" ".join(raw_text[s].data) for s in train_stories]
    val_texts = [" ".join(raw_text[s].data) for s in val_stories]

    # --- Manual hyperparameters ---
    hidden_size = 128
    num_layers = 4
    dropout = 0.1
    lr = 5e-4
    batch_size = 2
    num_heads = 4
    intermediate_size = 512
    max_len = 32
    epochs = 10

    # --- Create informative tag ---
    tag = f"hs{hidden_size}_nl{num_layers}_do{int(dropout*10)}_lr{lr:.0e}_ep{epochs}"

    # --- Prepare dataset and dataloaders ---
    dataset = TextDataset(texts, tokenizer, max_len=max_len)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    val_dataset = TextDataset(val_texts, tokenizer, max_len=max_len)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # --- Initialize model ---
    model = Encoder(
        vocab_size=tokenizer.vocab_size,
        hidden_size=hidden_size,
        num_heads=num_heads,
        num_layers=num_layers,
        intermediate_size=intermediate_size,
        max_len=max_len,
        dropout=dropout
    )

    # --- Train model ---
    trained_model = train_bert(
        model,
        dataloader,
        tokenizer,
        val_dataloader=val_dataloader,
        epochs=epochs,
        lr=lr,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        tag=tag
    )

    # --- Save embeddings ---
    os.makedirs("../embeddings", exist_ok=True)

    embedding_weights = trained_model.token_emb.weight.data.cpu().numpy()
    vocab = tokenizer.get_vocab()
    id_to_token = {id: token for token, id in vocab.items()}

    vector_size = embedding_weights.shape[1]
    kv = KeyedVectors(vector_size)
    kv.add_vectors([id_to_token[i] for i in range(len(id_to_token))], embedding_weights)

    embedding_path = f"../embeddings/bert_embeddings_{tag}.bin"
    kv.save_word2vec_format(embedding_path, binary=True)
    print(f"Saved embeddings to {embedding_path}")

    # --- Save hyperparameters and final loss ---
    result = {
        'hidden_size': hidden_size,
        #'num_heads': num_heads,
        'num_layers': num_layers,
        #'intermediate_size': intermediate_size,
        'dropout': dropout,
        'lr': lr,
        #'batch_size': batch_size,
        #'max_len': max_len,
        #'epochs': epochs,
        'final_val_loss': trained_model.final_val_loss,
        'min_val_loss': trained_model.min_val_loss,
        'avg_val_loss_last5': trained_model.avg_val_loss_last5
    }

    csv_path = "../embeddings/tuning_results_epochs10.csv"
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        df = pd.concat([df, pd.DataFrame([result])], ignore_index=True)
    else:
        df = pd.DataFrame([result])

    df.to_csv(csv_path, index=False)
    print(f"Saved tuning results to {csv_path}")

