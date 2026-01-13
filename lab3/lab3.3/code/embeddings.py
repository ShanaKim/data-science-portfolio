import os
import sys
import torch
import numpy as np
from transformers import BertTokenizerFast, BertModel

# Initialize tokenizer and model globally to avoid reloading
tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

def make_word_embeddings(word_list, chunk_size=400):
    """
    Generate word embeddings for a list of words using BERT.

    Args:
    - word_list (list): List of words.
    - chunk_size (int): Number of words per chunk for processing.

    Returns:
    - torch.Tensor: Embedding matrix of shape (num_words, 768).
    """
    chunks = [word_list[i:i + chunk_size] for i in range(0, len(word_list), chunk_size)]
    all_word_embeddings = []
    global_word_offset = 0

    for chunk in chunks:
        # Tokenize with word_ids for alignment
        inputs = tokenizer(
            chunk,
            is_split_into_words=True,
            return_tensors='pt',
            padding=False,
            truncation=True,
            max_length=512
        )
        token_ids = np.array(inputs["input_ids"]).flatten()

        # Get word_ids to map tokens to words
        encoding = tokenizer(chunk, is_split_into_words=True, return_tensors=None)
        word_ids = encoding.word_ids(batch_index=0)[1:-1]  # Exclude [CLS] and [SEP]

        with torch.no_grad():
            outputs = model(**inputs)

        # Extract hidden states, excluding [CLS] and [SEP]
        hidden_state = outputs.last_hidden_state[0, 1:-1, :]

        # Group embeddings by word, handling contractions
        word_to_embeddings = {}
        i = 0
        while i < len(word_ids):
            if (
                i + 2 < len(word_ids) and
                tokenizer.convert_ids_to_tokens([token_ids[i + 1]])[0] == "'" and
                word_ids[i] is not None and
                word_ids[i + 2] == word_ids[i]
            ):
                # Handle contraction (e.g., ["doesn", "'", "t"])
                contraction_tokens = [i, i + 1, i + 2]
                word_id = global_word_offset + word_ids[i]
                if word_id not in word_to_embeddings:
                    word_to_embeddings[word_id] = []
                word_to_embeddings[word_id].extend(hidden_state[contraction_tokens])
                i += 3
            else:
                # Regular word or subword
                if word_ids[i] is not None:
                    word_id = global_word_offset + word_ids[i]
                    if word_id not in word_to_embeddings:
                        word_to_embeddings[word_id] = []
                    word_to_embeddings[word_id].append(hidden_state[i])
                i += 1

        # Average embeddings for each word in this chunk
        for word_id in word_to_embeddings:
            embeddings = torch.stack(word_to_embeddings[word_id])
            word_embedding = embeddings.mean(dim=0)
            all_word_embeddings.append((word_id, word_embedding))

        global_word_offset += len(chunk)

    # Initialize the final embedding matrix
    embedding_matrix = torch.zeros(len(word_list), 768)
    for word_id, emb in sorted(all_word_embeddings, key=lambda x: x[0]):
        embedding_matrix[word_id] = emb

    return embedding_matrix

def make_word_dict(story_list, raw_text):
    """
    Create a dictionary of word embeddings for multiple stories.

    Args:
    - story_list (list): List of story names.
    - raw_text (dict): Dictionary containing story data as word lists.

    Returns:
    - dict: Dictionary where each key is a story name and each value is the embedding matrix.
    """
    word_dict = {}
    for story in story_list:
        word_list = raw_text[story].data
        word_dict[story] = make_word_embeddings(word_list)
    return word_dict


