import pandas as pd
from transformers import T5Tokenizer

def load_and_prepare(csv_path):
    df = pd.read_csv(csv_path)

    # Fill any missing text fields to prevent tokenizer errors
    df["context"] = df["context"].fillna("")
    df["user"] = df["user"].fillna("")
    df["agent"] = df["agent"].fillna("")
    df["intent"] = df["intent"].fillna("unknown")

    # Combine context + user message (so the model gets a full prompt)
    df["input_text"] = "context: " + df["context"] + " user: " + df["user"]

    tokenizer = T5Tokenizer.from_pretrained("t5-small")

    encodings = tokenizer(
        df["input_text"].tolist(),
        padding=True,
        truncation=True,
        return_tensors="tf"
    )

    decodings = tokenizer(
        df["agent"].tolist(),
        padding=True,
        truncation=True,
        return_tensors="tf"
    )

    return df, encodings, decodings, tokenizer
