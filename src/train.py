# src/train.py
import os
import tensorflow as tf
from transformers import T5Tokenizer, TFT5ForConditionalGeneration
from src.data_prep import load_and_prepare

# --- Hyperparams (tweak if needed) ---
MODEL_NAME = "t5-small"
CSV_PATH = "data/customer_support_data.csv"
OUTPUT_DIR = "./checkpoints"
EPOCHS = 2
BATCH_SIZE = 4
LEARNING_RATE = 3e-4
MAX_INPUT_LEN = 128
MAX_TARGET_LEN = 128
# -------------------------------------

os.makedirs(OUTPUT_DIR, exist_ok=True)

def make_tf_dataset(encodings, decodings, batch_size=BATCH_SIZE):
    """
    Create a tf.data.Dataset where each element is a dict:
      { 'input_ids': ..., 'attention_mask': ..., 'labels': ... }
    This is the format TFT5ForConditionalGeneration expects.
    """
    # Convert tokenizer outputs (which are numpy) into tensors
    input_ids = tf.convert_to_tensor(encodings["input_ids"])
    attention_mask = tf.convert_to_tensor(encodings["attention_mask"])
    labels = tf.convert_to_tensor(decodings["input_ids"])

    dataset = tf.data.Dataset.from_tensor_slices({
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    })

    dataset = dataset.shuffle(1024).batch(batch_size)
    return dataset

def train():
    print("Loading and preprocessing data...")
    df, encodings, decodings, tokenizer = load_and_prepare(CSV_PATH)

    print("Loading model...")
    model = TFT5ForConditionalGeneration.from_pretrained(MODEL_NAME)

    # Use an optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)

    # Compile model with the provided compute_loss so HuggingFace TF model handles loss
    model.compile(optimizer=optimizer, run_eagerly=False)

    # Build tf.data.Dataset with labels inside the input dict
    train_ds = make_tf_dataset(encodings, decodings, batch_size=BATCH_SIZE)

    print("Starting training...")
    history = model.fit(train_ds, epochs=EPOCHS)

    # Save final checkpoint
    print(f"Saving model to {OUTPUT_DIR} ...")
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print("Training complete.")

if __name__ == "__main__":
    train()
