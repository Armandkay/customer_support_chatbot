# Customer Support Chatbot

## Overview
This project is a Customer Support Chatbot built with Python and Transformer models. The chatbot can handle customer queries, provide relevant responses, and escalate complex issues if needed.

## Features
- Answer common customer questions
- Provide information about products or services
- Escalate issues when necessary
- Log conversations for review

## Technologies Used
- Python 3.x
- Hugging Face Transformers
- TensorFlow
- NLTK / spaCy
- Flask / Gradio (optional for interface)

## Installation

1. Clone the repository:
```bash
git clone <repository_link>
cd customer_support_chatbot
````

2. Create a virtual environment:

```bash
python -m venv venv
```

3. Activate the virtual environment:

* **Windows:**

```bash
venv\Scripts\activate
```

* **Linux / macOS:**

```bash
source venv/bin/activate
```

4. Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

1. Run the chatbot:

```bash
python src/main.py
```

2. Optional: Run the Gradio web interface:

```bash
python -m app.gradio_app
```

3. Interact with the chatbot in the terminal or web interface.

## Project Structure

```
customer_support_chatbot/
│── data/                 # Datasets for training and testing
│── notebooks/            # Jupyter notebooks for exploration
│── src/                  # Scripts for preprocessing, training, and chatbot interaction
│── outputs/              # Logs, model outputs
│── checkpoints/          # Trained models (large files excluded)
│── app/                  # Gradio / Flask interface scripts
│── requirements.txt      # Python dependencies
│── README.md             # Project documentation
```
Do you want me to do that?
```
