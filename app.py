"""
AskIt - Question & Answer Generation Web Application

A Flask application that uses a fine-tuned T5 model (PEFT) to generate
questions and answers from user-provided context text.
"""

import re
from pathlib import Path
from typing import List, Tuple

import torch
from flask import Flask, request, render_template
from peft import PeftModel
from transformers import T5ForConditionalGeneration, T5Tokenizer

# Configuration
MODEL_PATH = Path("./QA_flan_Peft_Model")
BASE_MODEL_NAME = "t5-small"
MIN_WORDS_FOR_INPUT = 3
MAX_INPUT_LENGTH = 512
MAX_OUTPUT_LENGTH = 128
NUM_BEAMS = 5

app = Flask(__name__)

# Global model state (loaded at startup)
_model = None
_tokenizer = None
_device = None


def _get_device() -> str:
    """Determine the best available compute device (CUDA or CPU)."""
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_model():
    """
    Load the fine-tuned PEFT model and tokenizer.

    Returns:
        tuple: (model, tokenizer) or (None, None) on failure.

    Raises:
        FileNotFoundError: If the model directory does not exist.
    """
    global _model, _tokenizer, _device

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model directory not found: {MODEL_PATH}")

    try:
        base_model = T5ForConditionalGeneration.from_pretrained(BASE_MODEL_NAME)
        _model = PeftModel.from_pretrained(base_model, str(MODEL_PATH))
        _tokenizer = T5Tokenizer.from_pretrained(str(MODEL_PATH))
        _device = _get_device()
        _model.to(_device)
        return _model, _tokenizer
    except Exception as e:
        raise RuntimeError(f"Failed to load model: {e}") from e


def model_generate(context_text: str) -> str:
    """
    Generate a question and answer from the given context using the model.

    Args:
        context_text: The input context/passage to generate Q&A from.

    Returns:
        The generated question and answer string.
    """
    prompt = (
        f"Generate Question and Answer or Fill in the Blanks from the given context: "
        f"{context_text}"
    )
    inputs = _tokenizer(
        prompt,
        return_tensors="pt",
        max_length=MAX_INPUT_LENGTH,
        truncation=True,
        padding="max_length",
    )
    inputs = {k: v.to(_device) for k, v in inputs.items()}

    with torch.no_grad():
        output = _model.generate(
            **inputs,
            max_length=MAX_OUTPUT_LENGTH,
            num_beams=NUM_BEAMS,
            early_stopping=True,
        )

    return _tokenizer.decode(output[0], skip_special_tokens=True)


def clean_input(text: str) -> str:
    """
    Validate and clean input text.

    Returns empty string if text has fewer than MIN_WORDS_FOR_INPUT words.

    Args:
        text: Raw input text.

    Returns:
        Cleaned text or empty string if invalid.
    """
    text = text.strip()
    word_count = len(text.split())
    return text if word_count >= MIN_WORDS_FOR_INPUT else ""


def split_text_into_sentences(text: str) -> List[str]:
    """
    Split text into sentence pairs for batch Q&A generation.

    Sentences are split on period, exclamation, or question mark followed by
    whitespace. Pairs are formed from consecutive sentences; odd sentences
    are kept as single-item pairs.

    Args:
        text: Input text to split.

    Returns:
        List of sentence pairs (or single sentences for odd counts).
    """
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    sentence_pairs = []

    for i in range(0, len(sentences) - 1, 2):
        pair = sentences[i] + " " + sentences[i + 1]
        sentence_pairs.append(pair)

    if len(sentences) % 2 != 0:
        sentence_pairs.append(sentences[-1])

    return sentence_pairs


def _parse_model_output(question_answer: str) -> Tuple[str, str]:
    """Parse model output into (question, answer) tuple."""
    if "Answer:" in question_answer:
        question, answer = question_answer.split("Answer:", 1)
        question = question.replace("Question:", "").strip()
        answer = answer.strip()
    else:
        question = question_answer.strip()
        answer = "Not provided"
    return question, answer


def generate_qa_pairs(input_text: str) -> List[Tuple[str, str]]:
    """
    Generate multiple Q&A pairs from input text.

    Splits text into sentence pairs, validates each, and generates
    question-answer pairs via the model.

    Args:
        input_text: Raw context text from the user.

    Returns:
        List of (question, answer) tuples.
    """
    sentences = split_text_into_sentences(input_text)
    qa_pairs = []

    for sentence in sentences:
        cleaned = clean_input(sentence)
        if not cleaned:
            continue

        try:
            raw_output = model_generate(cleaned)
            question, answer = _parse_model_output(raw_output)
            qa_pairs.append((question, answer))
        except Exception:
            # Skip failed generations; continue with others
            continue

    return qa_pairs


@app.route("/", methods=["GET", "POST"])
def home():
    """
    Home page: display form and generated Q&A.

    GET: Renders the form.
    POST: Processes context text and renders results.
    """
    generated_qas = []

    if request.method == "POST":
        context_text = request.form.get("context_text", "").strip()
        if context_text:
            generated_qas = generate_qa_pairs(context_text)

    return render_template("index.html", generated_qas=generated_qas)


def init_app():
    """Initialize the application and load the model."""
    load_model()


# Load model when app is imported (e.g. for flask run, gunicorn)
init_app()

if __name__ == "__main__":
    app.run(debug=True)
