# AskIt

**Explore. Discover. Answer.**

AskIt is a web application that generates questions and answers from user-provided text using a fine-tuned T5 model with PEFT (Parameter-Efficient Fine-Tuning). Paste any passage or context, and the app will automatically create relevant Q&A pairs or fill-in-the-blank exercises.

## Features

- **Q&A Generation**: Automatically generates questions and answers from input text
- **Fill-in-the-Blanks**: Can produce fill-in-the-blank style exercises
- **Sentence Pair Processing**: Splits long text into sentence pairs for focused generation
- **Web Interface**: Modern Slate/Indigo themed UI with responsive design

## Requirements

- Python 3.9 or higher
- CUDA-capable GPU (optional, for faster inference; CPU is supported)

## Setup

### 1. Clone or download the project

```bash
cd askit
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Ensure the model is present

The app expects the fine-tuned PEFT model in `./QA_flan_Peft_Model/`. This directory should contain:

- `adapter_config.json`
- `adapter_model.safetensors` or `adapter_model.bin`
- Tokenizer files (`spiece.model`, `tokenizer_config.json`, etc.)

## Usage

### Run the application

```bash
python app.py
```

Or using Flask's development server:

```bash
export FLASK_APP=app.py
flask run
```

### Access the app

Open your browser and go to [http://127.0.0.1:5000](http://127.0.0.1:5000).

### How to use

1. Paste your text (paragraph, article, or passage) into the text area
2. Click **Generate Q&A**
3. View the generated questions and answers below

**Note**: Input text should have at least 3 words per sentence pair for valid generation.

## Project Structure

```
askit/
├── app.py                 # Flask application and Q&A generation logic
├── requirements.txt       # Python dependencies
├── README.md              # This file
├── templates/
│   └── index.html         # Web UI
└── QA_flan_Peft_Model/    # Fine-tuned T5 (PEFT) model files
    ├── adapter_config.json
    ├── adapter_model.*
    ├── spiece.model
    └── ...
```

## Model Information

The application uses a PEFT (LoRA) fine-tuned T5-small model. Key details:

- **Base model**: `t5-small`
- **Fine-tuning**: LoRA (Low-Rank Adaptation)
- **Task**: Sequence-to-sequence (Q&A generation, fill-in-the-blanks)

See `QA_flan_Peft_Model/README.md` for the full model card.

## Configuration

You can adjust generation parameters in `app.py`:

| Variable | Default | Description |
|----------|---------|--------------|
| `MAX_INPUT_LENGTH` | 512 | Maximum token length for input |
| `MAX_OUTPUT_LENGTH` | 128 | Maximum token length for output |
| `NUM_BEAMS` | 5 | Beam search width for generation |
| `MIN_WORDS_FOR_INPUT` | 3 | Minimum words required per input segment |

## License

[Specify your license here]
