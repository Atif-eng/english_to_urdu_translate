# English to Urdu AI Assistant

A transformer-based conversational AI application that provides two separate capabilities:

1. Chat Mode for generating intelligent responses.
2. Translation Mode for translating English text into Urdu.

The application is built with Streamlit and runs locally on CPU-compatible systems. It is designed as an initial MVP for a final-year data science and natural language processing project.

---

## Project Overview

This project combines two independent NLP models in a single Streamlit dashboard:

- An instruction-tuned language model for chatbot responses.
- A neural machine translation model for English-to-Urdu translation.

The system separates conversation and translation into two modes so that each task is handled by a model designed specifically for that purpose.

---

## Features

- Professional Streamlit dashboard.
- Separate Chat Mode and Translation Mode.
- English-to-Urdu neural machine translation.
- Instruction-based conversational chatbot.
- Urdu response generation option.
- English response generation option.
- Conversation history using Streamlit session state.
- Urdu Unicode normalization.
- Urdu output validation.
- Beam-search control for translation.
- CPU-compatible local execution.
- Cached model loading.
- Response-time display.
- Clear chat-history control.
- Modular architecture for future improvements.

---

## System Architecture

```text
                    User Input
                        |
                Streamlit Dashboard
                        |
          +-------------+-------------+
          |                           |
      Chat Mode                 Translation Mode
          |                           |
 Qwen Instruction Model       OPUS-MT Model
          |                           |
    Chat Response              Urdu Translation
```

---

## Models

### Chat Model

```text
Qwen/Qwen2.5-0.5B-Instruct
```

The chat model is used for instruction-following and conversational responses.

### Translation Model

```text
Helsinki-NLP/opus-mt-en-ur
```

The translation model is used specifically for English-to-Urdu neural machine translation.

---

## Technologies Used

- Python
- Streamlit
- PyTorch
- Hugging Face Transformers
- OPUS-MT
- Qwen Instruct
- SentencePiece
- Sacremoses
- Conda
- Git and GitHub

---

## Project Structure

```text
english-urdu-chatbot/
│
├── app.py
├── requirements.txt
├── README.md
└── .gitignore
```

---

## Hardware Configuration

The current local prototype is designed for CPU execution.

```text
RAM: 16 GB
GPU: Not available
Execution device: CPU
```

The application does not require a dedicated local GPU for the basic prototype. However, CPU inference can be slower than GPU inference, especially for the chat model.

For future fine-tuning and larger experiments, Kaggle GPU or another cloud GPU platform is recommended.

---

## Installation

### 1. Clone the repository

```bash
git clone [https://github.com/YOUR_USERNAME/english-urdu-chatbot.git](https://github.com/YOUR_USERNAME/english-urdu-chatbot.git)
```

Move into the project directory:

```bash
cd english-urdu-chatbot
```

Replace `YOUR_USERNAME` with your GitHub username.

---

### 2. Create a Conda environment

```bash
conda create -n english_urdu_chatbot python=3.11 -y
```

Activate the environment:

```bash
conda activate english_urdu_chatbot
```

Python 3.11 is recommended for compatibility with Streamlit, PyTorch, and Hugging Face Transformers.

---

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

The required packages are:

```text
streamlit==1.45.1
transformers==4.56.2
torch
sentencepiece
sacremoses
accelerate
```

---

## Running the Application

From the project directory, run:

```bash
python -m streamlit run app.py
```

The application will open in the browser at:

```text
http://localhost:8501
```

To stop the application, press:

```text
Ctrl + C
```

---

## How to Use

### Chat Mode

Select `Chat Mode` from the sidebar and enter a question.

Example prompts:

```text
What is machine learning?
```

```text
Explain data science in simple words.
```

```text
How are transformers used in NLP?
```

You can select the response language:

- Urdu
- English

---

### Translation Mode

Select `Translation Mode` from the sidebar and enter English text.

Example:

```text
Machine learning is useful for solving real-world problems.
```

The translation model will generate an Urdu translation.

---

## Model Loading and Caching

The application uses Streamlit's resource caching:

```python
@st.cache_resource
```

This prevents the models from loading again on every interaction.

During the first run:

```text
Model files are downloaded.
Models are loaded into memory.
```

During later interactions in the same session:

```text
Cached models are reused.
```

If the application is completely restarted, the models will be loaded into memory again. However, downloaded model files are normally reused from the local Hugging Face cache.

---

## CPU Execution

The current application is configured for CPU execution:

```python
device=-1
```

A dedicated GPU is not required for the initial demo. However, CPU inference may take longer, particularly for the conversational model.

For larger transformer models, fine-tuning, and image generation experiments, a cloud GPU such as Kaggle is recommended.

---

## Output Validation

The application performs basic Urdu output validation:

- Checks whether Urdu or Arabic script is present.
- Checks whether English alphabet characters appear.
- Normalizes common Urdu Unicode variations.
- Displays a validation status in the interface.

Example normalization mappings:

```text
ي → ی
ى → ی
ك → ک
ۀ → ہ
ة → ہ
```

This validation is a rule-based quality check. It does not guarantee perfect grammar or perfect translation.

---

## Important Limitations

This is an MVP prototype and has the following limitations:

- The chat model may generate inaccurate information.
- CPU inference may be slow.
- The translation model may produce incorrect or unnatural translations.
- The system does not currently use a verified knowledge base.
- The system does not currently use retrieval-augmented generation.
- Urdu translation quality depends on the pretrained model.
- The system does not guarantee completely English-free Urdu output.
- Large prompts may be truncated.
- Chat responses are not a replacement for expert advice.

---

## Future Improvements

The planned improvements include:

### Dataset Development

- Collect English-Urdu parallel corpora.
- Use BPCC, OPUS, Tatoeba, and manually verified data.
- Remove duplicates and noisy sentence pairs.
- Normalize Urdu Unicode.
- Filter Roman Urdu and unwanted English text.
- Create train, validation, and test splits.

### Model Development

- Fine-tune NLLB-200 on Kaggle GPU.
- Evaluate IndicTrans2 for English-to-Urdu translation.
- Compare OPUS-MT, NLLB, IndicTrans2, and mBART.
- Add model reranking and ensemble translation.
- Apply parameter-efficient fine-tuning where appropriate.

### Chatbot Improvements

- Add intent classification.
- Add a verified FAQ knowledge base.
- Add retrieval-augmented generation.
- Add conversation context management.
- Add hallucination reduction techniques.
- Add source-based answer generation.

### Image Generation

- Add an optional text-to-image generation mode.
- Use Stable Diffusion or another diffusion model.
- Run image generation on Kaggle GPU or a dedicated GPU service.
- Add image preview and download functionality.

### Evaluation

The final system will be evaluated using:

- BLEU
- chrF++
- COMET
- Human fluency score
- Human adequacy score
- English-script leakage rate
- Chat response quality
- Response latency

---

## Planned Final Architecture

```text
User Input
    |
    v
Intent Detection
    |
    +-- Chat Request
    |       |
    |       +-- Chat Model / RAG Module
    |
    +-- Translation Request
    |       |
    |       +-- NLLB / IndicTrans2 / OPUS-MT
    |       |
    |       +-- Urdu Normalization
    |       |
    |       +-- Quality Validation
    |
    +-- Image Request
            |
            +-- Diffusion Model
```

---

## Deployment

The application can be deployed through Streamlit Community Cloud using GitHub.

Required files:

```text
app.py
requirements.txt
README.md
.gitignore
```

Basic deployment process:

1. Push the project to GitHub.
2. Open Streamlit Community Cloud.
3. Connect the GitHub account.
4. Select the repository.
5. Select the `main` branch.
6. Select `app.py` as the main file.
7. Click Deploy.

The `requirements.txt` file must be present in the repository so that the deployment platform can install the required dependencies.

---

## Git Commands

Initialize the repository:

```bash
git init
```

Add project files:

```bash
git add .
```

Create the first commit:

```bash
git commit -m "Initial English to Urdu AI Assistant"
```

Rename the branch:

```bash
git branch -M main
```

Add the GitHub remote:

```bash
git remote add origin [https://github.com/YOUR_USERNAME/english-urdu-chatbot.git](https://github.com/YOUR_USERNAME/english-urdu-chatbot.git)
```

Push the project:

```bash
git push -u origin main
```

---

## `.gitignore`

The repository should not include local environments, cache files, model files, or secret credentials.

Recommended `.gitignore`:

```gitignore
venv/
env/
.venv/
.conda/

__pycache__/
*.py[cod]
.ipynb_checkpoints/

.cache/
huggingface/
models/
checkpoints/

.env
.env.*
.streamlit/secrets.toml

*.log
*.tmp
*.temp

.DS_Store
Thumbs.db

.vscode/
.idea/

*.png
*.jpg
*.jpeg
*.webp
*.zip
```

---

## Project Status

```text
Current status: MVP prototype
```

Implemented:

- Streamlit dashboard
- Chat Mode
- Translation Mode
- CPU execution
- Model caching
- Conversation history
- Urdu output validation

Planned:

- Custom English-Urdu dataset
- NLLB fine-tuning
- IndicTrans2 evaluation
- Multi-model ensemble
- RAG chatbot
- Diffusion-based image generation
- Comprehensive evaluation
- Production deployment

---

## License

This project is developed for academic and educational purposes.

The licenses of the pretrained models should be reviewed before commercial deployment:

- Qwen model license
- OPUS-MT model license
- Hugging Face Transformers license
- Streamlit license

---

## Acknowledgements

This project uses open-source technologies and pretrained models from:

- Hugging Face
- Helsinki-NLP
- Qwen
- Streamlit
- PyTorch
- The open-source NLP community

---

## Author

Developed as a final-year data science and natural language processing project.

```text
Author: YOUR_NAME
Institution: YOUR_INSTITUTION
Department: Data Science
```
