# Multimodal English-to-Urdu Translation

An open-source NLP project for translating English into Urdu using Meta’s pretrained **NLLB-200 distilled 600M** sequence-to-sequence model. The intended system accepts either English text or an image containing English text and returns an Urdu text translation.

> **Project status:** The checked-in notebook currently focuses on preparing text-based English–Urdu parallel data and fine-tuning/evaluating NLLB. Image OCR and a user-facing app or API are not implemented in the repository yet. For image input, an OCR component must first extract English text before it can be passed to the translation model.

## Project goal

Support two input paths that meet at the same translation model:

```text
English text ───────────────────────┐
                                    ├─> NLLB-200 ─> Urdu text
Image containing English text ─> OCR┘
```

The image path is a planned part of the multimodal project, not a feature provided by the current notebook. OCR quality, especially for low-resolution, stylized, or complex images, will affect the final translation.

## Model

- **Base checkpoint:** `facebook/nllb-200-distilled-600M`
- **Source language:** `eng_Latn` (English)
- **Target language:** `urd_Arab` (Urdu)
- **Purpose:** Fine-tuning a pretrained multilingual translation model on English–Urdu sentence pairs.

The notebook configures a five-epoch training run, with options for a zero-shot baseline and translation evaluation. It uses BLEU and chrF++, with METEOR and COMET as optional additional metrics. Metric results are not a substitute for human review of translation quality.

## Repository contents

```text
.
├── Notebooks/
│   └── fyp-01.ipynb   # Data preparation, NLLB fine-tuning, and evaluation workflow
├── data/
│   ├── raw/           # Raw dataset files
│   └── processed/     # Reserved for processed data
├── src/               # Application/source-code placeholder; no runnable app yet
├── tests/             # Test placeholder; no automated tests yet
└── README.md
```

## Notebook workflow

`Notebooks/fyp-01.ipynb` is the current implementation entry point. It:

1. Sets up the Python packages used for training and evaluation.
2. Loads English–Urdu datasets for general, parallel-corpus, and Quranic text.
3. Cleans text, normalizes common Urdu character variants, filters unsuitable pairs, and removes duplicates.
4. Configures NLLB-200 distilled 600M for English-to-Urdu fine-tuning.
5. Provides training, baseline comparison, and translation-metric evaluation settings.

The notebook is configured for **Kaggle**. It expects these datasets to be added as notebook inputs:

- `/kaggle/input/datasets/atifkhan12/english-urdu`
- `/kaggle/input/datasets/atifkhan12/parallel-corpus`
- `/kaggle/input/datasets/atifkhan12/quran-df`

It writes its model and training outputs under `/kaggle/working/nllb_en_ur` when run on Kaggle. You may need to change the dataset paths and output directory to run it in another environment.

## Run the notebook

1. Open `Notebooks/fyp-01.ipynb` in Kaggle or another Jupyter environment.
2. In Kaggle, attach the three dataset inputs listed above and enable a GPU accelerator for practical training speed.
3. Run the notebook cells in order. The notebook installs several missing dependencies, downloads the base model, prepares the data, and runs its configured training/evaluation workflow.
4. Save or download the generated model/checkpoints and evaluation artifacts from the configured output directory before the runtime is cleared.

The notebook uses Hugging Face Transformers, Datasets, PyTorch, SentencePiece, Accelerate, and SacreBLEU. METEOR and COMET are optional. COMET may require substantial disk space and compatible package versions; the notebook attempts to disable it if setup or its smoke test fails. Training can be slow or run out of memory depending on the accelerator and settings.

There is currently no `requirements.txt`, application entry point, or documented command to launch a web interface or API. The notebook’s text translation/training workflow is the available starting point; end-to-end text and image translation requires building the application and integrating OCR.

## Next steps for the multimodal application

- Add an OCR component to extract English text from uploaded images.
- Pass extracted text and direct text input through the same translation function.
- Add a simple interface or API for text entry, image upload, and Urdu output.
- Add tests for OCR-to-translation flow, Urdu text handling, and error cases.
- Evaluate text-only translation and image OCR separately, then assess the complete pipeline.

## License and attribution

The repository does not currently specify a project license. The base checkpoint is published by Meta as `facebook/nllb-200-distilled-600M`; review the model’s Hugging Face page and license terms before using or distributing it. Add a repository license file before declaring the project’s own licensing terms.
