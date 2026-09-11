import re
import time
from typing import Dict, List, Optional, Tuple

import streamlit as st
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline


# ============================================================
# Page Configuration
# ============================================================

st.set_page_config(
    page_title="English to Urdu AI Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# Model Configuration
# ============================================================

TRANSLATION_MODEL_NAME = "Helsinki-NLP/opus-mt-en-ur"
CHAT_MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"


# ============================================================
# Translation Model
# ============================================================

@st.cache_resource(show_spinner="Loading translation model...")
def load_translation_model():
    """
    Loads the English-to-Urdu translation model on CPU.
    """

    translator = pipeline(
        task="translation",
        model=TRANSLATION_MODEL_NAME,
        tokenizer=TRANSLATION_MODEL_NAME,
        device=-1,
    )

    return translator


# ============================================================
# Chat Model
# ============================================================

@st.cache_resource(show_spinner="Loading chat model...")
def load_chat_model():
    """
    Loads the Qwen instruction-following chat model on CPU.
    """

    tokenizer = AutoTokenizer.from_pretrained(
        CHAT_MODEL_NAME
    )

    model = AutoModelForCausalLM.from_pretrained(
        CHAT_MODEL_NAME,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
    )

    model.to("cpu")
    model.eval()

    return tokenizer, model


# ============================================================
# Urdu Processing
# ============================================================

def normalize_urdu(text: str) -> str:
    """
    Normalizes common Urdu and Arabic Unicode variants.
    """

    replacements = {
        "ي": "ی",
        "ى": "ی",
        "ك": "ک",
        "ۀ": "ہ",
        "ة": "ہ",
        "ـ": "",
    }

    for old_character, new_character in replacements.items():
        text = text.replace(old_character, new_character)

    text = re.sub(r"\s+", " ", text)

    return text.strip()


def contains_english_characters(text: str) -> bool:
    """
    Checks whether English alphabet characters exist.
    """

    return bool(re.search(r"[A-Za-z]", text))


def contains_urdu_script(text: str) -> bool:
    """
    Checks whether Urdu or Arabic script exists.
    """

    return bool(re.search(r"[\u0600-\u06FF]", text))


def validate_urdu_output(text: str) -> Dict[str, bool]:
    """
    Validates Urdu output.
    """

    urdu_detected = contains_urdu_script(text)
    english_detected = contains_english_characters(text)

    return {
        "urdu_detected": urdu_detected,
        "english_detected": english_detected,
        "valid": urdu_detected and not english_detected,
    }


# ============================================================
# Translation Function
# ============================================================

def translate_to_urdu(
    translator,
    text: str,
    beam_size: int = 3,
) -> Tuple[str, Dict[str, bool]]:
    """
    Translates English text into Urdu.
    """

    text = text.strip()

    if not text:
        return "", {
            "urdu_detected": False,
            "english_detected": False,
            "valid": False,
        }

    result = translator(
        text,
        max_length=256,
        num_beams=beam_size,
        early_stopping=True,
    )

    translated_text = result[0]["translation_text"]
    translated_text = normalize_urdu(translated_text)

    validation = validate_urdu_output(translated_text)

    return translated_text, validation


# ============================================================
# Chat Generation
# ============================================================

def generate_chat_response(
    tokenizer,
    model,
    user_prompt: str,
    conversation_history: List[Dict[str, str]],
    answer_language: str,
) -> str:
    """
    Generates a response using the Qwen instruction model.
    """

    system_instruction = (
        "You are a helpful and professional AI assistant. "
        "Answer the user's question clearly and accurately. "
        "Do not claim to know information that you do not know. "
        "Keep the answer concise and useful. "
    )

    if answer_language == "Urdu":
        system_instruction += (
            "Always answer in natural Urdu script. "
            "Do not use Roman Urdu. "
            "Avoid unnecessary English words."
        )
    else:
        system_instruction += (
            "Answer in clear English."
        )

    messages = [
        {
            "role": "system",
            "content": system_instruction,
        }
    ]

    for message in conversation_history[-6:]:
        messages.append(
            {
                "role": message["role"],
                "content": message["content"],
            }
        )

    messages.append(
        {
            "role": "user",
            "content": user_prompt,
        }
    )

    prompt_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    model_inputs = tokenizer(
        [prompt_text],
        return_tensors="pt",
        truncation=True,
        max_length=1024,
    )

    model_inputs = {
        key: value.to("cpu")
        for key, value in model_inputs.items()
    }

    with torch.inference_mode():
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=180,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id,
        )

    input_length = model_inputs["input_ids"].shape[1]

    new_tokens = generated_ids[:, input_length:]

    response = tokenizer.batch_decode(
        new_tokens,
        skip_special_tokens=True,
    )[0].strip()

    if answer_language == "Urdu":
        response = normalize_urdu(response)

    return response


# ============================================================
# Session State
# ============================================================

if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []

if "translation_messages" not in st.session_state:
    st.session_state.translation_messages = []


# ============================================================
# Sidebar
# ============================================================

with st.sidebar:
    st.header("Application Settings")

    selected_mode = st.radio(
        "Select Mode",
        options=[
            "Chat Mode",
            "Translation Mode",
        ],
    )

    st.divider()

    st.subheader("Chat Settings")

    answer_language = st.selectbox(
        "Chat Answer Language",
        options=[
            "Urdu",
            "English",
        ],
    )

    st.divider()

    st.subheader("Translation Settings")

    beam_size = st.slider(
        "Beam Search Size",
        min_value=1,
        max_value=5,
        value=3,
        help="Higher values may improve quality but increase CPU time.",
    )

    st.divider()

    st.subheader("Models")

    st.write("**Chat Model**")
    st.code(CHAT_MODEL_NAME)

    st.write("**Translation Model**")
    st.code(TRANSLATION_MODEL_NAME)

    st.write("**Device**")
    st.info("CPU mode")

    st.divider()

    if st.button(
        "Clear Current Mode History",
        use_container_width=True,
    ):
        if selected_mode == "Chat Mode":
            st.session_state.chat_messages = []
        else:
            st.session_state.translation_messages = []

        st.rerun()


# ============================================================
# Load Only Required Model
# ============================================================

translation_model = None
chat_tokenizer = None
chat_model = None

try:
    if selected_mode == "Translation Mode":
        translation_model = load_translation_model()
    else:
        chat_tokenizer, chat_model = load_chat_model()

except Exception as error:
    st.error("Model loading failed.")
    st.exception(error)
    st.stop()


# ============================================================
# Main Header
# ============================================================

st.title("English to Urdu AI Assistant")

st.caption(
    "A CPU-friendly application with separate chat and translation modes."
)


# ============================================================
# Chat Mode
# ============================================================

if selected_mode == "Chat Mode":

    st.subheader("Chat Mode")

    st.write(
        "Ask a question and the instruction-tuned chat model "
        "will generate a response."
    )

    for message in st.session_state.chat_messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

            if message["role"] == "assistant":
                metadata = message.get("metadata")

                if metadata:
                    st.caption(
                        f"Response time: "
                        f"{metadata['response_time']:.2f} seconds"
                    )

    user_prompt = st.chat_input(
        "Ask anything in English..."
    )

    if user_prompt:
        st.session_state.chat_messages.append(
            {
                "role": "user",
                "content": user_prompt,
            }
        )

        with st.chat_message("user"):
            st.markdown(user_prompt)

        start_time = time.perf_counter()

        with st.chat_message("assistant"):
            with st.spinner("Generating chat response..."):
                response = generate_chat_response(
                    tokenizer=chat_tokenizer,
                    model=chat_model,
                    user_prompt=user_prompt,
                    conversation_history=(
                        st.session_state.chat_messages
                    ),
                    answer_language=answer_language,
                )

            elapsed_time = time.perf_counter() - start_time

            st.markdown(response)
            st.caption(
                f"Chat model response | "
                f"Response time: {elapsed_time:.2f} seconds"
            )

        st.session_state.chat_messages.append(
            {
                "role": "assistant",
                "content": response,
                "metadata": {
                    "response_time": elapsed_time,
                },
            }
        )


# ============================================================
# Translation Mode
# ============================================================

else:

    st.subheader("Translation Mode")

    st.write(
        "Enter English text and the transformer model will "
        "translate it into Urdu."
    )

    for message in st.session_state.translation_messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

            if message["role"] == "assistant":
                metadata = message.get("metadata")

                if metadata:
                    st.caption(
                        f"Response time: "
                        f"{metadata['response_time']:.2f} seconds"
                    )

    user_text = st.chat_input(
        "Enter English text to translate..."
    )

    if user_text:
        st.session_state.translation_messages.append(
            {
                "role": "user",
                "content": user_text,
            }
        )

        with st.chat_message("user"):
            st.markdown(user_text)

        start_time = time.perf_counter()

        with st.chat_message("assistant"):
            with st.spinner("Translating into Urdu..."):
                translated_text, validation = translate_to_urdu(
                    translator=translation_model,
                    text=user_text,
                    beam_size=beam_size,
                )

            elapsed_time = time.perf_counter() - start_time

            st.markdown(translated_text)

            if validation["valid"]:
                st.success("Urdu output validation passed.")

            elif validation["english_detected"]:
                st.warning(
                    "The output contains some English characters."
                )

            else:
                st.info(
                    "Translation generated, but output validation "
                    "needs manual review."
                )

            st.caption(
                f"Transformer translation | "
                f"Response time: {elapsed_time:.2f} seconds"
            )

        st.session_state.translation_messages.append(
            {
                "role": "assistant",
                "content": translated_text,
                "metadata": {
                    "response_time": elapsed_time,
                    "validation": validation,
                },
            }
        )