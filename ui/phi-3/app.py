import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import gradio as gr
import requests

# --------------------------
# MODEL CONFIG
# --------------------------
MODEL_REPO = "lauraloretta/phi3-merged-mini"
device = "cpu"

print("Loading merged Phi-3 model...")

# Load model + tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_REPO,
    trust_remote_code=True
)

# --------------------------
# FIX TOKENIZER ISSUES
# --------------------------

# 1. Phi models MUST use eos_token as pad_token
tokenizer.pad_token = tokenizer.eos_token

# 2. Override broken chat_template stored inside tokenizer_config.json
tokenizer.chat_template = """
{% for message in messages %}
{% if message['role'] == 'system' %}
<|system|>
{{ message['content'] }}
<|end|>
{% elif message['role'] == 'user' %}
<|user|>
{{ message['content'] }}
<|end|>
{% elif message['role'] == 'assistant' %}
<|assistant|>
{{ message['content'] }}
<|end|>
{% endif %}
{% endfor %}
{% if add_generation_prompt %}
<|assistant|>
{% endif %}
"""

tokenizer.clean_up_tokenization_spaces = True

# Load model
model = AutoModelForCausalLM.from_pretrained(
    MODEL_REPO,
    torch_dtype=torch.float32,
    low_cpu_mem_usage=True,
)

model.to(device)
model.eval()

print("Model loaded successfully.")


# --------------------------
# OPTIONAL: WEATHER FEATURE
# --------------------------
ner_pipeline = pipeline(
    "token-classification",
    model="dslim/bert-base-NER",
    aggregation_strategy="simple",
)


def extract_text(content):
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(blk.get("text", "") for blk in content if isinstance(blk, dict))
    return str(content)


def extract_city_from_text(user_text, default_city="Stockholm"):
    try:
        ents = ner_pipeline(user_text)
    except Exception:
        return default_city
    locs = [e["word"] for e in ents if e.get("entity_group") == "LOC"]
    return locs[-1].strip(" ,.!?") if locs else default_city


def get_external_context(user_text):
    user_lower = user_text.lower()
    parts = []

    if any(w in user_lower for w in ["weather", "today", "forecast"]):
        try:
            city = extract_city_from_text(user_text)
            geo = requests.get(
                f"https://geocoding-api.open-meteo.com/v1/search?name={city}&count=1",
                timeout=5,
            ).json()

            lat = geo["results"][0]["latitude"]
            lon = geo["results"][0]["longitude"]

            url = (
                f"https://api.open-meteo.com/v1/forecast"
                f"?latitude={lat}&longitude={lon}"
                f"&current_weather=true"
                f"&timezone=auto"
            )

            resp = requests.get(url, timeout=5).json()
            cw = resp["current_weather"]
            temp = cw["temperature"]
            wind = cw["windspeed"]

            parts.append(f"Weather now in {city}: {temp}°C, wind {wind} km/h.")

        except Exception:
            parts.append("Could not fetch weather data.")

    return "\n".join(parts) if parts else ""


# --------------------------
# MAIN CHAT FN
# --------------------------
def chat_fn(message, history):
    if history is None:
        history = []

    user_text = extract_text(message)
    external_context = get_external_context(user_text)

    # Build system message
    system_message = "You are Laurapp, a helpful AI assistant."
    if external_context:
        system_message += "\n\nExternal information:\n" + external_context

    messages = [{"role": "system", "content": system_message}]

    # Add history
    for turn in history:
        messages.append({"role": "user", "content": turn["user"]})
        messages.append({"role": "assistant", "content": turn["assistant"]})

    # Current message
    messages.append({"role": "user", "content": user_text})

    # Format according to Phi-3 template
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to(device)

    input_len = inputs.shape[1]

    # Generate response
    with torch.no_grad():
        output = model.generate(
            input_ids=inputs,
            max_new_tokens=180,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    reply = tokenizer.decode(output[0, input_len:], skip_special_tokens=True)
    return reply.strip()


# --------------------------
# GRADIO INTERFACE
# --------------------------
demo = gr.ChatInterface(
    fn=chat_fn,
    title="Fine-Tuned Phi-3.5 Mini",
    description="A fine-tuned assistant powered by a merged Phi-3.5 model.",
)

if __name__ == "__main__":
    demo.launch()
