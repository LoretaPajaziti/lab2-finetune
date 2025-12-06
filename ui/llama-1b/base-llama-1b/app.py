import torch
from transformers import AutoTokenizer, AutoModelForCausalLM,  pipeline
from peft import PeftModel
import gradio as gr
import requests
import re
import os

BASE_MODEL = "unsloth/Llama-3.2-1B-Instruct"
LORA_REPO = "lauraloretta/llama-1B-10000-params"


device = "cpu"

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)


print("Loading base model on CPU (this can take a while)...")
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float32,
    device_map=None
)
base_model.to(device)

if hasattr(base_model, "hf_device_map"):
    del base_model.hf_device_map


print("Loading LoRA adapter...")
model = PeftModel.from_pretrained(base_model, LORA_REPO, device_map=None)
model.to(device)
model.eval()

ner_pipeline = pipeline(
    "token-classification",
    model="dslim/bert-base-NER",
    aggregation_strategy="simple",
)

def extract_text(content):
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
        return "\n".join(parts)
    return str(content)
    

def extract_city_from_text(user_text: str, default_city: str = "Stockholm") -> str:
    """
    Uses NER model to try to extract city/main place from the user msg. 
    If it does not find it returns default_city
    """
    text_for_ner = user_text.title()
    
    try:
        ents = ner_pipeline(user_text)
    except Exception:
        return default_city

    locs = [e["word"] for e in ents if e.get("entity_group") in ("LOC", "ORG", "PER", "MISC")]

    if not locs:
        return default_city

    # Choose the last mention one (usually the most meaningful)
    city = locs[-1].strip()
    # Small cleanup
    city = city.strip(" ,.?!")
    return city or default_city


def get_external_context(user_text: str) -> str:
    """
    Decide which tools to call based on the user message,
    and return a short text summary to inject into the prompt.
    """
    user_lower = user_text.lower()
    parts = []
    WEATHER_KEYWORDS = [
        "weather", "forecast", "temperature", "rain", "snow", "wind",
        "outside", "today", "tomorrow", "conditions", "climate",
        "humid", "humidity", "uv", "heat", "cold", "storm",
        "cloudy", "sunny", "hot", "warm", "freezing"
    ]

    if any(word in user_lower for word in WEATHER_KEYWORDS):  
        try:


            city = city = extract_city_from_text(user_text, default_city="Stockholm")
            print("City:", city)

            geo = requests.get(
                f"https://geocoding-api.open-meteo.com/v1/search?name={city}&count=1",
                timeout=5
            ).json()

            print("GEO: ", geo)

            lat = geo["results"][0]["latitude"]
            lon = geo["results"][0]["longitude"]

            url = (
                    f"https://api.open-meteo.com/v1/forecast"
                    f"?latitude={lat}&longitude={lon}"
                    f"&current_weather=true"
                    f"&daily=temperature_2m_max,temperature_2m_min,precipitation_sum,"
                    f"precipitation_probability_max,uv_index_max,weathercode"
                    f"&hourly=relative_humidity_2m,apparent_temperature,cloudcover,windspeed_10m"
                    f"&timezone=auto"
                )
            
            resp = requests.get(
                url,
                timeout=5
            ).json()
            
            cw = resp.get("current_weather")
            dw = resp.get("daily", {})
            hourly = resp.get("hourly", {})
            
            if cw and dw:
                temp = round(cw.get("temperature"))
                code = cw.get("weathercode")
                wind = round(cw.get("windspeed"))
                
                mapping = {
                    0: "clear sky", 1: "mainly clear", 2: "partly cloudy", 3: "cloudy",
                    45: "foggy", 48: "depositing rime fog",
                    51: "light drizzle", 53: "moderate drizzle", 55: "dense drizzle",
                    61: "light rain", 63: "moderate rain", 65: "heavy rain",
                    66: "freezing rain", 67: "heavy freezing rain",
                    71: "light snow", 73: "moderate snow", 75: "heavy snow",
                    95: "thunderstorm", 96: "thunderstorm with hail"
                }
                desc = mapping.get(code, "unknown conditions")

                # Daily
                rain_prob = dw.get("precipitation_probability_max", [None])[0]
                rain_mm = dw.get("precipitation_sum", [None])[0]
                tmax = dw.get("temperature_2m_max", [None])[0]
                tmin = dw.get("temperature_2m_min", [None])[0]
                uv = dw.get("uv_index_max", [None])[0]

                # Hourly 
                humidity = hourly.get("relative_humidity_2m", [None])[0]
                feels_like = hourly.get("apparent_temperature", [None])[0]
                cloudcover = hourly.get("cloudcover", [None])[0]

                weather_text = (
                    f"Current weather in {city.title()}: {temp}°C, {desc}. "
                    f"Feels like {feels_like}°C. "
                    f"Humidity: {humidity}%. Wind: {wind} km/h. Cloud cover: {cloudcover}%.\n"
                    f"Today's high: {tmax}°C, low: {tmin}°C. "
                    f"Chance of rain: {rain_prob}%, total: {rain_mm} mm. "
                    f"UV index max: {uv}."
                )

                parts.append(weather_text)
            else:
                print("API error:", resp.weather_text)
                parts.append("There is no current weather aviable")

        except Exception as e:
            print("Exception:", e)
            parts.append("Could not fetch current weather due to an error.")

    # Join all tool outputs into one context string
    if parts:
        context = (
            "You have access to the following external, real-time information. "
            "Use it when answering the user. Do NOT say that you lack "
            "current information if the answer is provided here.\n\n"
            + "\n".join(parts)
        )
        print("DEBUG external context:\n", context)  # 👈 debug
        return context
    else:
        print("DEBUG external context: (none)")
        return ""



def chat_fn(message, history):
   
    if history is None:
        history = []

    # 1) Get current user text (we need it early to decide tools)
    if isinstance(message, dict):
        user_text = extract_text(message.get("content", ""))
    else:
        user_text = str(message)

    # 2) Call tools / APIs based on the user_text
    external_context = get_external_context(user_text)

    messages = []

    messages.append({
        "role": "system",
        "content": "You are Laurapp, a helpful AI assistant."
    })

    
    if external_context:
        messages.insert(1, {
            "role": "system",
            "content": external_context,
        })


    # 2) Historial
    for m in history:
        if isinstance(m, dict):
            role = m.get("role", "user")
            content = extract_text(m.get("content", ""))
            messages.append({"role": role, "content": content})
        elif isinstance(m, (list, tuple)) and len(m) == 2:
            # Soporte legacy si algún día usas type="tuples"
            user_msg, bot_msg = m
            if user_msg:
                messages.append({"role": "user", "content": user_msg})
            if bot_msg:
                messages.append({"role": "assistant", "content": bot_msg})

    messages.append({"role": "user", "content": user_text})

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
        
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
    ).to(device)

    input_length = inputs["input_ids"].shape[1]

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=128,
            do_sample=True,
            temperature=0.7,
        )

    generated_ids = output[0, input_length:]
    text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    
    return text.strip()

demo = gr.ChatInterface(
    fn=chat_fn,
    title="Lab 2 – Fine-tuned Llama 1B",
    description="Small demo running on CPU with a fine-tuned LoRA adapter. with 10.000 params on a full batch",
)

if __name__ == "__main__":
    demo.launch()