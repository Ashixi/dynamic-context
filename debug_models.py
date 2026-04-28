import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import warnings

warnings.filterwarnings("ignore")

BASE_MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
MAIN_LORA_PATH = "./qwen_context_manager_lora"
TAGGER_LORA_PATH = "./checkpoints_tagger_checkpoint-500"

print("[*] Завантаження токенізатора...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)

print("[*] Завантаження базових моделей...")
base_model_main = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_ID, device_map="auto", torch_dtype=torch.bfloat16
)
base_model_tagger = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_ID, device_map="auto", torch_dtype=torch.bfloat16
)

print("[*] Коригування розміру словника для тегувальника (151669)...")
base_model_tagger.resize_token_embeddings(151669)

print("[+] Підключення головної моделі (Context Manager)...")
model_main = PeftModel.from_pretrained(base_model_main, MAIN_LORA_PATH)

print("[+] Підключення моделі-тегувальника (Agentic Memory)...")
model_tagger = PeftModel.from_pretrained(base_model_tagger, TAGGER_LORA_PATH)


def generate_tagger_text(model, user_text):
    """Генерація для тегувальника ІЗ ЖОРСТКИМ СИСТЕМНИМ ПРОМПТОМ ТА СПИСКОМ ТЕГІВ"""
    
    allowed_tags = [
        "coding_logic", "frontend", "backend", "mobile_dev", "database", 
        "api_integration", "testing_qa", "architecture", "documentation", "legacy_code",
        "cloud_computing", "deployment", "security", "network", "linux_unix", 
        "docker_k8s", "performance_tuning", "cryptography", "smart_contracts", "ci_cd",
        "machine_learning", "data_science", "big_data", "ai_agents", "nlp", "mathematics", "statistics",
        "finance", "marketing", "legal", "project_management", "startups", "investment", "hr_recruiting", "product_design",
        "physics", "chemistry", "biology", "astronomy", "history", "philosophy", "languages",
        "mental_health", "lifestyle", "relations", "dating", "travel", "health_fitness", "gaming",
        "unknown_general"
    ]
    
    system_prompt = (
        "You are a passive data extraction tool. Your ONLY job is to parse input into JSON. "
        "DO NOT answer questions. DO NOT provide advice. DO NOT solve problems.\n"
        "FIELDS SPECIFICATION:\n"
        "1. 'extracted_query': Summary of what the user is asking for (string).\n"
        "2. 'query_tags': Technical tags from the allowed list (array).\n"
        "3. 'extracted_data': A literal string containing ONLY facts, logs, or configs from the user's text. "
        "Example: 'Error 137; RAM 16GB; shared_buffers=4GB'. DO NOT write solutions here.\n"
        "4. 'data_tags': Tags describing the extracted facts (array).\n"
        f"ALLOWED TAGS: {allowed_tags}.\n"
        "OUTPUT ONLY RAW JSON."
    )
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_text}
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([text], return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            max_new_tokens=512,
            temperature=0.1, 
            pad_token_id=tokenizer.eos_token_id
        )
    
    input_length = inputs.input_ids.shape[1]
    return tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True).strip()

def generate_main_text(model, user_text):
    """Генерація для головної моделі (БЕЗ СИСТЕМНОГО ПРОМПТА, як вчилася)"""
    messages = [{"role": "user", "content": user_text}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([text], return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            max_new_tokens=512,
            temperature=0.1,
            pad_token_id=tokenizer.eos_token_id
        )
    
    input_length = inputs.input_ids.shape[1]
    return tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True).strip()

if __name__ == "__main__":
    print("\n" + "="*50)
    print("🛠 RAW DEBUGGER ГОТОВИЙ (System Prompt + Allowed Tags)")
    print("="*50)
    
    while True:
        try:
            user_in = input("\nВведіть запит (або 'exit'): ")
            if user_in.lower() in ['exit', 'quit']: break
            if not user_in.strip(): continue
            
            print("\n" + "-"*50)
            print("1️⃣ СИРИЙ ВИВІД ТЕГУВАЛЬНИКА (з System Prompt):")
            tagger_out = generate_tagger_text(model_tagger, user_in)
            print(f">>>\n{tagger_out}\n<<<")
            
            print("\n" + "-"*50)
            print("2️⃣ СИРИЙ ВИВІД ГОЛОВНОЇ МОДЕЛІ (без System Prompt):")
            main_out = generate_main_text(model_main, user_in)
            print(f">>>\n{main_out}\n<<<")
            print("-"*50)
            
        except KeyboardInterrupt:
            break