import json
import re
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel
from peft import PeftModel
import warnings

warnings.filterwarnings("ignore")

# ==========================================
# 1. CONFIGURATION
# ==========================================
BASE_MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
MAIN_LORA_PATH = "./qwen_context_manager_lora"
TAGGER_LORA_PATH = "./checkpoints_tagger_checkpoint-500"

DB_FILE = "real_knowledge_base.json"

# ==========================================
# 2. MODEL INITIALIZATION (FIXED FOR WINDOWS DEADLOCK)
# ==========================================
print("[SYSTEM] Initializing tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)

print("[SYSTEM] Loading SHARED base model into VRAM...")
use_cuda = torch.cuda.is_available()
if use_cuda:
    capability = torch.cuda.get_device_capability(0)
    bf16_supported = capability[0] >= 8
    load_dtype = torch.bfloat16 if bf16_supported else torch.float16
    device_map = {"": 0}
    print(f"[SYSTEM] CUDA device: {torch.cuda.get_device_name(0)} | capability={capability} | dtype={load_dtype}")
else:
    load_dtype = torch.float32
    device_map = {"": "cpu"}
    print("[SYSTEM] CUDA not available. Falling back to CPU.")

load_started = time.time()
try:
    shared_base_model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        device_map=device_map,
        dtype=load_dtype,
        low_cpu_mem_usage=True,
    )
except Exception as e:
    print(f"[SYSTEM] Primary load failed ({type(e).__name__}: {e}). Retrying on CPU...")
    shared_base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        device_map={"": "cpu"},
        dtype=torch.float32,
        low_cpu_mem_usage=True,
    )

print(f"[SYSTEM] Base model loaded in {time.time() - load_started:.2f}s")

print("[SYSTEM] Adjusting vocabulary size (151669)...")
shared_base_model.resize_token_embeddings(151669)

print("[LORA] Mounting Context Manager weights (Adapter: 'main')...")
multi_lora_model = PeftModel.from_pretrained(
    shared_base_model, 
    MAIN_LORA_PATH, 
    adapter_name="main"
)

print("[LORA] Mounting Agentic Memory weights (Adapter: 'tagger')...")
# Додаємо другий адаптер
multi_lora_model.load_adapter(TAGGER_LORA_PATH, adapter_name="tagger")

try:
    with open(DB_FILE, 'r', encoding='utf-8') as f:
        KNOWLEDGE_BASE = json.load(f)
    print(f"[DATABASE] Knowledge base loaded ({len(KNOWLEDGE_BASE)} entries).")
except FileNotFoundError:
    print(f"[DATABASE] Warning: '{DB_FILE}' not found. Initializing empty DB.")
    KNOWLEDGE_BASE = []

# ==========================================
# 3. GENERATION FUNCTIONS
# ==========================================
def generate_tagger_text(model, user_text, max_tokens=512, temp=0.1):
    # Перемикаємося на адаптер тегувальника
    if isinstance(model, PeftModel):
        model.set_adapter("tagger")
        
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
        "1. 'extracted_query': Summary of the user's intent.\n"
        "2. 'query_tags': Technical tags from the allowed list (array).\n"
        "3. 'extracted_data': A literal string containing ONLY facts, logs, or configs from the user's text. Example: 'Error 137; RAM 16GB'. DO NOT write solutions here.\n"
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
            max_new_tokens=max_tokens,
            temperature=temp,
            pad_token_id=tokenizer.eos_token_id
        )
    
    input_length = inputs.input_ids.shape[1]
    return tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True).strip()

def generate_main_text(model, prompt_text, max_tokens=1024, temp=0.4):
    # Перемикаємося на головний адаптер
    if isinstance(model, PeftModel):
        model.set_adapter("main")
        
    messages = [{"role": "user", "content": prompt_text}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([text], return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            max_new_tokens=max_tokens,
            temperature=temp,
            pad_token_id=tokenizer.eos_token_id
        )
    
    input_length = inputs.input_ids.shape[1]
    return tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True).strip()

# ==========================================
# 4. RETRIEVAL (RAG)
# ==========================================
def hybrid_search(query_text, tags):
    if not query_text:
        return ""
        
    scored_chunks = []
    query_words = [w for w in query_text.lower().split() if len(w) > 3]
    
    for chunk in KNOWLEDGE_BASE:
        tag_score = len(set(tags) & set(chunk.get('tags', []))) * 5
        text_score = 0
        desc_lower = chunk.get('description', '').lower()
        content_lower = chunk.get('content', '').lower()
        
        for word in query_words:
            if word in desc_lower:
                text_score += 3
            elif word in content_lower:
                text_score += 1
                
        total_score = tag_score + text_score
        if total_score > 0:
            scored_chunks.append((total_score, chunk))

    scored_chunks.sort(key=lambda x: x[0], reverse=True)
    
    final_context = ""
    for score, chunk in scored_chunks[:3]:
        final_context += f"--- {chunk.get('file', 'Document')} ---\n{chunk.get('content', '')}\n\n"
        
    return final_context

SESSION_CONTEXT = ""

# ==========================================
# 5. PIPELINE: TAGGER -> MEMORY -> MAIN AGENT
# ==========================================
def process_pipeline(user_input):
    global SESSION_CONTEXT
    print("\n" + "="*60)
    
    # === STEP 1: Tagger Analysis ===
    print("[AGENT: TAGGER] Analyzing payload & structuring schema...")
    tagger_response = generate_tagger_text(multi_lora_model, user_input, max_tokens=512, temp=0.1)
    
    extracted_query = user_input
    query_tags, data_tags = [], []
    extracted_data = ""
    
    try:
        json_match = re.search(r'\{.*\}', tagger_response, re.DOTALL)
        if json_match:
            parsed_json = json.loads(json_match.group(0))
            extracted_query = parsed_json.get("extracted_query", "")
            query_tags = parsed_json.get("query_tags", [])
            extracted_data = parsed_json.get("extracted_data", "")
            data_tags = parsed_json.get("data_tags", [])
            print(f"  ↳ Intent: '{extracted_query}'")
    except json.JSONDecodeError:
        pass

    all_tags = list(set(query_tags + data_tags))

    # === STEP 2: Fetch New Context & Update Session ===
    print("[MEMORY] Executing hybrid context retrieval...")
    new_context = hybrid_search(extracted_query, all_tags)
    
    if new_context and new_context not in SESSION_CONTEXT:
        SESSION_CONTEXT += new_context + "\n"

    # === STEP 3: Main Agent Reasoning ===
    print("[AGENT: MAIN] Processing reasoning loop...")
    
    base_prompt = (
        f"SYSTEM: You are the Memory Controller and an intelligent AI agent. "
        f"You maintain a rolling context window. If you notice that some information in your Active Context is completely irrelevant to the current User Query (e.g. topic abruptly changed), "
        f"you MUST autonomously output <DROP_CONTEXT>irrelevant topic or text</DROP_CONTEXT> to free up tokens.\n"
        f"If you lack info, use <REQUEST_CONTEXT>keyword</REQUEST_CONTEXT>.\n\n"
        f"Active Context Window:\n{SESSION_CONTEXT if SESSION_CONTEXT else 'Empty'}\n\n"
        f"User Query: {extracted_query}\n\n"
        f"Analyze context relevance, use tags if needed, and provide the answer."
    )

    final_reply = generate_main_text(multi_lora_model, base_prompt, max_tokens=1024, temp=0.1)

    # === STEP 4: Pruning Check (Implicit Drop) ===
    drop_match = re.search(r'<DROP_CONTEXT>(.*?)</DROP_CONTEXT>', final_reply, re.IGNORECASE | re.DOTALL)
    if drop_match:
        drop_target = drop_match.group(1).strip()
        print(f"\n[ PRUNE ] Model autonomously detected irrelevant context!")
        print(f"  ↳ Target obsolete data: '{drop_target}'")
        
        SESSION_CONTEXT = SESSION_CONTEXT.replace(drop_target, "").strip()
        print("  ↳ Session Context successfully pruned. Tokens optimized.")
        
        final_reply = re.sub(r'<DROP_CONTEXT>.*?</DROP_CONTEXT>', '', final_reply, flags=re.IGNORECASE|re.DOTALL).strip()

    # === STEP 5: Multi-hop Check ===
    request_match = re.search(r'<REQUEST_CONTEXT>(.*?)</REQUEST_CONTEXT>', final_reply, re.IGNORECASE | re.DOTALL)
    if request_match:
        requested_info = request_match.group(1).strip()
        print(f"\n[ MULTI-HOP ] Model autonomously requested extended context.")
        print(f"  ↳ Internal query: '{requested_info}'")
        
        agent_tagger_response = generate_tagger_text(multi_lora_model, requested_info, max_tokens=512, temp=0.1)
        agent_tags = []
        try:
            agent_json_match = re.search(r'\{.*\}', agent_tagger_response, re.DOTALL)
            if agent_json_match:
                agent_json = json.loads(agent_json_match.group(0))
                agent_tags = list(set(agent_json.get("query_tags", []) + agent_json.get("data_tags", [])))
        except: pass
            
        print("[MEMORY] Fetching extended context...")
        extra_context = hybrid_search(requested_info, agent_tags)
        
        if extra_context:
            if extra_context not in SESSION_CONTEXT:
                SESSION_CONTEXT += extra_context + "\n"
            print("  ↳ Extended context retrieved successfully.")
            
        print("[AGENT: MAIN] Generating final response...")
        final_prompt = (
            f"Extended Context:\n{extra_context if extra_context else 'None found'}\n\n"
            f"Active Context:\n{SESSION_CONTEXT}\n\n"
            f"Original Query: {extracted_query}\n\n"
            f"Provide a direct, final answer. DO NOT use <REQUEST_CONTEXT> or <DROP_CONTEXT> tags."
        )
        final_reply = generate_main_text(multi_lora_model, final_prompt, max_tokens=1024, temp=0.4)

    # Clean up tags
    final_reply = re.sub(r'<REQUEST_CONTEXT>.*?</REQUEST_CONTEXT>', '', final_reply, flags=re.IGNORECASE|re.DOTALL)
    final_reply = re.sub(r'<DROP_CONTEXT>.*?</DROP_CONTEXT>', '', final_reply, flags=re.IGNORECASE|re.DOTALL)
    
    print("\n[ OUTPUT ]")
    print(final_reply.strip())

# ==========================================
# 6. EXECUTION ENTRY POINT
# ==========================================
if __name__ == "__main__":
    print("\n[SYSTEM] Agentic Workflow Ready. Awaiting user input.")
    while True:
        try:
            user_in = input("\nUSER >> ")
            if user_in.lower() in ['exit', 'quit']: 
                print("\n[SYSTEM] Shutting down...")
                break
            if not user_in.strip(): continue
            process_pipeline(user_in)
        except KeyboardInterrupt:
            print("\n[SYSTEM] Shutting down...")
            break