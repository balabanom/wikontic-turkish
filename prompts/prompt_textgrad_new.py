import textgrad as tg
import os
import re
import json
from dotenv import load_dotenv
from textgrad.engine.openai import ChatOpenAI

load_dotenv()

# ==========================================
# 1. AYARLAR VE OPENROUTER BAĞLANTISI
# ==========================================
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

os.environ["OPENAI_API_KEY"] = OPENROUTER_API_KEY
os.environ["OPENAI_BASE_URL"] = "https://openrouter.ai/api/v1"

MODEL_NAME = "google/gemini-2.5-flash-lite"
engine = ChatOpenAI(model_string=MODEL_NAME, temperature=0.1, max_tokens=8192)
tg.set_backward_engine(engine, override=True)

# ==========================================
# 2. EĞİTİM VERİSETİ
# ==========================================
train_texts = [
    "Marie Curie (7 Kasım 1867 - 4 Temmuz 1934) radyoaktivite üzerine öncü araştırmalar yapmış bir fizikçi ve kimyagerdi. 1903'te Nobel Fizik Ödülü'nü ve 1911'de Nobel Kimya Ödülü'nü aldı.",
    "Albert Einstein (14 Mart 1879 - 18 Nisan 1955), görelilik teorisini geliştiren Alman doğumlu teorik fizikçidir. 1921 yılında fotoelektrik etki üzerine çalışmaları nedeniyle Nobel Fizik Ödülü'nü kazanmıştır."
]

# ==========================================
# 3. İLK PROMPT (BASE PROMPT)
# ==========================================
initial_system_prompt = """You are an algorithm designed to extract structured knowledge from texts to build a Wikidata-like knowledge graph consisting of triplets (subject, relation, object) and their qualifiers.
    
    - **Subject**: A named entity or a concept that describes a group of people, events, or any abstract objects that serves as the source of the relation.
    - **Relation**: A Wikidata-style predicate that connects the subject and object.
    - **Object**: A named entity or a concept that describes a group of people, events, or any abstract objects that is related to the subject.

    Additionally, some triplets may have **qualifiers** that provide more context (e.g., date, place, or other attributes). Qualifiers should have relations and object like triplets do, but instead of subject their relation connects an object and the triplet qualifier belongs to. **Qualifiers must always be attached to a triplet** and never exist as standalone triplets.

    **IMPORTANT NOTE (TURKISH OUTPUT REQUIREMENT):** Regardless of the input text's language, all extracted entities (subject, object), relations, and type labels (subject_type, object_type) MUST BE STRICTLY IN TURKISH. The JSON keys themselves must remain in English. 
    
    STRICT RULES:
    1. The output MUST BE STRICTLY in JSON format containing a "triplets" list.
    2. Each triplet dictionary MUST ONLY contain:
        - "subject": Subject entity.  
        - "relation": Relation connecting subject and object.  
        - "object": Object entity.  
        - "qualifiers": List of dictionaries, where each dictionary contains:
            - "relation": Relation connecting triplet and object,
            - "object": Object entity connected to the main triplet
        - "subject_type": a class that describes the subject 
        - "object_type": a class that describes the object 
        - "kaynak_cumle": original sentence from the text where this relationship was found
    3. Qualifiers must always be attached to a main triplet and must follow the [{'relation': '...', 'object': '...'}] structure.
    4. **TURKISH LANGUAGE REQUIREMENT:** The JSON keys must remain in English (subject, relation, etc.), BUT all extracted values corresponding to these keys (entities, relations, types) MUST BE STRICTLY IN TURKISH.
    5. NEVER compress the JSON output into a single line! DO NOT use Markdown (```json) blocks. Output pure JSON.
    """

# ==========================================
# 4. AKILLI YÖNETİCİ: OPTİMİZE ET VEYA YÜKLE
# ==========================================
def load_or_optimize_prompt(prompt_file):
    """Kayıtlı prompt varsa yükler, yoksa TextGrad eğitimini başlatır."""
    
    # DURUM 1: Dosya var, eğitimi atla
    if os.path.exists(prompt_file):
        print(f"[{prompt_file}] bulundu. Eğitim atlanıyor, hazır optimize prompt yükleniyor...")
        with open(prompt_file, "r", encoding="utf-8") as f:
            saved_prompt = f.read().strip()
            
        # TextGrad mantığı gereği yüklenen metni "Variable" nesnesine çevirmeliyiz
        return tg.Variable(
            value=saved_prompt,
            requires_grad=False, # Artık eğitime gerek yok
            role_description="The optimized system prompt instructing the LLM."
        )
    
    # DURUM 2: Dosya yok, TextGrad eğitim döngüsünü başlat
    print("Hazır prompt bulunamadı. TextGrad 'Metinsel Türev' optimizasyonu başlatılıyor...")
    
    system_prompt_var = tg.Variable(
        value=initial_system_prompt,
        requires_grad=True, # Hata yaptıkça değişecek (Eğitime açık)
        role_description="The main system prompt instructing the LLM to extract knowledge graph triplets in JSON format."
    )
    
    optimizer = tg.TGD(parameters=[system_prompt_var])
    model = tg.BlackboxLLM(engine=engine, system_prompt=system_prompt_var)
    
    for epoch, egitim_metni in enumerate(train_texts):
        print(f"--- EPOCH {epoch+1}/{len(train_texts)} ---")
        
        user_input = tg.Variable(
            value=f"Text: {egitim_metni}\nOutput:", 
            requires_grad=False, 
            role_description="The raw input text to be analyzed."
        )
        
        response = model(user_input)
        
        dinamik_degerlendirme = (
            f"Carefully review the original input text:\n'{egitim_metni}'\n\n"
            "Now evaluate the model's JSON output based on these STRICT CRITERIA:\n"
            "1. STRICT FORMAT CONTROL: The output must be a valid JSON object containing a 'triplets' list. Each object inside 'triplets' MUST ONLY contain these exact keys: 'subject', 'relation', 'object', 'qualifiers', 'subject_type', 'object_type', 'kaynak_cumle'.\n"
            "2. QUALIFIER STRUCTURE: The 'qualifiers' key must be a list containing objects with ONLY 'relation' and 'object' keys. It must be attached to the correct triplet.\n"
            "3. TURKISH CONTENT REQUIREMENT (Crucial): While the JSON keys MUST be in English, ALL extracted values (entities, relations, types) MUST be strictly in TURKISH. If any value is in English, heavily penalize it.\n"
            "4. COMPREHENSIVENESS: Have all important entities, events, and dates from the original text been successfully extracted?\n\n"
            "If the model violated the key structure, missed qualifiers, or failed the Turkish values rule, provide a strong textual gradient (criticism) instructing the system prompt to enforce these rules more aggressively. Focus only on how the system prompt should be updated."
        )
        
        loss_evaluator = tg.TextLoss(dinamik_degerlendirme)
        loss = loss_evaluator(response)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    
    # Optimizasyon bitti, şampiyon promptu kaydet
    print(f"\nOptimizasyon tamamlandı! Elde edilen mükemmel prompt [{prompt_file}] dosyasına kaydediliyor...")
    with open(prompt_file, "w", encoding="utf-8") as f:
        f.write(system_prompt_var.value)
        
    return system_prompt_var

# ==========================================
# 5. ANA AKIŞ: DOSYA OKUMA VE LLM CALL
# ==========================================
if __name__ == "__main__":
    input_file = "hedef_metin.txt"
    output_file = "sonuc_grafigi_textgrad.json"
    prompt_file = "optimize_edilmis_sablon_textgrad.txt"

    # 1. Hedef Metni Oku
    try:
        with open(input_file, "r", encoding="utf-8") as f:
            target_text = f.read().strip()
    except FileNotFoundError:
        print(f"HATA: '{input_file}' dosyası bulunamadı. Lütfen analiz edilecek metni bu dosyaya koy.")
        exit()

    print("\n--- TextGrad Bilgi Çıkarımı Başlıyor ---\n")

    # 2. Yöneticiyi çalıştır: Optimize et veya Yükle
    final_prompt_variable = load_or_optimize_prompt(prompt_file)

    # 3. Asıl Hedef Metin için Nihai LLM Call (Inference)
    print("\nŞampiyon prompt kullanılarak hedef metinden bilgi çıkarılıyor...")
    
    final_model = tg.BlackboxLLM(engine=engine, system_prompt=final_prompt_variable)
    test_input = tg.Variable(
        value=f"Text: {target_text}\nOutput:", 
        requires_grad=False, 
        role_description="The raw input text to be analyzed."
    )
    
    final_response = final_model(test_input)

    # 4. Çıktıyı Temizle ve JSON Olarak Kaydet
    cleaned_output = re.sub(r'```json\n?|\n?```', '', final_response.value).strip()

    try:
        parsed_json = json.loads(cleaned_output)
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(parsed_json, f, ensure_ascii=False, indent=4)
        print(f"\n✅ Başarılı! Çıkarılan bilgi grafiği '{output_file}' dosyasına kaydedildi.")
    except json.JSONDecodeError:
        print(f"\n⚠️ Model tam ve geçerli bir JSON üretmedi. Üretilen ham metin '{output_file}' içine olduğu gibi kaydediliyor.")
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(cleaned_output)