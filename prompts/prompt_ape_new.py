import os
import json
import re
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# ==========================================
# 1. AYARLAR VE OPENROUTER BAGLANTISI
# ==========================================
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
MODEL_NAME = "google/gemini-2.5-flash-lite"

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
)

# APE Optimizasyonu için kullanılacak örnek veri
train_examples = [
    {
        "input": "Marie Curie (7 Kasım 1867 - 4 Temmuz 1934) radyoaktivite üzerine öncü araştırmalar yapmış bir fizikçi ve kimyagerdi. 1903'te Nobel Fizik Ödülü'nü ve 1911'de Nobel Kimya Ödülü'nü aldı.",
        "output": """{
            "triplets": [
                {"subject": "Marie Curie", "relation": "doğum tarihi", "object": "7 Kasım 1867", "qualifiers": [], "subject_type": "insan", "object_type": "tarih", "kaynak_cumle": "Marie Curie (7 Kasım 1867 - 4 Temmuz 1934) radyoaktivite üzerine öncü araştırmalar yapmış bir fizikçi ve kimyagerdi. 1903'te Nobel Fizik Ödülü'nü ve 1911'de Nobel Kimya Ödülü'nü aldı."},
                {"subject": "Marie Curie", "relation": "ölüm tarihi", "object": "4 Temmuz 1934", "qualifiers": [], "subject_type": "insan", "object_type": "tarih", "kaynak_cumle": "Marie Curie (7 Kasım 1867 - 4 Temmuz 1934) radyoaktivite üzerine öncü araştırmalar yapmış bir fizikçi ve kimyagerdi. 1903'te Nobel Fizik Ödülü'nü ve 1911'de Nobel Kimya Ödülü'nü aldı."},
                {"subject": "Marie Curie", "relation": "meslek", "object": "fizikçi", "qualifiers": [], "subject_type": "insan", "object_type": "meslek", "kaynak_cumle": "Marie Curie (7 Kasım 1867 - 4 Temmuz 1934) radyoaktivite üzerine öncü araştırmalar yapmış bir fizikçi ve kimyagerdi. 1903'te Nobel Fizik Ödülü'nü ve 1911'de Nobel Kimya Ödülü'nü aldı."},
                {"subject": "Marie Curie", "relation": "meslek", "object": "kimyager", "qualifiers": [], "subject_type": "insan", "object_type": "meslek", "kaynak_cumle": "Marie Curie (7 Kasım 1867 - 4 Temmuz 1934) radyoaktivite üzerine öncü araştırmalar yapmış bir fizikçi ve kimyagerdi. 1903'te Nobel Fizik Ödülü'nü ve 1911'de Nobel Kimya Ödülü'nü aldı."},
                {"subject": "Marie Curie", "relation": "çalışma alanı", "object": "radyoaktivite", "qualifiers": [], "subject_type": "insan", "object_type": "fiziksel fenomen", "kaynak_cumle": "Marie Curie (7 Kasım 1867 - 4 Temmuz 1934) radyoaktivite üzerine öncü araştırmalar yapmış bir fizikçi ve kimyagerdi. 1903'te Nobel Fizik Ödülü'nü ve 1911'de Nobel Kimya Ödülü'nü aldı."},
                {"subject": "Marie Curie", "relation": "kazandığı ödül", "object": "Nobel Fizik Ödülü", "qualifiers": [{"relation": "zaman noktası", "object": "1903"}], "subject_type": "insan", "object_type": "ödül", "kaynak_cumle": "Marie Curie (7 Kasım 1867 - 4 Temmuz 1934) radyoaktivite üzerine öncü araştırmalar yapmış bir fizikçi ve kimyagerdi. 1903'te Nobel Fizik Ödülü'nü ve 1911'de Nobel Kimya Ödülü'nü aldı."},
                {"subject": "Marie Curie", "relation": "kazandığı ödül", "object": "Nobel Kimya Ödülü", "qualifiers": [{"relation": "zaman noktası", "object": "1911"}], "subject_type": "insan", "object_type": "ödül", "kaynak_cumle": "Marie Curie (7 Kasım 1867 - 4 Temmuz 1934) radyoaktivite üzerine öncü araştırmalar yapmış bir fizikçi ve kimyagerdi. 1903'te Nobel Fizik Ödülü'nü ve 1911'de Nobel Kimya Ödülü'nü aldı."}
            ]
        }"""
    }
]

BASE_PROMPT = """
    You are an algorithm designed to extract structured knowledge from texts to build a Wikidata-like knowledge graph consisting of triplets (subject, relation, object) and their qualifiers.
    
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
# 2. APE ADIM 1: ADAY TALIMAT URETIMI
# ==========================================
def generate_prompt_candidates(examples, num_candidates=3):
    examples_str = ""
    for i, ex in enumerate(examples):
        examples_str += f"Example {i+1}:\nInput: {ex['input']}\nOutput: {ex['output']}\n\n"

    proposal_prompt = f"""You are an expert Prompt Engineer. Your task is to OPTIMIZE and IMPROVE a base system prompt to ensure it perfectly transforms text inputs into desired JSON knowledge graphs.

Here is the Base Prompt provided by the user:
---
{BASE_PROMPT}
---

Review the following ideal input-output examples that the final prompt must be able to generate perfectly:
{examples_str}

Please act as an optimizer. Write {num_candidates} different candidate system prompts that improve upon the Base Prompt. Make the rules stricter, clearer, and more robust to prevent any LLM hallucinations. All instructions in your candidate prompts MUST be written in English. Ensure the "TURKISH OUTPUT REQUIREMENT" is heavily emphasized in your candidates.

Make sure to EXPLICITLY instruct the model in your candidate prompts to ALWAYS output the "qualifiers" key. Tell the model: 'If there are no qualifiers, you MUST still include the "qualifiers" key with an empty list [].

Output your response strictly in the following format:
Candidate 1: [Improved Prompt text]
Candidate 2: [Improved Prompt text]
"""

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": proposal_prompt}],
        temperature=0.7 
    )
    
    content = response.choices[0].message.content
    kaba_adaylar = re.split(r"Candidate \d+:", content)
    candidates = [c.strip() for c in kaba_adaylar if c.strip()]
    return candidates[:num_candidates]

# ==========================================
# 3. APE ADIM 2: LLM CALL VE DEGERLENDIRME
# ==========================================
def extract_triplets(system_prompt, text_input):
    """Nihai LLM çağrısını yapan ve JSON metnini çıkaran ana fonksiyon."""
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Text: {text_input}\nOutput:"}
        ],
        temperature=0.1 
    )
    return response.choices[0].message.content

def evaluate_candidate(candidate_prompt, test_input):
    """Aday promptu test eder ve bir hakem LLM ile puanlar."""
    extracted_output = extract_triplets(candidate_prompt, test_input)

    evaluation_prompt = f"""You are evaluating a knowledge graph extraction system.
    
Input Text: {test_input}
System Output: {extracted_output}

Rules for Evaluation:
1. Is the output strictly a valid JSON object containing a "triplets" list?
2. Are all required keys present: 'subject', 'relation', 'object', 'qualifiers', 'subject_type', 'object_type', 'kaynak_cumle'?
3. Are ALL extracted values (entities, relations, types) strictly in Turkish? (Keys must be English).
4. Are all details from the input text comprehensively extracted?
5. Are the relations logical verbs or phrases?

Based on these rules, give the system output a score between 0 and 100. Your output MUST BE ONLY A NUMBER. Do not write any other explanation."""

    eval_response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": evaluation_prompt}],
        temperature=0.1,
        max_tokens=8192
    )
    
    try:
        score = float(eval_response.choices[0].message.content.strip())
    except ValueError:
        score = 0.0 

    return score

# ==========================================
# 4. AKILLI OPTIMIZASYON VE CALISTIRMA YONETICISI
# ==========================================
def run_ape_pipeline(target_text, prompt_dosyasi):
    """Dosya varsa şampiyonu yükler, yoksa optimizasyon sürecini başlatır."""
    
    # 1. DURUM: Şampiyon prompt önceden kaydedilmişse (Optimizasyonu atla)
    if os.path.exists(prompt_dosyasi):
        print(f"[{prompt_dosyasi}] bulundu. Optimizasyon atlanıyor, hazır şampiyon prompt yükleniyor...")
        with open(prompt_dosyasi, "r", encoding="utf-8") as f:
            best_prompt = f.read().strip()
            
    # 2. DURUM: Dosya yoksa ilk kez optimize et ve kaydet
    else:
        print("Hazır şampiyon prompt bulunamadı. APE optimizasyonu başlatılıyor (Bu işlem birkaç dakika sürebilir)...")
        candidates = generate_prompt_candidates(train_examples, num_candidates=3)
        best_score = 0.0
        best_prompt = candidates[0] if candidates else BASE_PROMPT
        
        for i, candidate in enumerate(candidates):
            print(f"Aday {i+1} test ediliyor...")
            score = evaluate_candidate(candidate, target_text)
            print(f"Aday {i+1} Puanı: {score}/100")
            
            if score > best_score:
                best_score = score
                best_prompt = candidate
                
        print(f"\nOptimizasyon tamamlandı! Şampiyon prompt kaydediliyor (Puan: {best_score}/100)...")
        with open(prompt_dosyasi, "w", encoding="utf-8") as f:
            f.write(best_prompt)

    # Nihai şampiyon prompt ile asıl hedef metni işle
    print("\nŞampiyon prompt kullanılarak hedef metinden bilgi çıkarılıyor...")
    final_output = extract_triplets(best_prompt, target_text)
    return final_output

# ==========================================
# 5. ANA AKIŞ: DOSYA OKUMA VE JSON ÜRETİMİ
# ==========================================
if __name__ == "__main__":
    input_file = "hedef_metin.txt"
    output_file = "sonuc_grafigi_ape.json"
    prompt_file = "optimize_edilmis_sablon_ape.txt"

    # 1. Hedef Metni Oku
    try:
        with open(input_file, "r", encoding="utf-8") as f:
            target_text = f.read().strip()
    except FileNotFoundError:
        print(f"HATA: '{input_file}' dosyası bulunamadı. Lütfen analiz edilecek metni bu dosyaya koyduğundan emin ol.")
        exit()

    print("\n--- APE Bilgi Çıkarımı Başlıyor ---\n")

    # 2. APE Sürecini Çalıştır (Kaydet/Yükle mantığı ile)
    raw_output = run_ape_pipeline(target_text, prompt_file)

    # 3. Çıktıyı Temizle ve JSON Olarak Kaydet
    cleaned_output = re.sub(r'```json\n?|\n?```', '', raw_output).strip()

    try:
        parsed_json = json.loads(cleaned_output)
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(parsed_json, f, ensure_ascii=False, indent=4)
        print(f"\n✅ Başarılı! Çıkarılan bilgi grafiği '{output_file}' dosyasına kaydedildi.")
    except json.JSONDecodeError:
        print(f"\n⚠️ Model tam ve geçerli bir JSON üretmedi. Üretilen ham metin '{output_file}' içine olduğu gibi kaydediliyor.")
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(cleaned_output)