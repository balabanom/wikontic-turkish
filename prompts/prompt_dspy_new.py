import dspy
import json
import os
import re
from dotenv import load_dotenv

load_dotenv()

# ==========================================
# 1. AYARLAR VE OPENROUTER BAĞLANTISI
# ==========================================
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
MODEL_NAME = "openrouter/google/gemini-2.5-flash-lite" 

lm = dspy.LM(
    api_base="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
    model=MODEL_NAME,
    max_tokens=8192,
    temperature=0.1
)
dspy.settings.configure(lm=lm)

# ==========================================
# 2. DSPy İMZASI (SIGNATURE) VE ÖRNEKLER
# ==========================================
class KnowledgeGraphExtraction(dspy.Signature):
    """You are an algorithm designed to extract structured knowledge from texts to build a Wikidata-like knowledge graph consisting of triplets (subject, relation, object) and their qualifiers.
    
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
    girdi_metni = dspy.InputField(desc="The raw text to be analyzed.")
    bilgi_grafigi = dspy.OutputField(desc="A valid, error-free JSON object strictly containing the 'triplets' key.")

# Sistemi eğitecek örnek (Few-shot example)
examples = [
    dspy.Example(
        girdi_metni="Marie Curie (7 Kasım 1867 - 4 Temmuz 1934) radyoaktivite üzerine öncü araştırmalar yapmış bir fizikçi ve kimyagerdi. 1903'te Nobel Fizik Ödülü'nü ve 1911'de Nobel Kimya Ödülü'nü aldı.",
        bilgi_grafigi="""{
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
    ).with_inputs("girdi_metni")
]

def simple_metric(example, prediction, trace=None):
    output = prediction.bilgi_grafigi
    required_keys = ["triplets", "subject", "relation", "object", "qualifiers", "kaynak_cumle"]
    return all(key in output for key in required_keys)

# ==========================================
# 3. DSPy OPTİMİZASYON VE ÇALIŞTIRMA MODÜLÜ
# ==========================================
def process_with_dspy(text):
    sablon_dosyasi = "optimize_edilmis_sablon_dspy.json"
    kg_module = dspy.ChainOfThought(KnowledgeGraphExtraction)
    
    # 1. DURUM: Daha önce optimize edilmiş bir şablon varsa onu yükle
    if os.path.exists(sablon_dosyasi):
        print(f"[{sablon_dosyasi}] bulundu. Optimizasyon atlanıyor, hazır şablon yükleniyor...")
        kg_module.load(sablon_dosyasi)
        return kg_module(girdi_metni=text)
    
    # 2. DURUM: Dosya yoksa ilk kez optimize et ve kaydet
    else:
        print("Hazır şablon bulunamadı. Model optimize ediliyor (Bu işlem sadece ilk seferde uzun sürer)...")
        optimizer = dspy.teleprompt.BootstrapFewShot(metric=simple_metric, max_bootstrapped_demos=1)
        compiled_module = optimizer.compile(kg_module, trainset=examples)
        
        # Gelecekteki kullanımlar için derlenmiş modeli kaydet
        compiled_module.save(sablon_dosyasi)
        print(f"Optimizasyon tamamlandı! Şablon [{sablon_dosyasi}] olarak kaydedildi.")
        
        return compiled_module(girdi_metni=text)

# ==========================================
# 4. ANA AKIŞ: DOSYA OKUMA VE KAYDETME
# ==========================================
if __name__ == "__main__":
    input_file = "hedef_metin.txt"
    output_file = "sonuc_grafigi_dspy.json"

    # 1. Metni oku
    try:
        with open(input_file, "r", encoding="utf-8") as f:
            target_text = f.read().strip()
    except FileNotFoundError:
        print(f"HATA: '{input_file}' dosyası bulunamadı. Lütfen analiz edilecek metni bu dosyaya koyduğundan emin ol.")
        exit()

    print("DSPy çalışıyor ve metin analiz ediliyor. Bu işlem birkaç saniye sürebilir...\n")

    # 2. Metni DSPy'a gönder
    dspy_result = process_with_dspy(target_text)

    # 3. Çıktıyı temizle (Markdown tag'lerini kaldır)
    cleaned_output = re.sub(r'```json\n?|\n?```', '', dspy_result.bilgi_grafigi).strip()

    # 4. JSON olarak kaydet
    try:
        # Metnin düzgün bir JSON objesi olup olmadığını doğrula
        parsed_json = json.loads(cleaned_output)
        
        with open(output_file, "w", encoding="utf-8") as f:
            # ensure_ascii=False sayesinde Türkçe karakterler bozulmaz
            json.dump(parsed_json, f, ensure_ascii=False, indent=4)
            
        print(f"✅ Başarılı! Çıkarılan bilgi grafiği '{output_file}' dosyasına kaydedildi.")
        
    except json.JSONDecodeError:
        print(f"⚠️ Model tam ve geçerli bir JSON üretmedi. Üretilen ham metin '{output_file}' içine olduğu gibi kaydediliyor.")
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(cleaned_output)