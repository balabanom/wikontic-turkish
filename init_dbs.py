import os
from dotenv import load_dotenv, find_dotenv
from pymongo import MongoClient
from pymongo.errors import CollectionInvalid

from wikontic.create_wikidata_ontology_db import create_wikidata_ontology_database
from wikontic.create_ontological_triplets_db import create_ontological_triplets_database

load_dotenv(find_dotenv())

mongo_uri = os.getenv("MONGO_URI", "mongodb://localhost:27018/?directConnection=true")
client = MongoClient(mongo_uri)

# ---------------------------------------------------------
# 1) Ontology DB (SIFIRDAN KUR)
# ---------------------------------------------------------
create_wikidata_ontology_database(
    mongo_uri=mongo_uri,
    database="wikidata_ontology",
    drop_collections=True,   # <-- Ontology DB'yi her seferinde tazeler
)

# ---------------------------------------------------------
# 2) Demo Triplets DB (VARSA KORU, YOKSA OLUŞTUR)
# ---------------------------------------------------------
demo_db = client["demo"]

# Streamlit'in kullandığı/oluşturduğu temel koleksiyonlar:
required_collections = {
    "triplets",
    "entity_aliases",
    "property_aliases",
    "entity_type_aliases",
    "entity_types",
    "properties",
}

existing = set(demo_db.list_collection_names())

if required_collections.issubset(existing):
    print("✅ demo DB zaten hazır (koleksiyonlar mevcut). Demo init adımı atlandı.")
else:
    print("ℹ️ demo DB eksik görünüyor. Tamamlamaya çalışıyorum...")
    try:
        create_ontological_triplets_database(
            mongo_uri=mongo_uri,
            db_name="demo",
            drop_collections=False,  # <-- var olanı silme
        )
        print("✅ demo DB oluşturma/tamamlama başarılı.")
    except CollectionInvalid as e:
        # create_ontological_triplets_database idempotent değil:
        # Koleksiyon varsa tekrar create_collection deneyince buraya düşüyor.
        print(f"⚠️ demo DB koleksiyonları zaten var gibi görünüyor: {e}")
        print("✅ demo DB init adımı güvenli şekilde atlandı (veri korunuyor).")

print("✅ wikidata_ontology + demo DB hazır.")