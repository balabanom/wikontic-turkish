from typing import List, Tuple, Set, Dict, Optional
from transformers import AutoTokenizer, AutoModel
from dataclasses import dataclass
from pydantic import BaseModel, ValidationError
from pymongo import MongoClient, UpdateOne
import torch
from dotenv import load_dotenv, find_dotenv
import os
from pathlib import Path
from ..profiles.runtime_profile import DEFAULT_RUNTIME_PROFILE, RuntimeProfile

_ = load_dotenv(find_dotenv())


@dataclass
class PropertyConstraints:
    subject_properties: Set[str]
    object_properties: Set[str]


class EntityAlias(BaseModel):
    _id: int
    label: str
    entity_type: str
    alias: str
    sample_id: str
    alias_text_embedding: List[float]


class Aligner:
    def __init__(
        self,
        ontology_db,
        triplets_db,
        embedding_model_name: str | None = None,
        device=None,
        runtime_profile: RuntimeProfile | None = None,
    ):
        self.ontology_db = ontology_db
        self.triplets_db = triplets_db

        profile = runtime_profile or DEFAULT_RUNTIME_PROFILE

        self.embedding_profile_id = profile.embedding_profile_id
        self.embedding_model_name = profile.embedding_model_name

        self.entity_type_collection_name = "entity_types"
        self.entity_type_aliases_collection_name = profile.entity_type_aliases_collection_name
        self.property_collection_name = "properties"
        self.property_aliases_collection_name = profile.property_aliases_collection_name

        self.entity_type_vector_index_name = profile.entity_type_vector_index_name
        self.property_vector_index_name = profile.property_vector_index_name

        self.entity_aliases_collection_name = profile.entity_aliases_collection_name
        self.triplets_collection_name = "triplets"
        self.filtered_triplets_collection_name = "filtered_triplets"
        self.ontology_filtered_triplets_collection_name = "ontology_filtered_triplets"
        self.initial_triplets_collection_name = "initial_triplets"
        self.entities_vector_index_name = profile.entity_aliases_vector_index_name

        if device is None:
            if torch.cuda.is_available():
                device = "cuda:0"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"

        self.device = torch.device(device)

        resolved_model_name = embedding_model_name or DEFAULT_RUNTIME_PROFILE.embedding_model_name
        self.tokenizer = AutoTokenizer.from_pretrained(resolved_model_name, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(
            resolved_model_name, use_safetensors=True, trust_remote_code=True
        ).to(self.device)
        self.model.eval()

    def get_embedding(self, text):

        def mean_pooling(token_embeddings, mask):
            token_embeddings = token_embeddings.masked_fill(
                ~mask[..., None].bool(), 0.0
            )
            sentence_embeddings = (
                token_embeddings.sum(dim=1) / mask.sum(dim=1)[..., None]
            )
            return sentence_embeddings

        if not text or not isinstance(text, str):
            return None

        inputs = self.tokenizer(
            [text], padding=True, truncation=True, return_tensors="pt"
        )
        with torch.no_grad():
            outputs = self.model(**inputs.to(self.device))
        embeddings = mean_pooling(outputs[0], inputs["attention_mask"])
        return embeddings.detach().cpu().tolist()[0]

    def _get_unique_similar_entity_types(
        self, target_entity_type: str, k: int = 5, max_attempts: int = 10
    ) -> List[str]:
        query_k = k * 2
        attempt = 0
        unique_ranked_entities: List[str] = []
        query_embedding = self.get_embedding(target_entity_type)
        collection = self.ontology_db.get_collection(
            self.entity_type_aliases_collection_name
        )

        while len(unique_ranked_entities) < k and attempt < max_attempts:
            search_pipeline = [
                {
                    "$vectorSearch": {
                        "index": self.entity_type_vector_index_name,
                        "queryVector": query_embedding,
                        "path": "alias_text_embedding",
                        "numCandidates": 150 if query_k < 150 else query_k,
                        "limit": query_k,
                    }
                },
                {"$project": {"_id": 0, "entity_type_id": 1}},
            ]
            result = collection.aggregate(search_pipeline)
            for res in result:
                if res["entity_type_id"] not in unique_ranked_entities:
                    unique_ranked_entities.append(res["entity_type_id"])
                if len(unique_ranked_entities) == k:
                    break
            query_k *= 2
            attempt += 1

        return unique_ranked_entities

    def retrieve_similar_entity_types(
        self, triplet: Dict[str, str], k: int = 10
    ) -> Tuple[List[str], List[str]]:

        similar_subject_types = self._get_unique_similar_entity_types(
            target_entity_type=triplet["subject_type"], k=k
        )
        if "object_type" in triplet:
            similar_object_types = self._get_unique_similar_entity_types(
                target_entity_type=triplet["object_type"], k=k
            )
        else:
            similar_object_types = []
        return similar_subject_types, similar_object_types

    def _get_valid_property_ids_by_entity_type(
        self, entity_type: str, is_object: bool = True
    ) -> Tuple[Set[str], Set[str]]:
        collection = self.ontology_db.get_collection(self.entity_type_collection_name)

        extended_types = [entity_type, "ANY"]
        hirerarchy = collection.find_one(
            {"entity_type_id": entity_type}, {"parent_type_ids": 1, "_id": 0}
        )
        extended_types.extend(hirerarchy["parent_type_ids"])

        pipeline = [
            {"$match": {"entity_type_id": {"$in": extended_types}}},
            {
                "$group": {
                    "_id": None,
                    "subject_ids": {
                        "$addToSet": {"$ifNull": ["$valid_subject_property_ids", []]}
                    },
                    "object_ids": {
                        "$addToSet": {"$ifNull": ["$valid_object_property_ids", []]}
                    },
                }
            },
            {
                "$project": {
                    "subject_ids": {
                        "$reduce": {
                            "input": "$subject_ids",
                            "initialValue": [],
                            "in": {"$setUnion": ["$$value", "$$this"]},
                        }
                    },
                    "object_ids": {
                        "$reduce": {
                            "input": "$object_ids",
                            "initialValue": [],
                            "in": {"$setUnion": ["$$value", "$$this"]},
                        }
                    },
                }
            },
        ]
        result = collection.aggregate(pipeline)
        result_data = next(result, {})

        subject_props = result_data.get("subject_ids", [])
        object_props = result_data.get("object_ids", [])

        if is_object:
            direct_props = set(object_props)
            inverse_props = set(subject_props)
        else:
            direct_props = set(subject_props)
            inverse_props = set(object_props)

        return direct_props, inverse_props

    def _get_ranked_properties(
        self,
        prop_2_direction: Dict[str, List[str]],
        target_property: str,
        k: int,
    ) -> List[Tuple[str, str]]:
        collection = self.ontology_db.get_collection(
            self.property_aliases_collection_name
        )
        query_embedding = self.get_embedding(target_property)
        if query_embedding is None:
            return []
        props = list(prop_2_direction.keys())

        query_k = k * 2
        max_attempts = 5
        attempt = 0
        unique_ranked_properties: List[str] = []

        while len(unique_ranked_properties) < k and attempt < max_attempts:
            pipeline = [
                {
                    "$vectorSearch": {
                        "index": self.property_vector_index_name,
                        "queryVector": query_embedding,
                        "path": "alias_text_embedding",
                        "numCandidates": 150 if query_k < 150 else query_k,
                        "limit": query_k,
                        "filter": {"relation_id": {"$in": props}},
                    }
                },
                {
                    "$project": {
                        "_id": 0,
                        "relation_id": 1,
                    }
                },
            ]

            similar_properties = collection.aggregate(pipeline)

            for prop in similar_properties:
                if prop["relation_id"] not in unique_ranked_properties:
                    unique_ranked_properties.append(prop["relation_id"])
                if len(unique_ranked_properties) == k:
                    break

            query_k *= 2
            attempt += 1

        unique_ranked_properties_with_direction = []
        for prop_id in unique_ranked_properties:
            for direction in prop_2_direction[prop_id]:
                unique_ranked_properties_with_direction.append((prop_id, direction))
        return unique_ranked_properties_with_direction

    def retrieve_properties_for_entity_type(
        self,
        target_relation: str,
        object_types: List[str],
        subject_types: List[str],
        k: int = 10,
    ) -> List[Tuple[str, str]]:
        direct_props = PropertyConstraints(set(), set())
        inverse_props = PropertyConstraints(set(), set())

        for obj_type in object_types:
            obj_direct, obj_inverse = self._get_valid_property_ids_by_entity_type(
                obj_type, is_object=True
            )
            direct_props.object_properties.update(obj_direct)
            inverse_props.subject_properties.update(obj_inverse)

        for subj_type in subject_types:
            subj_direct, subj_inverse = self._get_valid_property_ids_by_entity_type(
                subj_type, is_object=False
            )
            direct_props.subject_properties.update(subj_direct)
            inverse_props.object_properties.update(subj_inverse)

        valid_direct = direct_props.subject_properties & direct_props.object_properties
        valid_inverse = (
            inverse_props.subject_properties & inverse_props.object_properties
        )

        prop_id_2_direction = {prop_id: ["direct"] for prop_id in valid_direct}
        for prop_id in valid_inverse:
            if prop_id in prop_id_2_direction:
                prop_id_2_direction[prop_id].append("inverse")
            else:
                prop_id_2_direction[prop_id] = ["inverse"]

        return self._get_ranked_properties(prop_id_2_direction, target_relation, k)

    def retrieve_properties_labels_and_constraints(
        self, property_id_list: List[str]
    ) -> Dict[str, Dict[str, str]]:
        collection = self.ontology_db.get_collection(self.property_collection_name)

        pipeline = [
            {"$match": {"property_id": {"$in": property_id_list}}},
            {
                "$project": {
                    "_id": 0,
                    "property_id": 1,
                    "label": 1,
                    "valid_subject_type_ids": 1,
                    "valid_object_type_ids": 1,
                }
            },
        ]
        result = collection.aggregate(pipeline)

        result_dict = {
            item["property_id"]: {
                "label": item["label"],
                "valid_subject_type_ids": item["valid_subject_type_ids"],
                "valid_object_type_ids": item["valid_object_type_ids"],
            }
            for item in result
        }

        return result_dict

    def retrieve_entity_type_labels(self, entity_type_ids: List[str]):
        collection = self.ontology_db.get_collection(self.entity_type_collection_name)
        pipeline = [
            {"$match": {"entity_type_id": {"$in": entity_type_ids}}},
            {
                "$project": {
                    "_id": 0,
                    "entity_type_id": 1,
                    "label": 1,
                }
            },
        ]
        result = collection.aggregate(pipeline)
        result_dict = {item["entity_type_id"]: item["label"] for item in result}
        return result_dict

    def retrieve_entity_type_hierarchy(self, entity_type: str) -> List[str]:
        collection = self.ontology_db.get_collection(self.entity_type_collection_name)
        entity_id_parent_types = collection.find_one(
            {"label": entity_type},
            {"entity_type_id": 1, "parent_type_ids": 1, "label": 1, "_id": 0},
        )
        parent_type_id_labels = collection.find(
            {"entity_type_id": {"$in": entity_id_parent_types["parent_type_ids"]}},
            {"_id": 0, "label": 1, "entity_type_id": 1},
        )
        if entity_id_parent_types:
            extended_types = [entity_id_parent_types["entity_type_id"]] + [
                item["entity_type_id"] for item in parent_type_id_labels
            ]
        return extended_types

    def retrieve_entity_by_type(self, entity_name, entity_type, sample_id, k=10):
        collection = self.ontology_db.get_collection(self.entity_type_collection_name)
        entity_id_parent_types = collection.find_one(
            {"label": entity_type},
            {"entity_type_id": 1, "parent_type_ids": 1, "label": 1, "_id": 0},
        )
        extended_types = [
            entity_id_parent_types["entity_type_id"]
        ] + entity_id_parent_types["parent_type_ids"]
        extended_types = [
            elem["label"]
            for elem in collection.find(
                {"entity_type_id": {"$in": extended_types}},
                {"_id": 0, "label": 1, "entity_type_id": 1},
            )
        ]

        collection = self.triplets_db.get_collection(
            self.entity_aliases_collection_name
        )

        query_embedding = self.get_embedding(entity_name)
        if query_embedding is None:
            return {}

        if not sample_id:
            filter_query = {
                "entity_type": {"$in": extended_types},
            }
        else:
            filter_query = {
                "entity_type": {"$in": extended_types},
                "sample_id": {"$eq": sample_id},
            }
        pipeline = [
            {
                "$vectorSearch": {
                    "index": self.entities_vector_index_name,
                    "queryVector": query_embedding,
                    "path": "alias_text_embedding",
                    "numCandidates": 150 if k < 150 else k,
                    "limit": k,
                    "filter": filter_query,
                }
            },
            {"$project": {"_id": 0, "label": 1, "alias": 1}},
        ]

        result = collection.aggregate(pipeline)
        result_dict = {item["alias"]: item["label"] for item in result}
        return result_dict

    def add_entity(self, entity_name, alias, entity_type, sample_id):
        collection = self.triplets_db.get_collection(
            self.entity_aliases_collection_name
        )
        if not sample_id:
            sample_id = "all"

        if not collection.find_one(
            {
                "label": entity_name,
                "entity_type": entity_type,
                "alias": alias,
                "sample_id": {"$eq": sample_id},
            }
        ):
            collection.insert_one(
                {
                    "label": entity_name,
                    "entity_type": entity_type,
                    "alias": alias,
                    "sample_id": sample_id,
                    "alias_text_embedding": self.get_embedding(alias),
                }
            )

    def add_triplets(self, triplets_list, sample_id):
        collection = self.triplets_db.get_collection(self.triplets_collection_name)

        operations = []
        if not sample_id:
            sample_id = "all"
        for triple in triplets_list:
            triple["sample_id"] = sample_id
            triple["embedding_profile_id"] = self.embedding_profile_id
            triple["embedding_model_name"] = self.embedding_model_name
            filter_query = {
                "subject": triple["subject"],
                "relation": triple["relation"],
                "object": triple["object"],
                "subject_type": triple["subject_type"],
                "object_type": triple["object_type"],
                "sample_id": triple["sample_id"],
            }
            operations.append(
                UpdateOne(filter_query, {"$setOnInsert": triple}, upsert=True)
            )

        if operations:
            collection.bulk_write(operations)

    def add_filtered_triplets(self, triplets_list, sample_id):
        collection = self.triplets_db.get_collection(
            self.filtered_triplets_collection_name
        )

        operations = []
        if not sample_id:
            sample_id = "all"
        for triple in triplets_list:
            triple["sample_id"] = sample_id
            triple["embedding_profile_id"] = self.embedding_profile_id
            triple["embedding_model_name"] = self.embedding_model_name
            filter_query = {
                "subject": triple["subject"],
                "relation": triple["relation"],
                "object": triple["object"],
                "subject_type": triple["subject_type"],
                "object_type": triple["object_type"],
                "sample_id": triple["sample_id"],
            }
            operations.append(
                UpdateOne(filter_query, {"$setOnInsert": triple}, upsert=True)
            )

        if operations:
            collection.bulk_write(operations)

    def add_ontology_filtered_triplets(self, triplets_list, sample_id):
        collection = self.triplets_db.get_collection(
            self.ontology_filtered_triplets_collection_name
        )

        operations = []
        if not sample_id:
            sample_id = "all"
        for triple in triplets_list:
            triple["sample_id"] = sample_id
            triple["embedding_profile_id"] = self.embedding_profile_id
            triple["embedding_model_name"] = self.embedding_model_name
            filter_query = {
                "subject": triple["subject"],
                "relation": triple["relation"],
                "object": triple["object"],
                "subject_type": triple["subject_type"],
                "object_type": triple["object_type"],
                "sample_id": triple["sample_id"],
            }
            operations.append(
                UpdateOne(filter_query, {"$setOnInsert": triple}, upsert=True)
            )

        if operations:
            collection.bulk_write(operations)

    def add_initial_triplets(self, triplets_list, sample_id):
        if not sample_id:
            sample_id = "all"
        collection = self.triplets_db.get_collection(
            self.initial_triplets_collection_name
        )
        operations = []
        for triple in triplets_list:
            triple["sample_id"] = sample_id
            triple["embedding_profile_id"] = self.embedding_profile_id
            triple["embedding_model_name"] = self.embedding_model_name
            filter_query = {
                "subject": triple["subject"],
                "relation": triple["relation"],
                "object": triple["object"],
                "subject_type": triple["subject_type"],
                "object_type": triple["object_type"],
                "sample_id": triple["sample_id"],
            }
            operations.append(
                UpdateOne(filter_query, {"$setOnInsert": triple}, upsert=True)
            )
        if operations:
            collection.bulk_write(operations)

    def retrieve_similar_entity_names(
        self, entity_name: str, k: int = 10, sample_id: str = None
    ) -> List[Dict[str, str]]:
        embedded_query = self.get_embedding(entity_name)
        if embedded_query is None:
            return []
        collection = self.triplets_db.get_collection(
            self.entity_aliases_collection_name
        )

        if sample_id:
            pipeline = [
                {
                    "$vectorSearch": {
                        "index": self.entities_vector_index_name,
                        "queryVector": embedded_query,
                        "path": "alias_text_embedding",
                        "numCandidates": 150,
                        "limit": k,
                        "filter": {
                            "sample_id": {"$eq": sample_id},
                        },
                    }
                },
                {"$project": {"_id": 0, "label": 1, "entity_type": 1}},
            ]
        else:
            pipeline = [
                {
                    "$vectorSearch": {
                        "index": self.entities_vector_index_name,
                        "queryVector": embedded_query,
                        "path": "alias_text_embedding",
                        "numCandidates": 150,
                        "limit": k,
                    }
                },
                {"$project": {"_id": 0, "label": 1, "entity_type": 1}},
            ]

        result = collection.aggregate(pipeline)
        result_list = list(result)
        result_dict = [{"entity": item["label"]} for item in result_list]
        return result_dict

    # ── Ontology Neighborhood ─────────────────────────────────────────────────

    def get_ontology_neighborhood(
        self, entity_type_label: str, max_properties: int = 15
    ) -> Optional[Dict]:
        """
        Return the 1-hop ontology neighborhood for the given entity type label.

        Returns:
        {
            "center": {"id": ..., "label": ...},
            "parents": [{"id": ..., "label": ...}, ...],
            "properties": [
                {
                    "id": ...,
                    "label": ...,
                    "direction": "subject" | "object",  # whether this type acts as subject or object
                    "valid_subject_type_ids": [...],
                    "valid_object_type_ids": [...],
                },
                ...
            ]
        }
        Returns None if the entity type is not found.
        """
        try:
            et_collection = self.ontology_db.get_collection(
                self.entity_type_collection_name
            )

            center_doc = et_collection.find_one(
                {"label": entity_type_label},
                {"entity_type_id": 1, "label": 1, "parent_type_ids": 1,
                 "valid_subject_property_ids": 1, "valid_object_property_ids": 1,
                 "_id": 0},
            )
            if not center_doc:
                return None

            center_id = center_doc["entity_type_id"]
            parent_ids = center_doc.get("parent_type_ids", [])
            subj_prop_ids = center_doc.get("valid_subject_property_ids", [])
            obj_prop_ids  = center_doc.get("valid_object_property_ids", [])

            parents = []
            if parent_ids:
                parent_docs = et_collection.find(
                    {"entity_type_id": {"$in": parent_ids}},
                    {"entity_type_id": 1, "label": 1, "_id": 0},
                )
                parents = [
                    {"id": d["entity_type_id"], "label": d.get("label", d["entity_type_id"])}
                    for d in parent_docs
                ]

            prop_collection = self.ontology_db.get_collection(
                self.property_collection_name
            )

            # Build a direction map: property_id → "subject" | "object".
            # Subject properties come first; object properties fill remaining slots.
            prop_direction_map: Dict[str, str] = {}
            all_prop_ids = []

            for pid in subj_prop_ids:
                if pid not in prop_direction_map:
                    prop_direction_map[pid] = "subject"
                    all_prop_ids.append(pid)
            for pid in obj_prop_ids:
                if pid not in prop_direction_map:
                    prop_direction_map[pid] = "object"
                    all_prop_ids.append(pid)

            all_prop_ids = all_prop_ids[:max_properties]

            properties = []
            if all_prop_ids:
                prop_docs = prop_collection.find(
                    {"property_id": {"$in": all_prop_ids}},
                    {"property_id": 1, "label": 1,
                     "valid_subject_type_ids": 1, "valid_object_type_ids": 1,
                     "_id": 0},
                )
                for d in prop_docs:
                    pid = d["property_id"]
                    properties.append({
                        "id": pid,
                        "label": d.get("label", pid),
                        "direction": prop_direction_map.get(pid, "subject"),
                        "valid_subject_type_ids": d.get("valid_subject_type_ids", []),
                        "valid_object_type_ids": d.get("valid_object_type_ids", []),
                    })

            return {
                "center": {"id": center_id, "label": entity_type_label},
                "parents": parents,
                "properties": properties,
            }

        except Exception:
            return None
