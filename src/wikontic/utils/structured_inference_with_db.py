from unidecode import unidecode
import re
import warnings
from typing import Dict, List, Tuple
from langchain.tools import tool
import logging

from .base_inference_with_db import BaseInferenceWithDB
from .run_logger import start_run, log_artifact, finish_run
from .timing_utils import StageTimer

warnings.filterwarnings("ignore")
logger = logging.getLogger("StructuredInferenceWithDB")
logger.setLevel(logging.ERROR)

# ── Reason Codes ──────────────────────────────────────────────────────────────
REASON_INVALID_TRIPLET_FORMAT   = "INVALID_TRIPLET_FORMAT"
REASON_ENTITY_TYPE_NOT_FOUND    = "ENTITY_TYPE_NOT_FOUND"
REASON_PROPERTY_NOT_IN_ONTOLOGY = "PROPERTY_NOT_IN_ONTOLOGY"
REASON_ONTOLOGY_VIOLATION       = "ONTOLOGY_VIOLATION"
REASON_LLM_REFINE_FAILED        = "LLM_REFINE_FAILED"

_EXCEPTION_REASON_MAP = [
    ("format",  REASON_INVALID_TRIPLET_FORMAT),
    ("parse",   REASON_INVALID_TRIPLET_FORMAT),
    ("json",    REASON_INVALID_TRIPLET_FORMAT),
    ("key",     REASON_INVALID_TRIPLET_FORMAT),
    ("refine",  REASON_LLM_REFINE_FAILED),
    ("llm",     REASON_LLM_REFINE_FAILED),
    ("timeout", REASON_LLM_REFINE_FAILED),
    ("api",     REASON_LLM_REFINE_FAILED),
]


def _reason_from_exception(exc_text: str) -> str:
    lower = exc_text.lower()
    for keyword, code in _EXCEPTION_REASON_MAP:
        if keyword in lower:
            return code
    return REASON_LLM_REFINE_FAILED


def _reason_from_validation_msg(exception_msg: str) -> str:
    lower = exception_msg.lower()
    if "violates property constraints" in lower:
        return REASON_ONTOLOGY_VIOLATION
    if "not in candidate relations" in lower:
        return REASON_PROPERTY_NOT_IN_ONTOLOGY
    if "subject type not in candidate" in lower or "object type not in candidate" in lower:
        return REASON_ENTITY_TYPE_NOT_FOUND
    if "subject type" in lower or "object type" in lower:
        return REASON_ENTITY_TYPE_NOT_FOUND
    return REASON_PROPERTY_NOT_IN_ONTOLOGY


class StructuredInferenceWithDB(BaseInferenceWithDB):
    def __init__(self, extractor, aligner, triplets_db):
        self.extractor = extractor
        self.aligner = aligner
        self.triplets_db = triplets_db

        self.extract_triplets_with_ontology_filtering_tool = tool(
            self.extract_triplets_with_ontology_filtering
        )
        self.extract_triplets_with_ontology_filtering_and_add_to_db_tool = tool(
            self.extract_triplets_with_ontology_filtering_and_add_to_db
        )
        self.retrieve_similar_entity_names_tool = tool(
            self.retrieve_similar_entity_names
        )
        self.identify_relevant_entities_from_question_tool = tool(
            self.identify_relevant_entities_from_question_with_llm
        )
        self.get_1_hop_supporting_triplets_tool = tool(
            self.get_1_hop_supporting_triplets
        )
        self.answer_question_with_llm_tool = tool(self.answer_question_with_llm)

    def _get_model_name(self) -> str:
        for attr in ("model_name", "model", "llm_model", "model_id"):
            val = getattr(self.extractor, attr, None)
            if val and isinstance(val, str):
                return val
        return "unknown"

    def _refine_entity_types(self, text, triplet):
        candidate_subj_type_ids, candidate_obj_type_ids = (
            self.aligner.retrieve_similar_entity_types(triplet=triplet)
        )
        candidate_entity_type_id_2_label = self.aligner.retrieve_entity_type_labels(
            candidate_subj_type_ids + candidate_obj_type_ids
        )
        candidate_entity_type_label_2_id = {
            entity_label: entity_id
            for entity_id, entity_label in candidate_entity_type_id_2_label.items()
        }
        candidate_subject_types = [
            candidate_entity_type_id_2_label[t] for t in candidate_subj_type_ids
        ]
        candidate_object_types = [
            candidate_entity_type_id_2_label[t] for t in candidate_obj_type_ids
        ]

        if (
            triplet["subject_type"] in candidate_subject_types
            and triplet["object_type"] in candidate_object_types
        ):
            refined_subject_type    = triplet["subject_type"]
            refined_object_type     = triplet["object_type"]
            refined_subject_type_id = candidate_entity_type_label_2_id[triplet["subject_type"]]
            refined_object_type_id  = candidate_entity_type_label_2_id[triplet["object_type"]]
        else:
            if triplet["subject_type"] in candidate_subject_types:
                candidate_subject_types = [triplet["subject_type"]]
            if triplet["object_type"] in candidate_object_types:
                candidate_object_types = [triplet["object_type"]]

            self.extractor.reset_error_state()
            refined_entity_types = self.extractor.refine_entity_types(
                text=text,
                triplet=triplet,
                candidate_subject_types=candidate_subject_types,
                candidate_object_types=candidate_object_types,
            )
            refined_subject_type = refined_entity_types["subject_type"]
            refined_object_type  = refined_entity_types["object_type"]

            refined_subject_type_id = (
                candidate_entity_type_label_2_id[refined_subject_type]
                if refined_subject_type in candidate_subject_types
                else None
            )
            refined_object_type_id = (
                candidate_entity_type_label_2_id[refined_object_type]
                if refined_object_type in candidate_object_types
                else None
            )

        return (
            refined_subject_type,
            refined_subject_type_id,
            refined_object_type,
            refined_object_type_id,
        )

    def _get_candidate_entity_properties(
        self, triplet: Dict[str, str], subj_type_ids: List[str], obj_type_ids: List[str]
    ) -> Tuple[List[Tuple[str, str]], Dict[str, dict]]:
        properties: List[Tuple[str, str]] = (
            self.aligner.retrieve_properties_for_entity_type(
                target_relation=triplet["relation"],
                object_types=obj_type_ids,
                subject_types=subj_type_ids,
                k=10,
            )
        )
        prop_2_label_and_constraint = (
            self.aligner.retrieve_properties_labels_and_constraints(
                property_id_list=[p[0] for p in properties]
            )
        )
        return properties, prop_2_label_and_constraint

    def _refine_relation(self, text, triplet, refined_subject_type_id, refined_object_type_id):
        if refined_subject_type_id and refined_object_type_id:
            relation_direction_candidate_pairs, prop_2_label_and_constraint = (
                self._get_candidate_entity_properties(
                    triplet=triplet,
                    subj_type_ids=[refined_subject_type_id],
                    obj_type_ids=[refined_object_type_id],
                )
            )
            candidate_relations = [
                prop_2_label_and_constraint[p[0]]["label"]
                for p in relation_direction_candidate_pairs
            ]
            if triplet["relation"] in candidate_relations:
                refined_relation = triplet["relation"]
            else:
                self.extractor.reset_error_state()
                refined_relation = self.extractor.refine_relation(
                    text=text, triplet=triplet, candidate_relations=candidate_relations
                )["relation"]
        else:
            refined_relation    = triplet["relation"]
            candidate_relations = []

        if refined_relation in candidate_relations:
            refined_relation_id_candidates = [
                p_id
                for p_id in prop_2_label_and_constraint
                if prop_2_label_and_constraint[p_id]["label"] == refined_relation
            ]
            refined_relation_id = refined_relation_id_candidates[0]
            refined_relation_directions = [
                p[1]
                for p in relation_direction_candidate_pairs
                if p[0] == refined_relation_id
            ]
            refined_relation_direction = (
                "direct" if "direct" in refined_relation_directions else "inverse"
            )
            prop_subject_type_ids = [
                prop_2_label_and_constraint[prop]["valid_subject_type_ids"]
                for prop in prop_2_label_and_constraint
                if prop_2_label_and_constraint[prop]["label"] == refined_relation
            ][0]
            prop_object_type_ids = [
                prop_2_label_and_constraint[prop]["valid_object_type_ids"]
                for prop in prop_2_label_and_constraint
                if prop_2_label_and_constraint[prop]["label"] == refined_relation
            ][0]
        else:
            refined_relation_direction = "direct"
            refined_relation_id        = None
            prop_subject_type_ids      = []
            prop_object_type_ids       = []

        return (
            refined_relation,
            refined_relation_id,
            refined_relation_direction,
            prop_subject_type_ids,
            prop_object_type_ids,
        )

    def _validate_backbone(
        self,
        refined_subject_type,
        refined_object_type,
        refined_relation,
        refined_object_type_id,
        refined_subject_type_id,
        refined_relation_id,
        valid_subject_type_ids,
        valid_object_type_ids,
    ):
        exception_msg = ""
        if not refined_relation_id:
            exception_msg += "Refined relation not in candidate relations\n"
        if not refined_subject_type_id:
            exception_msg += "Refined subject type not in candidate subject types\n"
        if not refined_object_type_id:
            exception_msg += "Refined object type not in candidate object types\n"

        if exception_msg != "":
            return False, exception_msg

        subject_type_hierarchy = self.aligner.retrieve_entity_type_hierarchy(refined_subject_type)
        object_type_hierarchy  = self.aligner.retrieve_entity_type_hierarchy(refined_object_type)

        if valid_subject_type_ids == ["ANY"]:
            valid_subject_type_ids = subject_type_hierarchy
        if valid_object_type_ids == ["ANY"]:
            valid_object_type_ids = object_type_hierarchy

        if (
            any([t in subject_type_hierarchy for t in valid_subject_type_ids])
            and any([t in object_type_hierarchy for t in valid_object_type_ids])
        ):
            return True, exception_msg
        else:
            exception_msg += "Triplet backbone violates property constraints\n"
            return False, exception_msg

    def _refine_entity_name(self, text, triplet, sample_id, is_object=False):
        self.extractor.reset_error_state()
        if is_object:
            entity           = unidecode(triplet["object"])
            entity_type      = triplet["object_type"]
            entity_hierarchy = self.aligner.retrieve_entity_type_hierarchy(entity_type)
        else:
            entity           = unidecode(triplet["subject"])
            entity_type      = triplet["subject_type"]
            entity_hierarchy = []

        if any([t in ["Q186408", "Q309314"] for t in entity_hierarchy]):
            updated_entity = entity
        else:
            similar_entities = self.aligner.retrieve_entity_by_type(
                entity_name=entity, entity_type=entity_type, sample_id=sample_id
            )
            if len(similar_entities) > 0:
                if entity in similar_entities:
                    updated_entity = similar_entities[entity]
                else:
                    updated_entity = self.extractor.refine_entity(
                        text=text,
                        triplet=triplet,
                        candidates=list(similar_entities.values()),
                        is_object=is_object,
                    )
                    updated_entity = unidecode(updated_entity)
                    if re.sub(r"[^\w\s]", "", updated_entity) == "None":
                        updated_entity = entity
            else:
                updated_entity = entity

        self.aligner.add_entity(
            entity_name=updated_entity,
            alias=entity,
            entity_type=entity_type,
            sample_id=sample_id,
        )
        return updated_entity

    def extract_triplets_with_ontology_filtering(
        self, text, sample_id=None, source_text_id=None, run_id=None, timer=None
    ):
        """
        Extract and refine knowledge graph triplets from text using LLM.

        Args:
            text (str):           Input text to extract triplets from
            sample_id (str):      Sample ID
            source_text_id (str): Optional source text identifier
            run_id (str):         Optional run ID for trace logging
            timer (StageTimer):   Optional shared timer from outer function
        Returns:
            tuple: (initial_triplets, final_triplets, filtered_triplets,
                    ontology_filtered_triplets)
        """
        self.extractor.reset_tokens()
        self.extractor.reset_messages()
        self.extractor.reset_error_state()

        # ── Stage: llm_extract ────────────────────────────────────────────────
        if timer:
            with timer.measure("llm_extract"):
                extracted_triplets = self.extractor.extract_triplets_from_text(text)
            # Token usage — SDK response'dan al (best-effort)
            raw_response = getattr(self.extractor, "_last_response", None)
            usage = getattr(raw_response, "usage", None) if raw_response else None
            timer.record_token_usage(usage)
        else:
            extracted_triplets = self.extractor.extract_triplets_from_text(text)

        # ── Artifact: raw_llm_output ──────────────────────────────────────────
        if run_id:
            try:
                log_artifact(
                    run_id,
                    "raw_llm_output",
                    {
                        "text":   str(extracted_triplets),
                        "type":   type(extracted_triplets).__name__,
                        "format": "string",
                    },
                )
            except Exception as log_exc:
                logger.warning("run_logger raw_llm_output failed: %s", log_exc)

        # ── Stage: parse ──────────────────────────────────────────────────────
        initial_triplets = []

        def _parse():
            for triplet in extracted_triplets["triplets"]:
                triplet["prompt_token_num"], triplet["completion_token_num"] = (
                    self.extractor.calculate_used_tokens()
                )
                triplet["source_text_id"] = source_text_id
                triplet["sample_id"]      = sample_id
                initial_triplets.append(triplet.copy())

        if timer:
            with timer.measure("parse"):
                _parse()
        else:
            _parse()

        # ── Artifact: parsed_triplets ─────────────────────────────────────────
        if run_id:
            try:
                log_artifact(
                    run_id,
                    "parsed_triplets",
                    {
                        "triplets": [
                            {
                                "subject":  t.get("subject"),
                                "relation": t.get("relation"),
                                "object":   t.get("object"),
                            }
                            for t in initial_triplets
                        ],
                        "count": len(initial_triplets),
                    },
                )
            except Exception as log_exc:
                logger.warning("run_logger parsed_triplets failed: %s", log_exc)

        final_triplets             = []
        filtered_triplets          = []
        ontology_filtered_triplets = []
        entity_merge_log           = []

        # ── Stage: ontology_alignment (merge + validate, per-triplet loop) ────
        # Tüm per-triplet LLM işleri (entity type refine, relation refine,
        # entity name merge, validation) bu tek blok içinde ölçülür.
        def _process_triplets():
            for triplet in extracted_triplets["triplets"]:
                self.extractor.reset_tokens()
                try:
                    logger.log(logging.DEBUG, "Triplet: %s\n%s" % (str(triplet), "-" * 100))

                    (
                        refined_subject_type,
                        refined_subject_type_id,
                        refined_object_type,
                        refined_object_type_id,
                    ) = self._refine_entity_types(text=text, triplet=triplet)

                    (
                        refined_relation,
                        refined_relation_id,
                        refined_relation_direction,
                        prop_subject_type_ids,
                        prop_object_type_ids,
                    ) = self._refine_relation(
                        text=text,
                        triplet=triplet,
                        refined_subject_type_id=refined_subject_type_id,
                        refined_object_type_id=refined_object_type_id,
                    )

                    if refined_relation_direction == "inverse":
                        refined_subject_type_id, refined_object_type_id = (
                            refined_object_type_id,
                            refined_subject_type_id,
                        )

                    backbone_triplet = {
                        "subject": (
                            triplet["subject"]
                            if refined_relation_direction == "direct"
                            else triplet["object"]
                        ),
                        "relation": refined_relation,
                        "object": (
                            triplet["object"]
                            if refined_relation_direction == "direct"
                            else triplet["subject"]
                        ),
                        "subject_type": (
                            refined_subject_type
                            if refined_relation_direction == "direct"
                            else refined_object_type
                        ),
                        "object_type": (
                            refined_object_type
                            if refined_relation_direction == "direct"
                            else refined_subject_type
                        ),
                    }
                    backbone_triplet["qualifiers"] = triplet["qualifiers"]

                    # ── Entity name refinement + merge log ────────────────────
                    original_subject = backbone_triplet["subject"]
                    original_object  = backbone_triplet["object"]

                    if refined_subject_type_id:
                        backbone_triplet["subject"] = self._refine_entity_name(
                            text, backbone_triplet, sample_id, is_object=False
                        )
                        if backbone_triplet["subject"] != original_subject:
                            entity_merge_log.append({
                                "from":        original_subject,
                                "to":          backbone_triplet["subject"],
                                "entity_type": backbone_triplet["subject_type"],
                                "method":      "vectorSearch+LLM",
                            })

                    if refined_object_type_id:
                        backbone_triplet["object"] = self._refine_entity_name(
                            text, backbone_triplet, sample_id, is_object=True
                        )
                        if backbone_triplet["object"] != original_object:
                            entity_merge_log.append({
                                "from":        original_object,
                                "to":          backbone_triplet["object"],
                                "entity_type": backbone_triplet["object_type"],
                                "method":      "vectorSearch+LLM",
                            })

                    (
                        backbone_triplet["prompt_token_num"],
                        backbone_triplet["completion_token_num"],
                    ) = self.extractor.calculate_used_tokens()
                    backbone_triplet["source_text_id"] = source_text_id
                    backbone_triplet["sample_id"]      = sample_id

                    backbone_triplet_valid, backbone_triplet_exception_msg = (
                        self._validate_backbone(
                            backbone_triplet["subject_type"],
                            backbone_triplet["object_type"],
                            backbone_triplet["relation"],
                            refined_object_type_id,
                            refined_subject_type_id,
                            refined_relation_id,
                            prop_subject_type_ids,
                            prop_object_type_ids,
                        )
                    )

                    if backbone_triplet_valid:
                        final_triplets.append(backbone_triplet.copy())
                        logger.log(
                            logging.DEBUG,
                            "Final triplet: %s\n%s" % (str(backbone_triplet), "-" * 100),
                        )
                    else:
                        reason_code = _reason_from_validation_msg(backbone_triplet_exception_msg)
                        backbone_triplet["exception_text"] = backbone_triplet_exception_msg
                        backbone_triplet["reason_code"]    = reason_code
                        logger.log(
                            logging.ERROR,
                            "Ontology filtered [%s]: %s\n%s"
                            % (reason_code, str(backbone_triplet), "-" * 100),
                        )
                        ontology_filtered_triplets.append(backbone_triplet.copy())

                except Exception as e:
                    reason_code      = _reason_from_exception(str(e))
                    backbone_triplet = triplet.copy()
                    (
                        backbone_triplet["prompt_token_num"],
                        backbone_triplet["completion_token_num"],
                    ) = self.extractor.calculate_used_tokens()
                    backbone_triplet["source_text_id"] = source_text_id
                    backbone_triplet["sample_id"]      = sample_id
                    backbone_triplet["exception_text"] = str(e)
                    backbone_triplet["reason_code"]    = reason_code
                    filtered_triplets.append(backbone_triplet.copy())
                    logger.log(
                        logging.INFO,
                        "Filtered [%s]: %s\n%s" % (reason_code, str(backbone_triplet), "-" * 100),
                    )

        if timer:
            with timer.measure("ontology_alignment"):
                _process_triplets()
        else:
            _process_triplets()

        # ── Artifact: filtered_out ────────────────────────────────────────────
        if run_id:
            try:
                all_filtered = []
                for t in filtered_triplets:
                    all_filtered.append({
                        "subject":        t.get("subject"),
                        "relation":       t.get("relation"),
                        "object":         t.get("object"),
                        "reason_code":    t.get("reason_code", REASON_LLM_REFINE_FAILED),
                        "exception_text": t.get("exception_text", ""),
                        "filter_stage":   "pipeline_exception",
                    })
                for t in ontology_filtered_triplets:
                    all_filtered.append({
                        "subject":        t.get("subject"),
                        "relation":       t.get("relation"),
                        "object":         t.get("object"),
                        "reason_code":    t.get("reason_code", REASON_ONTOLOGY_VIOLATION),
                        "exception_text": t.get("exception_text", ""),
                        "filter_stage":   "ontology_validation",
                    })
                log_artifact(
                    run_id,
                    "filtered_out",
                    {
                        "triplets":                 all_filtered,
                        "count":                    len(all_filtered),
                        "pipeline_exception_count": len(filtered_triplets),
                        "ontology_filtered_count":  len(ontology_filtered_triplets),
                    },
                )
            except Exception as log_exc:
                logger.warning("run_logger filtered_out failed: %s", log_exc)

        # ── Artifact: merge_map_entities ──────────────────────────────────────
        if run_id:
            try:
                log_artifact(
                    run_id,
                    "merge_map_entities",
                    {
                        "merges": entity_merge_log,
                        "count":  len(entity_merge_log),
                    },
                )
            except Exception as log_exc:
                logger.warning("run_logger merge_map_entities failed: %s", log_exc)

        # ── Artifact: final_triplets ──────────────────────────────────────────
        if run_id:
            try:
                log_artifact(
                    run_id,
                    "final_triplets",
                    {
                        "triplets": [
                            {
                                "subject":      t.get("subject"),
                                "relation":     t.get("relation"),
                                "object":       t.get("object"),
                                "subject_type": t.get("subject_type"),
                                "object_type":  t.get("object_type"),
                            }
                            for t in final_triplets
                        ],
                        "count":                   len(final_triplets),
                        "filtered_count":          len(filtered_triplets),
                        "ontology_filtered_count": len(ontology_filtered_triplets),
                    },
                )
            except Exception as log_exc:
                logger.warning("run_logger final_triplets failed: %s", log_exc)

        return (
            initial_triplets,
            final_triplets,
            filtered_triplets,
            ontology_filtered_triplets,
        )

    def extract_triplets_with_ontology_filtering_and_add_to_db(
        self, text, sample_id=None, source_text_id=None
    ):
        """
        Extract and refine knowledge graph triplets from text using LLM,
        then add them to the database.

        Returns:
            tuple: (initial_triplets, final_triplets, filtered_triplets,
                    ontology_filtered_triplets, run_id)
        """
        model_name = self._get_model_name()
        run_id = start_run(
            sample_id=str(sample_id) if sample_id is not None else "unknown",
            model=model_name,
            input_text=text,
            extra_config={"source_text_id": source_text_id},
        )

        # Timer tüm pipeline boyunca yaşar, finish_run'a stats olarak geçer
        timer = StageTimer()

        try:
            (
                initial_triplets,
                final_triplets,
                filtered_triplets,
                ontology_filtered_triplets,
            ) = self.extract_triplets_with_ontology_filtering(
                text,
                sample_id=sample_id,
                source_text_id=source_text_id,
                run_id=run_id,
                timer=timer,
            )

            # ── Stage: db_write ───────────────────────────────────────────────
            with timer.measure("db_write"):
                if len(initial_triplets) > 0:
                    self.aligner.add_initial_triplets(initial_triplets, sample_id=sample_id)
                if len(final_triplets) > 0:
                    self.aligner.add_triplets(final_triplets, sample_id=sample_id)
                if len(filtered_triplets) > 0:
                    self.aligner.add_filtered_triplets(filtered_triplets, sample_id=sample_id)
                if len(ontology_filtered_triplets) > 0:
                    self.aligner.add_ontology_filtered_triplets(
                        ontology_filtered_triplets, sample_id=sample_id
                    )

            # Stats: timer süreleri + triplet count'ları birleştir
            stats = timer.to_stats()
            stats.update({
                "initial_count":           len(initial_triplets),
                "final_count":             len(final_triplets),
                "filtered_count":          len(filtered_triplets),
                "ontology_filtered_count": len(ontology_filtered_triplets),
            })

            finish_run(run_id=run_id, status="DONE", stats=stats)

        except Exception as e:
            timer.mark_failed_at("unknown")
            finish_run(
                run_id=run_id,
                status="FAILED",
                error=str(e),
                stats=timer.to_stats(),
            )
            raise

        return (
            initial_triplets,
            final_triplets,
            filtered_triplets,
            ontology_filtered_triplets,
            run_id,
        )