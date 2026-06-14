from __future__ import annotations

import json
import hashlib
from datetime import datetime
from typing import Any

import pandas as pd
import streamlit as st


st.set_page_config(
    page_title="JSON Report Viewer — Wikontic",
    page_icon="media/wikotic-wo-text.png",
    layout="wide",
)


TRIPLET_COLUMNS = [
    "subject",
    "relation",
    "object",
    "subject_type",
    "object_type",
    "sentence_id",
    "sentence_preview",
    "source_text_id",
    "sample_id",
]

FILTER_COLUMNS = [
    "subject",
    "relation",
    "object",
    "subject_type",
    "object_type",
    "reason_code",
    "filter_stage",
    "sentence_id",
    "sentence_preview",
    "exception_text",
    "source_text_id",
    "sample_id",
]


def _truncate(value: Any, length: int = 18) -> str:
    text = "—" if value in (None, "") else str(value)
    return text if len(text) <= length else f"{text[:length]}…"


def _format_datetime(value: Any) -> str:
    if not value:
        return "—"
    if isinstance(value, datetime):
        return value.strftime("%Y-%m-%d %H:%M:%S")
    text = str(value)
    try:
        normalized = text.replace("Z", "+00:00")
        return datetime.fromisoformat(normalized).strftime("%Y-%m-%d %H:%M:%S")
    except ValueError:
        return text


def _as_list(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, list):
        return [item for item in value if isinstance(item, dict)]
    return []


def _count_from(counts: dict[str, Any], *keys: str) -> int:
    for key in keys:
        value = counts.get(key)
        if isinstance(value, (int, float)):
            return int(value)
    return 0


def _triplet_df(triplets: list[dict[str, Any]], columns: list[str]) -> pd.DataFrame | None:
    if not triplets:
        return None
    df = pd.DataFrame(triplets)
    existing = [column for column in columns if column in df.columns]
    if not existing:
        return df
    return df[existing]


def _render_sentence_details(triplets: list[dict[str, Any]], key_prefix: str) -> None:
    detailed = [
        triplet for triplet in triplets
        if triplet.get("sentence_full") or triplet.get("sentence")
    ]
    if not detailed:
        return
    if st.checkbox("📖 Cümle detaylarını göster", key=f"{key_prefix}_sentences"):
        for triplet in detailed:
            sentence = triplet.get("sentence_full") or triplet.get("sentence")
            st.caption(
                f"[{triplet.get('sentence_id', '?')}] "
                f"**{triplet.get('subject', '')}** — "
                f"{triplet.get('relation', '')} — "
                f"**{triplet.get('object', '')}**"
            )
            st.info(sentence)


def _render_download(label: str, payload: Any, filename: str, key: str) -> None:
    st.download_button(
        label=label,
        data=json.dumps(payload, ensure_ascii=False, indent=2, default=str),
        file_name=filename,
        mime="application/json",
        key=key,
    )


def _paper_report_to_view(report: dict[str, Any], label: str | None = None) -> dict[str, Any]:
    run_info = report.get("run_info") or {}
    counts = report.get("triple_counts") or {}
    ontology_filtering = report.get("ontology_filtering") or {}
    pipeline_filtering = report.get("pipeline_filtering") or {}
    final_triples = _as_list(report.get("final_triples"))
    ontology_filtered = _as_list(ontology_filtering.get("triples"))
    pipeline_filtered = _as_list(pipeline_filtering.get("triples"))
    filtered = pipeline_filtered + ontology_filtered
    merge_info = report.get("merge_info") or {}

    return {
        "kind": "paper",
        "label": label or "Paper report",
        "schema_version": report.get("schema_version", "paper-report-v1"),
        "status": "DONE",
        "created_at": report.get("created_at"),
        "model": run_info.get("llm_model"),
        "sample_id": run_info.get("sample_id"),
        "run_id": run_info.get("run_id"),
        "source_url": (run_info.get("extra_config") or {}).get("source_url"),
        "profile_id": run_info.get("profile_id"),
        "ontology_language": run_info.get("ontology_language"),
        "embedding_model": run_info.get("embedding_model"),
        "prompt_type": run_info.get("prompt_type"),
        "counts": {
            "raw": _count_from(counts, "initial_raw_triple_count"),
            "parsed": _count_from(counts, "initial_raw_triple_count"),
            "final": _count_from(counts, "final_triple_count") or len(final_triples),
            "filtered": _count_from(counts, "filtered_count") or len(filtered),
            "pipeline_filtered": (
                _count_from(counts, "pipeline_filtered_count") or len(pipeline_filtered)
            ),
            "ontology_filtered": (
                _count_from(counts, "ontology_filtered_count") or len(ontology_filtered)
            ),
            "kg_inserted": _count_from(counts, "kg_inserted_count"),
            "kg_existing": _count_from(counts, "kg_already_existing_count"),
            "merges": _count_from(merge_info, "merge_count") or len(_as_list(merge_info.get("merges"))),
        },
        "raw_text": None,
        "parsed_triplets": [],
        "merge_rows": _as_list(merge_info.get("merges")),
        "filtered_triplets": filtered,
        "pipeline_filtered_triplets": pipeline_filtered,
        "ontology_filtered_triplets": ontology_filtered,
        "final_triplets": final_triples,
        "chunk_results": [],
        "relation_rows": _as_list((report.get("relation_info") or {}).get("top_relations")),
        "entity_info": report.get("entity_info") or {},
        "source_payload": report,
    }


def _batch_to_view(report: dict[str, Any]) -> dict[str, Any]:
    info = report.get("batch_info") or {}
    totals = report.get("totals") or {}
    merge_info = report.get("merge_info") or {}
    final_triples = _as_list(report.get("final_triples"))
    ontology_filtered = _as_list(report.get("ontology_filtered_triples"))

    pipeline_filtered: list[dict[str, Any]] = []
    for chunk_report in _as_list(report.get("chunk_reports")):
        pipeline_filtering = chunk_report.get("pipeline_filtering") or {}
        pipeline_filtered.extend(_as_list(pipeline_filtering.get("triples")))

    filtered = pipeline_filtered + ontology_filtered

    return {
        "kind": "batch",
        "label": "Batch toplamı",
        "schema_version": report.get("schema_version", "wiki-batch-report-v1"),
        "status": report.get("status", "—"),
        "created_at": report.get("created_at"),
        "model": info.get("llm_model"),
        "sample_id": info.get("sample_id"),
        "run_id": info.get("batch_id"),
        "source_url": info.get("source_url"),
        "profile_id": info.get("profile_id"),
        "ontology_language": info.get("ontology_language"),
        "embedding_model": info.get("embedding_model"),
        "prompt_type": info.get("prompt_type"),
        "counts": {
            "raw": _count_from(totals, "initial_raw_triple_count"),
            "parsed": _count_from(totals, "initial_raw_triple_count"),
            "final": _count_from(totals, "final_triple_count") or len(final_triples),
            "filtered": _count_from(totals, "filtered_count") or len(filtered),
            "pipeline_filtered": (
                _count_from(totals, "pipeline_filtered_count") or len(pipeline_filtered)
            ),
            "ontology_filtered": (
                _count_from(totals, "ontology_filtered_count") or len(ontology_filtered)
            ),
            "kg_inserted": _count_from(totals, "kg_inserted_count"),
            "kg_existing": _count_from(totals, "kg_already_existing_count"),
            "merges": _count_from(merge_info, "merge_count") or len(_as_list(merge_info.get("merges"))),
        },
        "raw_text": None,
        "parsed_triplets": [],
        "merge_rows": _as_list(merge_info.get("merges")),
        "filtered_triplets": filtered,
        "pipeline_filtered_triplets": pipeline_filtered,
        "ontology_filtered_triplets": ontology_filtered,
        "final_triplets": final_triples,
        "chunk_results": _as_list(report.get("chunk_results")),
        "chunk_plan": _as_list(report.get("chunk_plan")),
        "failed_chunks": _as_list(report.get("failed_chunks")),
        "relation_rows": _as_list((report.get("relation_info") or {}).get("top_relations")),
        "entity_info": report.get("entity_info") or {},
        "source_payload": report,
    }


def _export_to_view(report: dict[str, Any]) -> dict[str, Any]:
    run = report.get("run") or {}
    artifacts = report.get("artifacts") or {}
    repro = report.get("repro_config") or {}

    raw_art = artifacts.get("raw_llm_output") or {}
    parsed_art = artifacts.get("parsed_triplets") or {}
    merge_art = artifacts.get("merge_map_entities") or {}
    filtered_art = artifacts.get("filtered_out") or {}
    final_art = artifacts.get("final_triplets") or {}
    paper_art = artifacts.get("paper_report") or {}

    parsed = _as_list(parsed_art.get("triplets"))
    final = _as_list(final_art.get("triplets"))
    filtered = _as_list(filtered_art.get("triplets"))

    counts = (paper_art.get("triple_counts") if isinstance(paper_art, dict) else {}) or {}
    return {
        "kind": "export",
        "label": "Run export",
        "schema_version": report.get("schema_version"),
        "status": run.get("status"),
        "created_at": run.get("created_at"),
        "model": run.get("model") or repro.get("model"),
        "sample_id": run.get("sample_id"),
        "run_id": run.get("run_id"),
        "source_url": (run.get("extra_config") or {}).get("source_url"),
        "profile_id": run.get("profile_id") or repro.get("profile_id"),
        "ontology_language": run.get("ontology_language") or repro.get("ontology_language"),
        "embedding_model": run.get("embedding_model_name") or repro.get("embedding_model_name"),
        "prompt_type": (run.get("extra_config") or {}).get("prompt_type"),
        "counts": {
            "raw": parsed_art.get("count", len(parsed))
            or _count_from(counts, "initial_raw_triple_count"),
            "parsed": parsed_art.get("count", len(parsed)),
            "final": final_art.get("count", len(final))
            or _count_from(counts, "final_triple_count"),
            "filtered": filtered_art.get("count", len(filtered))
            or _count_from(counts, "filtered_count"),
            "pipeline_filtered": filtered_art.get("pipeline_exception_count")
            or _count_from(counts, "pipeline_filtered_count"),
            "ontology_filtered": filtered_art.get("ontology_filtered_count")
            or _count_from(counts, "ontology_filtered_count"),
            "kg_inserted": _count_from(counts, "kg_inserted_count"),
            "kg_existing": _count_from(counts, "kg_already_existing_count"),
            "merges": len(_as_list(merge_art.get("merges"))),
        },
        "raw_text": raw_art.get("text"),
        "parsed_triplets": parsed,
        "merge_rows": _as_list(merge_art.get("merges")),
        "filtered_triplets": filtered,
        "pipeline_filtered_triplets": [
            item for item in filtered
            if item.get("filter_stage") != "ontology"
        ],
        "ontology_filtered_triplets": [
            item for item in filtered
            if item.get("filter_stage") == "ontology"
        ],
        "final_triplets": final,
        "chunk_results": [],
        "relation_rows": _as_list((paper_art.get("relation_info") or {}).get("top_relations"))
        if isinstance(paper_art, dict) else [],
        "entity_info": (paper_art.get("entity_info") if isinstance(paper_art, dict) else {}) or {},
        "source_payload": report,
    }


def _generic_to_view(report: dict[str, Any]) -> dict[str, Any]:
    triplets = _as_list(report.get("triplets") or report.get("final_triples"))
    return {
        "kind": "generic",
        "label": "JSON",
        "schema_version": report.get("schema_version", "unknown"),
        "status": report.get("status", "—"),
        "created_at": report.get("created_at"),
        "model": report.get("model") or report.get("llm_model"),
        "sample_id": report.get("sample_id"),
        "run_id": report.get("run_id") or report.get("batch_id"),
        "source_url": report.get("source_url"),
        "profile_id": report.get("profile_id"),
        "ontology_language": report.get("ontology_language"),
        "embedding_model": report.get("embedding_model") or report.get("embedding_model_name"),
        "prompt_type": report.get("prompt_type"),
        "counts": {
            "raw": len(triplets),
            "parsed": len(triplets),
            "final": len(triplets),
            "filtered": 0,
            "pipeline_filtered": 0,
            "ontology_filtered": 0,
            "kg_inserted": 0,
            "kg_existing": 0,
            "merges": 0,
        },
        "raw_text": report.get("text"),
        "parsed_triplets": triplets,
        "merge_rows": [],
        "filtered_triplets": [],
        "pipeline_filtered_triplets": [],
        "ontology_filtered_triplets": [],
        "final_triplets": triplets,
        "chunk_results": [],
        "relation_rows": [],
        "entity_info": {},
        "source_payload": report,
    }


def _build_view(report: dict[str, Any]) -> dict[str, Any]:
    schema = report.get("schema_version")
    if schema == "wiki-batch-report-v1":
        return _batch_to_view(report)
    if schema == "paper-report-v1":
        return _paper_report_to_view(report)
    if "artifacts" in report and "run" in report:
        return _export_to_view(report)
    return _generic_to_view(report)


def _chunk_options(report: dict[str, Any]) -> list[tuple[str, dict[str, Any] | None]]:
    if report.get("schema_version") != "wiki-batch-report-v1":
        return []
    options: list[tuple[str, dict[str, Any] | None]] = [("Batch toplamı", None)]
    for idx, chunk_report in enumerate(_as_list(report.get("chunk_reports")), start=1):
        run_info = chunk_report.get("run_info") or {}
        extra = run_info.get("extra_config") or {}
        chunk_index = extra.get("chunk_index", idx)
        run_id = run_info.get("run_id", "")
        options.append((f"Chunk {chunk_index} · {_truncate(run_id, 10)}", chunk_report))
    return options


def _render_header(view: dict[str, Any]) -> None:
    metrics = st.columns(4)
    metrics[0].metric("Status", view.get("status") or "—")
    metrics[1].metric("Model", _truncate(view.get("model"), 24))
    metrics[2].metric("Created At", _format_datetime(view.get("created_at")))
    metrics[3].metric("Sample ID", _truncate(view.get("sample_id"), 18))

    profile_bits = []
    if view.get("profile_id"):
        profile_bits.append(f"Profile: `{view['profile_id']}`")
    if view.get("ontology_language"):
        profile_bits.append(f"Lang: `{view['ontology_language']}`")
    if view.get("embedding_model"):
        profile_bits.append(f"Embedding: `{view['embedding_model']}`")
    if view.get("prompt_type"):
        profile_bits.append(f"Prompt: `{view['prompt_type']}`")
    if profile_bits:
        st.caption("🗂 " + "  |  ".join(profile_bits))

    if view.get("run_id"):
        label = "Batch ID" if view.get("kind") == "batch" else "Run ID"
        st.caption(f"🔑 {label}: `{view['run_id']}`")
    if view.get("source_url"):
        st.caption(f"🔗 Source: {view['source_url']}")


def _render_summary_metrics(view: dict[str, Any]) -> None:
    counts = view.get("counts") or {}
    cols = st.columns(5)
    cols[0].metric("🔴 Raw", counts.get("raw", 0))
    cols[1].metric("🟡 Parsed", counts.get("parsed", 0))
    cols[2].metric("🟢 Final", counts.get("final", 0))
    cols[3].metric("⚠️ Filtered", counts.get("filtered", 0))
    cols[4].metric("🚫 Ontology", counts.get("ontology_filtered", 0))


def _render_tabs(view: dict[str, Any], key_prefix: str) -> None:
    tabs = st.tabs([
        "🔴 Raw-0: LLM Output",
        "🟡 Raw-1: Parsed Triplets",
        "🔀 Merge Log",
        "🚫 Filtered Out",
        "🟢 Final Triplets",
        "📊 Batch / Summary",
        "🧾 JSON",
    ])

    with tabs[0]:
        raw_text = view.get("raw_text")
        if raw_text:
            st.code(raw_text, language="json")
        else:
            st.info("Bu JSON formatında ham LLM çıktısı yok.")

    with tabs[1]:
        parsed = view.get("parsed_triplets") or []
        df = _triplet_df(parsed, TRIPLET_COLUMNS)
        if df is None:
            st.info("Bu JSON formatında parse edilmiş triplet listesi yok.")
        else:
            st.caption(f"**{len(parsed)} triplet**")
            st.dataframe(df, width="stretch", hide_index=True)
            _render_sentence_details(parsed, key_prefix=f"{key_prefix}_parsed")
            _render_download(
                "⬇️ parsed_triplets.json indir",
                parsed,
                "parsed_triplets.json",
                key=f"{key_prefix}_download_parsed",
            )

    with tabs[2]:
        merges = view.get("merge_rows") or []
        if not merges:
            st.info("Merge kaydı yok.")
        else:
            st.caption(f"**{len(merges)} merge**")
            df = pd.DataFrame(merges)
            preferred = ["from", "to", "entity_type", "method", "chunk_index", "run_id"]
            existing = [column for column in preferred if column in df.columns]
            st.dataframe(df[existing] if existing else df, width="stretch", hide_index=True)
            _render_download(
                "⬇️ merge_log.json indir",
                merges,
                "merge_log.json",
                key=f"{key_prefix}_download_merges",
            )

    with tabs[3]:
        filtered = view.get("filtered_triplets") or []
        counts = view.get("counts") or {}
        if not filtered and not counts.get("filtered"):
            st.success("Bu raporda elenen triplet yok.")
        else:
            metric_cols = st.columns(3)
            metric_cols[0].metric("🚫 Toplam", counts.get("filtered", len(filtered)))
            metric_cols[1].metric("⚠️ Pipeline", counts.get("pipeline_filtered", 0))
            metric_cols[2].metric("🔴 Ontology", counts.get("ontology_filtered", 0))
            df = _triplet_df(filtered, FILTER_COLUMNS)
            if df is not None:
                st.dataframe(df, width="stretch", hide_index=True)
                _render_sentence_details(filtered, key_prefix=f"{key_prefix}_filtered")
                _render_download(
                    "⬇️ filtered_out.json indir",
                    filtered,
                    "filtered_out.json",
                    key=f"{key_prefix}_download_filtered",
                )
            else:
                st.info("Özet sayım var ama triplet detayları JSON içinde yok.")

    with tabs[4]:
        final = view.get("final_triplets") or []
        counts = view.get("counts") or {}
        fc = st.columns(3)
        fc[0].metric("✅ Final", counts.get("final", len(final)))
        fc[1].metric("⚠️ Filtered", counts.get("filtered", 0))
        fc[2].metric("🚫 Ontology Filtered", counts.get("ontology_filtered", 0))

        df = _triplet_df(final, TRIPLET_COLUMNS)
        if df is None:
            st.info("Final triplet bulunamadı.")
        else:
            st.dataframe(df, width="stretch", hide_index=True)
            _render_sentence_details(final, key_prefix=f"{key_prefix}_final")
            _render_download(
                "⬇️ final_triplets.json indir",
                final,
                "final_triplets.json",
                key=f"{key_prefix}_download_final",
            )

    with tabs[5]:
        chunk_rows = view.get("chunk_results") or []
        relation_rows = view.get("relation_rows") or []
        entity_info = view.get("entity_info") or {}

        if chunk_rows:
            st.markdown("**Chunk sonuçları**")
            st.dataframe(pd.DataFrame(chunk_rows), width="stretch", hide_index=True)

        if relation_rows:
            st.markdown("**En sık relationlar**")
            st.dataframe(pd.DataFrame(relation_rows), width="stretch", hide_index=True)

        entity_count = entity_info.get("final_unique_entity_count")
        entities = entity_info.get("final_unique_entities")
        if entity_count is not None:
            st.metric("Final unique entity", entity_count)
        if entities:
            with st.expander("Entity listesini göster", expanded=False):
                st.write(", ".join(str(entity) for entity in entities))

        if not chunk_rows and not relation_rows and entity_count is None:
            st.info("Bu JSON için ek özet bölümü yok.")

    with tabs[6]:
        _render_download(
            "⬇️ görüntülenen_json.json indir",
            view.get("source_payload"),
            "wikontic_report_view.json",
            key=f"{key_prefix}_download_source",
        )
        with st.expander("JSON göster", expanded=True):
            st.json(view.get("source_payload"))


def _render_filter_panel(view: dict[str, Any], key_prefix: str) -> dict[str, Any]:
    final = view.get("final_triplets") or []
    filtered = view.get("filtered_triplets") or []
    all_triplets = final + filtered

    if not all_triplets:
        return view

    with st.sidebar:
        st.markdown("### 🔎 JSON Filtreleri")
        relation_options = sorted(
            {str(t.get("relation")) for t in all_triplets if t.get("relation")}
        )
        type_options = sorted(
            {
                str(value)
                for triplet in all_triplets
                for value in (triplet.get("subject_type"), triplet.get("object_type"))
                if value
            }
        )
        selected_relation = st.selectbox(
            "Relation",
            ["(tümü)"] + relation_options,
            key=f"{key_prefix}_relation_filter",
        )
        selected_type = st.selectbox(
            "Entity type",
            ["(tümü)"] + type_options,
            key=f"{key_prefix}_type_filter",
        )
        query = st.text_input(
            "Subject/Object ara",
            key=f"{key_prefix}_query_filter",
            placeholder="örn. Messi",
        ).strip().casefold()

    def keep(triplet: dict[str, Any]) -> bool:
        if selected_relation != "(tümü)" and str(triplet.get("relation")) != selected_relation:
            return False
        if selected_type != "(tümü)" and selected_type not in {
            str(triplet.get("subject_type")),
            str(triplet.get("object_type")),
        }:
            return False
        if query:
            haystack = " ".join(
                str(triplet.get(key, ""))
                for key in ("subject", "object", "relation")
            ).casefold()
            if query not in haystack:
                return False
        return True

    if selected_relation == "(tümü)" and selected_type == "(tümü)" and not query:
        return view

    filtered_final = [triplet for triplet in final if keep(triplet)]
    filtered_out = [triplet for triplet in filtered if keep(triplet)]
    next_view = dict(view)
    next_view["final_triplets"] = filtered_final
    next_view["filtered_triplets"] = filtered_out
    next_view["counts"] = {
        **(view.get("counts") or {}),
        "final": len(filtered_final),
        "filtered": len(filtered_out),
        "ontology_filtered": sum(
            1 for triplet in filtered_out
            if triplet in (view.get("ontology_filtered_triplets") or [])
            or triplet.get("filter_stage") == "ontology"
        ),
    }
    st.sidebar.caption(
        f"Filtre sonucu: {len(filtered_final)} final, {len(filtered_out)} elenen triplet."
    )
    return next_view


st.title("📥 JSON Report Viewer")
st.caption(
    "Wikontic run veya Wikipedia batch JSON dosyasını yükleyin; database bağlantısı gerekmez."
)

uploaded = st.file_uploader(
    "Wikontic JSON dosyası",
    type=["json"],
    help="Örn. wiki-batch-report-v1, paper-report-v1 veya Export Run içindeki run.json.",
)

if uploaded is None:
    st.info("Bir JSON dosyası seçince rapor burada görselleştirilecek.")
    st.stop()

try:
    payload = json.loads(uploaded.getvalue().decode("utf-8"))
except UnicodeDecodeError:
    st.error("Dosya UTF-8 JSON olarak okunamadı.")
    st.stop()
except json.JSONDecodeError as exc:
    st.error(f"Geçersiz JSON: {exc}")
    st.stop()

if not isinstance(payload, dict):
    st.error("Beklenen format JSON object olmalı.")
    st.stop()

uploaded_key = hashlib.sha1(uploaded.getvalue()).hexdigest()[:12]
chunk_options = _chunk_options(payload)
selected_chunk_payload: dict[str, Any] | None = None
selected_label = "JSON"
if chunk_options:
    labels = [label for label, _ in chunk_options]
    selected_label = st.selectbox(
        "Görünüm",
        labels,
        help="Batch toplamını veya tek bir chunk'ın paper report detayını seçin.",
    )
    selected_chunk_payload = chunk_options[labels.index(selected_label)][1]

if selected_chunk_payload is None:
    view = _build_view(payload)
else:
    view = _paper_report_to_view(selected_chunk_payload, label=selected_label)

view = _render_filter_panel(view, key_prefix=f"json_{uploaded_key}_{selected_label}")

st.subheader(view.get("label") or selected_label)
_render_header(view)
st.divider()
_render_summary_metrics(view)
st.divider()
_render_tabs(view, key_prefix=f"json_{uploaded_key}_{selected_label}_tabs")
