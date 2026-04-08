import json
import os
from pathlib import Path
import streamlit as st
from dotenv import load_dotenv, find_dotenv
import pandas as pd
from datetime import datetime, timezone

from src.wikontic.utils.run_reader import (
    get_run,
    get_artifact,
    get_all_artifacts,
    list_recent_runs,
    get_distinct_models,
    get_child_runs,
    delete_run,
)
from src.wikontic.utils.run_exporter import export_run
from src.wikontic.utils.run_compare import compare_runs, compare_telemetry

st.set_page_config(
    page_title="Run Viewer — Wikontic",
    page_icon="media/wikotic-wo-text.png",
    layout="wide",
)

_ = load_dotenv(find_dotenv())

# ── Sidebar: LLM Request Log download ────────────────────────────────────────
_LLM_LOG_PATH = Path(os.environ.get("LLM_LOG_PATH", "logs/llm_requests.jsonl"))
with st.sidebar:
    st.markdown("### 🔬 LLM Request Log")
    if _LLM_LOG_PATH.exists():
        with open(_LLM_LOG_PATH, "rb") as _lf:
            st.download_button(
                label="⬇️ llm_requests.jsonl indir",
                data=_lf.read(),
                file_name="llm_requests.jsonl",
                mime="application/jsonlines",
                key="download_llm_log",
            )
        _log_lines = sum(1 for _ in open(_LLM_LOG_PATH, encoding="utf-8"))
        _log_size  = _LLM_LOG_PATH.stat().st_size
        st.caption(f"{_log_lines} istek · {_log_size / 1024:.1f} KB")
    else:
        st.info("Henüz log yok.\nBir extraction çalıştır.")

# ── Session state ─────────────────────────────────────────────────────────────
for _k in ("rv_selected_run_id", "rv_delete_confirm_input"):
    if _k not in st.session_state:
        st.session_state[_k] = None
for _k in ("rv_replay_running", "rv_compare_mode", "rv_show_delete"):
    if _k not in st.session_state:
        st.session_state[_k] = False

_TIMED_STAGES = ["llm_extract", "parse", "ontology_alignment", "db_write"]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _status_badge(status: str) -> str:
    return {"DONE": "🟢", "FAILED": "🔴", "STARTED": "🟡"}.get(status, "⚪") + f" {status}"


def _fmt_ms(ms) -> str:
    if ms is None:
        return "N/A"
    return f"{ms / 1000:.2f} s" if ms >= 1000 else f"{ms:.0f} ms"


def _triplet_df(triplets: list, extra_cols: list | None = None) -> pd.DataFrame | None:
    if not triplets:
        return None
    base = ["subject", "relation", "object"]
    cols = base + (extra_cols or [])
    df   = pd.DataFrame(triplets)
    return df[[c for c in cols if c in df.columns]]


def _show_sentence_detail(triplets: list, key_prefix: str):
    has = any(t.get("sentence_full") for t in triplets)
    if not has:
        return
    if st.checkbox("📖 Cümle detaylarını göster", key=f"{key_prefix}_sent"):
        for t in triplets:
            if t.get("sentence_full"):
                st.caption(
                    f"[{t.get('sentence_id','?')}] "
                    f"**{t.get('subject','')}** — {t.get('relation','')} — **{t.get('object','')}**"
                )
                st.info(t["sentence_full"])


def _render_performance_card(run_meta: dict):
    stats = (run_meta or {}).get("stats") or {}
    with st.expander("⏱️ Performance", expanded=False):
        if not stats:
            st.info("Bu run için telemetri kaydı yok.")
            return
        st.metric("⏱️ Toplam Süre", _fmt_ms(stats.get("total_time_ms")))
        rows = []
        total_known = sum(
            stats.get(f"{s}_time_ms", 0) or 0
            for s in _TIMED_STAGES if stats.get(f"{s}_time_ms") is not None
        )
        for stage in _TIMED_STAGES:
            ms = stats.get(f"{stage}_time_ms")
            if ms is None:
                continue
            pct = round(ms / total_known * 100, 1) if total_known else 0
            rows.append({"Stage": stage, "Süre": _fmt_ms(ms), "%": pct})
        if rows:
            st.markdown("**Stage breakdown**")
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        total_tokens = stats.get("total_tokens")
        if total_tokens is not None:
            tc = st.columns(3)
            tc[0].metric("Prompt tokens",     str(stats.get("prompt_tokens"))     or "N/A")
            tc[1].metric("Completion tokens", str(stats.get("completion_tokens")) or "N/A")
            tc[2].metric("Total tokens",      str(total_tokens))
            cost = stats.get("cost_estimate")
            st.caption(f"Cost estimate: ${cost:.6f}" if cost else "Cost: 0 / N/A")
        else:
            st.caption("Token bilgisi: N/A")
        if stats.get("failed_stage"):
            st.warning(f"Hata olan stage: `{stats['failed_stage']}`")


def _render_stage_tabs(run_id: str):
    tabs = st.tabs([
        "📋 Input & Config",
        "🔴 Raw-0: LLM Output",
        "🟡 Parsed Triplets",
        "🔀 Merge Log",
        "🚫 Filtered Out",
        "🟢 Final Triplets",
    ])

    # ── Tab 0: Input & Config ─────────────────────────────────────────────────
    with tabs[0]:
        run_meta = get_run(run_id)
        if run_meta is None:
            st.warning("Run metadata bulunamadı.")
        else:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Status", _status_badge(run_meta.get("status", "—")))
            c2.metric("Model",  run_meta.get("model", "—"))
            created = run_meta.get("created_at", "")
            if hasattr(created, "strftime"):
                created = created.strftime("%Y-%m-%d %H:%M:%S")
            c3.metric("Created At", str(created))
            c4.metric("Sample ID", str(run_meta.get("sample_id", "—"))[:16] + "…")
            input_text = run_meta.get("input_text", "") or ""
            st.markdown("**Input Text**")
            with st.expander("Göster / Gizle", expanded=bool(input_text)):
                st.text_area("", value=input_text, height=150,
                             disabled=True, key=f"inp_{run_id}")
            extra = run_meta.get("extra_config") or {}
            if extra:
                st.markdown("**Extra Config**"); st.json(extra)
            if run_meta.get("status") == "FAILED" and run_meta.get("error"):
                st.error(f"Hata: {run_meta['error']}")

    # ── Tab 1: Raw LLM Output ─────────────────────────────────────────────────
    with tabs[1]:
        art = get_artifact(run_id, "raw_llm_output")
        if art is None:
            st.warning("Stage kaydı bulunamadı: raw_llm_output")
        else:
            with st.expander("Ham LLM Çıktısı", expanded=True):
                st.code(art.get("text", ""), language="json")

    # ── Tab 2: Parsed Triplets ────────────────────────────────────────────────
    with tabs[2]:
        art = get_artifact(run_id, "parsed_triplets")
        if art is None:
            st.warning("Stage kaydı bulunamadı: parsed_triplets")
        else:
            triplets = art.get("triplets", [])
            st.caption(f"**{art.get('count', len(triplets))} triplet** parse edildi")
            df = _triplet_df(triplets,
                             extra_cols=["sentence_id", "sentence_preview"])
            if df is not None:
                st.dataframe(df, use_container_width=True, hide_index=True)
                _show_sentence_detail(triplets, key_prefix=f"rv_parsed_{run_id}")
            else:
                st.info("Triplet bulunamadı.")

    # ── Tab 3: Merge Log ──────────────────────────────────────────────────────
    with tabs[3]:
        art = get_artifact(run_id, "merge_map_entities")
        if art is None:
            st.warning("Stage kaydı bulunamadı: merge_map_entities")
        else:
            merges = art.get("merges", [])
            if not merges:
                st.info("Bu run'da hiçbir entity merge edilmedi.")
            else:
                st.caption(f"**{len(merges)} entity** merge edildi")
                df = pd.DataFrame(merges)
                existing = [c for c in ["from", "to", "entity_type", "method"] if c in df.columns]
                st.dataframe(df[existing], use_container_width=True, hide_index=True)

    # ── Tab 4: Filtered Out ───────────────────────────────────────────────────
    with tabs[4]:
        art = get_artifact(run_id, "filtered_out")
        if art is None:
            st.warning("Stage kaydı bulunamadı: filtered_out")
        else:
            triplets = art.get("triplets", [])
            total    = art.get("count", len(triplets))
            if total == 0:
                st.success("Bu run'da hiçbir triplet elenmedi.")
            else:
                fc = st.columns(3)
                fc[0].metric("🚫 Toplam",   total)
                fc[1].metric("⚠️ Pipeline", art.get("pipeline_exception_count", 0))
                fc[2].metric("🔴 Ontology", art.get("ontology_filtered_count", 0))
                cols = ["subject", "relation", "object", "reason_code",
                        "filter_stage", "sentence_id", "sentence_preview", "exception_text"]
                df   = pd.DataFrame(triplets)
                st.dataframe(df[[c for c in cols if c in df.columns]],
                             use_container_width=True, hide_index=True)
                _show_sentence_detail(triplets, key_prefix=f"rv_filtered_{run_id}")

    # ── Tab 5: Final Triplets ─────────────────────────────────────────────────
    with tabs[5]:
        art = get_artifact(run_id, "final_triplets")
        if art is None:
            st.warning("Stage kaydı bulunamadı: final_triplets")
        else:
            triplets = art.get("triplets", [])
            count    = art.get("count", len(triplets))
            fc       = st.columns(3)
            fc[0].metric("✅ Final", count)
            if art.get("filtered_count") is not None:
                fc[1].metric("⚠️ Filtered", art["filtered_count"])
            if art.get("ontology_filtered_count") is not None:
                fc[2].metric("🚫 Ontology", art["ontology_filtered_count"])
            df = _triplet_df(triplets,
                             extra_cols=["subject_type", "object_type",
                                         "sentence_id", "sentence_preview"])
            if df is not None:
                st.dataframe(df, use_container_width=True, hide_index=True)
                _show_sentence_detail(triplets, key_prefix=f"rv_final_{run_id}")
            else:
                st.info("Final triplet bulunamadı.")


def _render_lineage(run_id: str, run_meta: dict):
    parent_id = run_meta.get("parent_run_id")
    children  = get_child_runs(run_id)
    if not parent_id and not children:
        return
    st.markdown("**🔗 Run Lineage**")
    lc = st.columns(2)
    with lc[0]:
        if parent_id:
            st.caption("Parent Run")
            pm = get_run(parent_id)
            if pm and hasattr(pm.get("created_at"), "strftime"):
                lbl = f"{pm['created_at'].strftime('%Y-%m-%d %H:%M:%S')} | {pm.get('model','')}"
            else:
                lbl = parent_id
            if st.button(f"⬆️ {lbl}", key=f"goto_parent_{run_id}"):
                st.session_state["rv_selected_run_id"] = parent_id
                st.rerun()
    with lc[1]:
        if children:
            st.caption(f"Child Runs ({len(children)} replay)")
            for child in children[:5]:
                if st.button(
                    f"⬇️ {_status_badge(child['status'])}  {child['created_at']}  {child['model']}",
                    key=f"goto_child_{child['run_id']}",
                ):
                    st.session_state["rv_selected_run_id"] = child["run_id"]
                    st.rerun()
            if len(children) > 5:
                st.caption(f"… ve {len(children) - 5} daha")


def _render_telemetry_compare(tele: dict):
    st.markdown("#### ⏱️ Telemetri Karşılaştırması")
    total_a = tele.get("total_time_ms_a")
    total_b = tele.get("total_time_ms_b")
    delta   = tele.get("total_delta_ms")
    tc = st.columns(3)
    tc[0].metric("Toplam A", _fmt_ms(total_a))
    tc[1].metric("Toplam B", _fmt_ms(total_b))
    tc[2].metric("Δ (B-A)",  _fmt_ms(abs(delta)) if delta is not None else "N/A")
    valid = [r for r in tele.get("stages", [])
             if r["ms_a"] is not None or r["ms_b"] is not None]
    if valid:
        df = pd.DataFrame(valid).rename(columns={
            "stage": "Stage", "ms_a": "A (ms)", "ms_b": "B (ms)", "delta_ms": "Δ ms"
        })
        for col in ["A (ms)", "B (ms)", "Δ ms"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        st.dataframe(df, use_container_width=True, hide_index=True)
    else:
        st.info("Stage telemetrisi bulunamadı.")


def _render_compare_panel(run_id_a: str, runs: list):
    st.subheader("⚖️ Run Karşılaştırma (A/B)")
    other    = [r for r in runs if r["run_id"] != run_id_a]
    if not other:
        st.warning("Karşılaştırmak için başka run bulunamadı.")
        return
    b_labels = [r["label"]  for r in other]
    b_ids    = [r["run_id"] for r in other]

    ca, cb = st.columns(2)
    with ca:
        m  = get_run(run_id_a) or {}
        ca_str = m.get("created_at", "")
        if hasattr(ca_str, "strftime"):
            ca_str = ca_str.strftime("%Y-%m-%d %H:%M:%S")
        st.info(f"**Run A (seçili)**\n\n`{run_id_a[:12]}…`\n{ca_str}\n{m.get('model','')}")
    with cb:
        sel_b    = st.selectbox("Run B seç:", b_labels, key="rv_compare_b")
        run_id_b = b_ids[b_labels.index(sel_b)]

    if run_id_a == run_id_b:
        st.warning("A ve B aynı. Farklı bir run seçin.")
        return

    with st.spinner("Karşılaştırılıyor…"):
        try:
            report = compare_runs(run_id_a, run_id_b)
            tele   = compare_telemetry(run_id_a, run_id_b)
        except Exception as e:
            st.error(f"Karşılaştırma hatası: {e}")
            return

    s  = report["summary"]
    sc = st.columns(4)
    sc[0].metric("Raw A→B",      f"{s['raw_count_a']} → {s['raw_count_b']}",
                 delta=s["raw_count_b"]           - s["raw_count_a"])
    sc[1].metric("Final A→B",    f"{s['final_count_a']} → {s['final_count_b']}",
                 delta=s["final_count_b"]         - s["final_count_a"])
    sc[2].metric("Filtered A→B", f"{s['filtered_count_a']} → {s['filtered_count_b']}",
                 delta=s["filtered_count_b"]      - s["filtered_count_a"])
    sc[3].metric("Merges A→B",
                 f"{s['merges_entity_count_a']} → {s['merges_entity_count_b']}",
                 delta=s["merges_entity_count_b"] - s["merges_entity_count_a"])

    ct1, ct2, ct3, ct4 = st.tabs([
        "🔀 Final Diff", "🔁 Merge Diff", "🚫 Filter Reason Diff", "⏱️ Telemetri"
    ])

    with ct1:
        fd = report["final_diff"]
        dc = st.columns(2)
        dc[0].metric("➕ Eklenen",  fd["added_count"])
        dc[1].metric("➖ Çıkarılan", fd["removed_count"])
        with st.expander(f"➕ Eklenen ({fd['added_count']})", expanded=fd["added_count"] > 0):
            if fd["added_edges"]:
                df = pd.DataFrame(fd["added_edges"])
                st.dataframe(df[[c for c in ["subject","relation","object"] if c in df.columns]].head(200),
                             use_container_width=True, hide_index=True)
                if len(fd["added_edges"]) > 200:
                    st.caption(f"İlk 200, toplam {len(fd['added_edges'])}")
            else:
                st.info("Yok.")
        with st.expander(f"➖ Çıkarılan ({fd['removed_count']})", expanded=fd["removed_count"] > 0):
            if fd["removed_edges"]:
                df = pd.DataFrame(fd["removed_edges"])
                st.dataframe(df[[c for c in ["subject","relation","object"] if c in df.columns]].head(200),
                             use_container_width=True, hide_index=True)
            else:
                st.info("Yok.")

    with ct2:
        md = report["merge_diff"]["entity"]
        mc = st.columns(3)
        mc[0].metric("Merge A", md["count_a"])
        mc[1].metric("Merge B", md["count_b"])
        mc[2].metric("Δ", md["count_b"] - md["count_a"])
        with st.expander(f"B'de eklenen ({len(md['added'])})", expanded=len(md["added"]) > 0):
            st.dataframe(pd.DataFrame(md["added"]),
                         use_container_width=True, hide_index=True) if md["added"] else st.info("Yok.")
        with st.expander(f"A'da olup B'de olmayan ({len(md['removed'])})",
                         expanded=len(md["removed"]) > 0):
            st.dataframe(pd.DataFrame(md["removed"]),
                         use_container_width=True, hide_index=True) if md["removed"] else st.info("Yok.")

    with ct3:
        reasons = report["filter_reason_diff"]["reasons"]
        if not reasons:
            st.info("Her iki run'da da filter yok.")
        else:
            st.dataframe(
                pd.DataFrame(reasons).rename(columns={
                    "reason": "Reason Code", "count_a": "Count A",
                    "count_b": "Count B",    "delta":   "Δ (B-A)",
                }),
                use_container_width=True, hide_index=True,
            )

    with ct4:
        _render_telemetry_compare(tele)

    st.divider()
    try:
        full = {**report, "telemetry": tele}
        ts   = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M")
        st.download_button(
            label="⬇️ Export Compare Report (JSON)",
            data=json.dumps(full, ensure_ascii=False, indent=2, default=str),
            file_name=f"compare_{run_id_a[:8]}_vs_{run_id_b[:8]}_{ts}.json",
            mime="application/json",
            key="export_compare",
        )
    except Exception as e:
        st.error(f"Compare export hatası: {e}")


def _render_delete_zone(run_id: str):
    with st.expander("🗑️ Danger Zone: Run Sil", expanded=False):
        st.error("Bu işlem geri alınamaz. Run ve tüm artifact'ları kalıcı olarak silinir.",
                 icon="⚠️")
        confirm = st.text_input(
            'Onaylamak için "DELETE" yazın:',
            key="rv_delete_confirm_input",
            placeholder="DELETE",
        )
        if st.button("🗑️ Kalıcı Olarak Sil", type="primary", key="rv_delete_btn"):
            if confirm != "DELETE":
                st.warning('Silmek için tam olarak "DELETE" yazmanız gerekiyor.')
            else:
                with st.spinner("Siliniyor…"):
                    result = delete_run(run_id)
                if result.get("ok"):
                    st.success(
                        f"Silindi: {result['runs_deleted']} run, "
                        f"{result['artifacts_deleted']} artifact."
                    )
                    st.session_state["rv_selected_run_id"]    = None
                    st.session_state["rv_delete_confirm_input"] = None
                    st.rerun()
                else:
                    st.error(f"Silme başarısız: {result.get('error', 'bilinmeyen hata')}")


st.title("🗂️ Run Viewer")
st.caption("Geçmiş extraction run'larını inceleyin, filtreleyin, export edin ve replay edin.")

left_col, right_col = st.columns([3, 7])

# ── Filters (left column) ─────────────────────────────────────────────────────
with left_col:
    st.subheader("Filtreler")
    try:
        all_models = ["(tümü)"] + get_distinct_models()
    except Exception:
        all_models = ["(tümü)"]

    filter_model_label = st.selectbox("Model", all_models, key="rv_filter_model")
    filter_model = None if filter_model_label == "(tümü)" else filter_model_label

    filter_status_label = st.selectbox(
        "Status", ["(tümü)", "DONE", "FAILED", "STARTED"], key="rv_filter_status"
    )
    filter_status = None if filter_status_label == "(tümü)" else filter_status_label

    filter_sample_id = st.text_input("Sample ID (tam)", key="rv_filter_sample") or None

    st.divider()

    try:
        runs = list_recent_runs(limit=50, sample_id=filter_sample_id,
                                status=filter_status, model=filter_model)
    except Exception as e:
        runs = []
        st.error(f"Run listesi alınamadı: {e}")

    if not runs:
        st.info("Gösterilecek run bulunamadı.")
    else:
        if st.session_state["rv_selected_run_id"] is None:
            st.session_state["rv_selected_run_id"] = runs[0]["run_id"]
        st.caption(f"**{len(runs)} run** listeleniyor")
        for r in runs:
            is_sel = r["run_id"] == st.session_state["rv_selected_run_id"]
            if st.button(
                f"{_status_badge(r['status'])}  {r['created_at']}\n{r['model']}",
                key=f"run_btn_{r['run_id']}",
                use_container_width=True,
                type="primary" if is_sel else "secondary",
                help=r.get("input_preview", "")[:80],
            ):
                st.session_state["rv_selected_run_id"] = r["run_id"]
                st.rerun()

# ── Detail view (right column) ────────────────────────────────────────────────
with right_col:
    selected_run_id = st.session_state.get("rv_selected_run_id")

    if not selected_run_id:
        st.info("Sol panelden bir run seçin.")
    else:
        run_meta = get_run(selected_run_id)

        compare_mode = st.toggle("⚖️ Compare mode",
                                  value=st.session_state["rv_compare_mode"],
                                  key="rv_compare_toggle")
        st.session_state["rv_compare_mode"] = compare_mode

        if compare_mode:
            if runs:
                _render_compare_panel(selected_run_id, runs)
            else:
                st.warning("Karşılaştırma için run listesi yüklenemedi.")
        else:
            st.subheader("Run Detayı")
            st.code(selected_run_id, language=None)

            if run_meta:
                _render_performance_card(run_meta)

            btn_col1, btn_col2, btn_col3 = st.columns(3)

            with btn_col1:
                if st.button("↗️ KG Extraction'da aç", key="open_in_extraction"):
                    st.session_state["last_run_id"]         = selected_run_id
                    st.session_state["selected_run_id"]     = selected_run_id
                    st.session_state["_rv_just_navigated"]  = True
                    st.switch_page("pages/1_KG_Extraction.py")

            with btn_col2:
                try:
                    zip_bytes, filename, mimetype = export_run(selected_run_id)
                    st.download_button(
                        label="⬇️ Export Run (ZIP)",
                        data=zip_bytes, file_name=filename, mime=mimetype,
                        key=f"export_{selected_run_id}",
                    )
                except Exception as e:
                    st.error(f"Export hazırlanamadı: {e}")

            with btn_col3:
                available_models = [
                    "google/gemini-2.5-flash-lite", "gpt-4o-mini",
                    "gpt-4.1-mini", "gpt-4.1",
                ]
                original_model = (run_meta or {}).get("model", available_models[0])
                with st.expander("🔄 Replay", expanded=False):
                    replay_model = st.selectbox(
                        "Model (override)", available_models,
                        index=available_models.index(original_model)
                        if original_model in available_models else 0,
                        key="rv_replay_model",
                    )
                    if not bool((run_meta or {}).get("input_text", "")):
                        st.warning("Bu run'da input_text yok.")
                    else:
                        if st.button("▶️ Replay başlat", key="replay_btn",
                                     disabled=st.session_state["rv_replay_running"]):
                            st.session_state["rv_replay_running"] = True
                            with st.spinner("Replay çalışıyor…"):
                                try:
                                    from src.wikontic.utils.replay_runner import replay_run
                                    new_run_id = replay_run(selected_run_id,
                                                            overrides={"model": replay_model})
                                    st.session_state["rv_selected_run_id"] = new_run_id
                                    st.session_state["rv_replay_running"]  = False
                                    st.success(f"✅ Replay → {new_run_id[:8]}…")
                                    st.rerun()
                                except Exception as e:
                                    st.session_state["rv_replay_running"] = False
                                    st.error(f"Replay başarısız: {e}")

            if run_meta:
                _render_lineage(selected_run_id, run_meta)

            st.divider()

            try:
                _render_stage_tabs(selected_run_id)
            except Exception as e:
                st.error(f"Stage tabları yüklenirken hata: {e}")

            st.divider()
            _render_delete_zone(selected_run_id)