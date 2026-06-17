from __future__ import annotations

import base64
import html
import json
import tempfile
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st
from pyvis.network import Network

from src.wikontic.profiles import (
    get_available_embedding_profiles,
    get_available_ontology_profiles,
    resolve_runtime_profile,
)


st.set_page_config(
    page_title="Ontology Explorer - Wikontic",
    page_icon="media/wikotic-wo-text.png",
    layout="wide",
)


ROOT_DIR = Path(__file__).resolve().parents[1]
MAPPINGS_DIR = ROOT_DIR / "src" / "wikontic" / "utils" / "ontology_mappings"
LOGO_PATH = ROOT_DIR / "media" / "wikontic.png"

LANGUAGE_LABELS = {
    "en": "English",
    "tr": "Türkçe",
}


@st.cache_data(show_spinner=False)
def _read_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data(show_spinner=False)
def _load_bundle(language: str) -> dict[str, Any]:
    lang_dir = MAPPINGS_DIR if language == "en" else MAPPINGS_DIR / language
    if not lang_dir.exists():
        raise FileNotFoundError(f"Ontology mapping directory not found: {lang_dir}")

    entity_labels = _read_json(str(lang_dir / "entity_type2label.json"))
    entity_aliases = _read_json(str(lang_dir / "entity_type2aliases.json"))
    property_labels = _read_json(str(lang_dir / "prop2label.json"))
    property_aliases = _read_json(str(lang_dir / "prop2aliases.json"))

    entity_labels.setdefault("ANY", "ANY")
    entity_aliases.setdefault("ANY", [])

    return {
        "language": language,
        "entity_labels": entity_labels,
        "entity_aliases": entity_aliases,
        "property_labels": property_labels,
        "property_aliases": property_aliases,
        "entity_hierarchy": _read_json(str(MAPPINGS_DIR / "entity_type2hierarchy.json")),
        "prop_constraints": _read_json(str(MAPPINGS_DIR / "prop2constraints.json")),
        "prop_data_type": _read_json(str(MAPPINGS_DIR / "prop2data_type.json")),
        "subj_constraint2prop": _read_json(str(MAPPINGS_DIR / "subj_constraint2prop.json")),
        "obj_constraint2prop": _read_json(str(MAPPINGS_DIR / "obj_constraint2prop.json")),
        "subject_object_constraints": _read_json(str(MAPPINGS_DIR / "subject_object_constraints.json")),
    }


@st.cache_data(show_spinner=False)
def _bundle_download_json(language: str) -> str:
    bundle = _load_bundle(language)
    payload = {
        "language": language,
        "localized": {
            "entity_type2label": bundle["entity_labels"],
            "entity_type2aliases": bundle["entity_aliases"],
            "prop2label": bundle["property_labels"],
            "prop2aliases": bundle["property_aliases"],
        },
        "shared": {
            "entity_type2hierarchy": bundle["entity_hierarchy"],
            "prop2constraints": bundle["prop_constraints"],
            "prop2data_type": bundle["prop_data_type"],
            "subj_constraint2prop": bundle["subj_constraint2prop"],
            "obj_constraint2prop": bundle["obj_constraint2prop"],
            "subject_object_constraints": bundle["subject_object_constraints"],
        },
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def _style_page() -> None:
    st.markdown(
        """
        <style>
        .ontology-header {
            display: flex;
            align-items: center;
            gap: 0.75rem;
            margin-bottom: 0.25rem;
        }
        .ontology-header img {
            width: 44px;
            height: 44px;
            object-fit: contain;
        }
        .ontology-header h1 {
            margin: 0;
            line-height: 1.1;
        }
        .ontology-note {
            border: 1px solid rgba(49, 51, 63, 0.18);
            border-radius: 8px;
            padding: 0.85rem 1rem;
            background: rgba(248, 249, 251, 0.75);
            color: rgb(49, 51, 63);
            margin: 0.35rem 0 1rem;
        }
        .ontology-note strong {
            color: rgb(20, 30, 45);
        }
        .ontology-legend {
            display: flex;
            flex-wrap: wrap;
            gap: 0.45rem 0.75rem;
            margin: 0.25rem 0 0.9rem;
        }
        .ontology-chip {
            display: inline-flex;
            align-items: center;
            gap: 0.35rem;
            font-size: 0.83rem;
            color: rgb(49, 51, 63);
        }
        .ontology-dot {
            width: 0.72rem;
            height: 0.72rem;
            border-radius: 999px;
            display: inline-block;
            border: 1px solid rgba(0, 0, 0, 0.2);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _render_header() -> None:
    if LOGO_PATH.exists():
        encoded_logo = base64.b64encode(LOGO_PATH.read_bytes()).decode("utf-8")
        st.markdown(
            f"""
            <div class="ontology-header">
                <img src="data:image/png;base64,{encoded_logo}" alt="Wikontic logo">
                <h1>Ontology Explorer</h1>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.title("Ontology Explorer")
    st.caption(
        "Wikidata ontology mappings, allowed subject/object constraints, hierarchy, "
        "aliases, and database materialization for English and Turkish profiles."
    )


def _label_entity(bundle: dict[str, Any], entity_id: str) -> str:
    if entity_id == "<ANY SUBJECT>":
        return "Any subject"
    if entity_id == "<ANY OBJECT>":
        return "Any object"
    return bundle["entity_labels"].get(entity_id, entity_id)


def _label_property(bundle: dict[str, Any], property_id: str) -> str:
    return bundle["property_labels"].get(property_id, property_id)


def _shorten(text: str, length: int = 32) -> str:
    text = str(text)
    return text if len(text) <= length else f"{text[: length - 3]}..."


def _graph_node_label(label: str, item_id: str) -> str:
    return f"{_shorten(label)}\n{item_id}"


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        if value and value not in seen:
            seen.add(value)
            out.append(value)
    return out


def _sorted_entities(bundle: dict[str, Any], entity_ids: list[str] | set[str]) -> list[str]:
    return sorted(
        set(entity_ids),
        key=lambda entity_id: (_label_entity(bundle, entity_id).lower(), entity_id),
    )


def _sorted_properties(bundle: dict[str, Any], property_ids: list[str] | set[str]) -> list[str]:
    return sorted(
        set(property_ids),
        key=lambda property_id: (_label_property(bundle, property_id).lower(), property_id),
    )


def _alias_preview(alias_map: dict[str, list[str]], key: str, limit: int = 4) -> str:
    aliases = alias_map.get(key) or []
    if not aliases:
        return "-"
    preview = ", ".join(str(alias) for alias in aliases[:limit])
    if len(aliases) > limit:
        preview += f" +{len(aliases) - limit}"
    return preview


def _constraint_sources(
    bundle: dict[str, Any],
    entity_id: str,
    direction: str,
    include_inherited: bool,
    include_any: bool,
) -> dict[str, set[str]]:
    source_map = (
        bundle["subj_constraint2prop"]
        if direction == "subject"
        else bundle["obj_constraint2prop"]
    )
    any_key = "<ANY SUBJECT>" if direction == "subject" else "<ANY OBJECT>"
    prop_sources: dict[str, set[str]] = {}

    def add(prop_id: str, source: str) -> None:
        prop_sources.setdefault(prop_id, set()).add(source)

    for prop_id in source_map.get(entity_id, []):
        add(prop_id, "direct")

    if include_inherited:
        for parent_id in bundle["entity_hierarchy"].get(entity_id, []):
            if parent_id == entity_id:
                continue
            parent_label = _label_entity(bundle, parent_id)
            for prop_id in source_map.get(parent_id, []):
                add(prop_id, f"inherited: {parent_label} ({parent_id})")

    if include_any:
        for prop_id in source_map.get(any_key, []):
            add(prop_id, "global ANY")

    return prop_sources


def _summarize_sources(sources: set[str]) -> str:
    parts: list[str] = []
    if "direct" in sources:
        parts.append("direct")
    inherited_count = sum(1 for source in sources if source.startswith("inherited:"))
    if inherited_count:
        parts.append(f"inherited ({inherited_count})")
    if "global ANY" in sources:
        parts.append("global ANY")
    return ", ".join(parts) if parts else "-"


def _property_rows(
    bundle: dict[str, Any],
    property_ids: list[str] | set[str],
    prop_sources: dict[str, set[str]] | None = None,
    limit: int | None = None,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    sorted_ids = _sorted_properties(bundle, property_ids)
    if limit is not None:
        sorted_ids = sorted_ids[:limit]

    for prop_id in sorted_ids:
        constraints = bundle["prop_constraints"].get(prop_id, {})
        subject_types = constraints.get("Subject type constraint", [])
        object_types = constraints.get("Value-type constraint", [])
        rows.append(
            {
                "property_id": prop_id,
                "label": _label_property(bundle, prop_id),
                "data_type": bundle["prop_data_type"].get(prop_id, "-"),
                "subject_type_count": len(subject_types),
                "object_type_count": len(object_types),
                "source": _summarize_sources(prop_sources.get(prop_id, set()))
                if prop_sources
                else "-",
                "aliases": _alias_preview(bundle["property_aliases"], prop_id),
            }
        )
    return pd.DataFrame(rows)


def _entity_rows(
    bundle: dict[str, Any],
    entity_ids: list[str] | set[str],
    limit: int | None = None,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    sorted_ids = _sorted_entities(bundle, entity_ids)
    if limit is not None:
        sorted_ids = sorted_ids[:limit]

    for entity_id in sorted_ids:
        rows.append(
            {
                "entity_type_id": entity_id,
                "label": _label_entity(bundle, entity_id),
                "parent_count": len(bundle["entity_hierarchy"].get(entity_id, [])),
                "valid_as_subject": len(bundle["subj_constraint2prop"].get(entity_id, [])),
                "valid_as_object": len(bundle["obj_constraint2prop"].get(entity_id, [])),
                "aliases": _alias_preview(bundle["entity_aliases"], entity_id),
            }
        )
    return pd.DataFrame(rows)


def _network() -> Network:
    net = Network(
        height="650px",
        width="100%",
        bgcolor="#ffffff",
        font_color="#172033",
        directed=True,
        cdn_resources="in_line",
    )
    net.set_options(
        """
        var options = {
          "layout": {"improvedLayout": true},
          "physics": {
            "enabled": true,
            "solver": "forceAtlas2Based",
            "forceAtlas2Based": {
              "gravitationalConstant": -65,
              "centralGravity": 0.012,
              "springLength": 155,
              "springConstant": 0.08,
              "damping": 0.48
            },
            "stabilization": {"iterations": 220}
          },
          "edges": {
            "arrows": {
              "to": {"enabled": true, "scaleFactor": 0.75},
              "middle": {"enabled": false},
              "from": {"enabled": false}
            },
            "color": {"inherit": false},
            "font": {
              "size": 11,
              "align": "middle",
              "background": "#ffffff",
              "strokeWidth": 3,
              "strokeColor": "#ffffff"
            },
            "smooth": {"enabled": true, "type": "dynamic"}
          },
          "nodes": {
            "font": {"size": 14, "face": "Inter, Arial, sans-serif"},
            "borderWidth": 1.5,
            "shadow": false
          },
          "interaction": {
            "hover": true,
            "navigationButtons": false,
            "keyboard": true
          }
        }
        """
    )
    return net


def _with_png_export_button(graph_html: str) -> str:
    export_controls = """
        <style>
          .wikontic-export-bar {
            display: flex;
            align-items: center;
            gap: 8px;
            padding: 8px 10px 10px;
            border-top: 1px solid rgba(148, 163, 184, 0.35);
            background: #ffffff;
            box-sizing: border-box;
            font-family: Inter, Arial, sans-serif;
          }
          .wikontic-export-button {
            border: 1px solid #cbd5e1;
            border-radius: 6px;
            background: #ffffff;
            color: #172033;
            cursor: pointer;
            font-size: 13px;
            line-height: 1.2;
            padding: 7px 11px;
          }
          .wikontic-export-button:hover {
            background: #f8fafc;
            border-color: #94a3b8;
          }
        </style>
        <div class="wikontic-export-bar">
          <button
            id="wikontic-download-visible-png"
            class="wikontic-export-button"
            type="button"
            aria-label="Download visible graph area as PNG"
          >
            Download visible PNG
          </button>
        </div>
        <script>
          (function () {
            function downloadVisibleGraphPng() {
              var container = document.getElementById("mynetwork");
              if (!container) {
                return;
              }

              var canvas = container.querySelector("canvas");
              if (!canvas) {
                return;
              }

              var exportCanvas = document.createElement("canvas");
              exportCanvas.width = canvas.width;
              exportCanvas.height = canvas.height;

              var context = exportCanvas.getContext("2d");
              context.fillStyle = "#ffffff";
              context.fillRect(0, 0, exportCanvas.width, exportCanvas.height);
              context.drawImage(canvas, 0, 0);

              var link = document.createElement("a");
              var timestamp = new Date().toISOString().replace(/[:.]/g, "-");
              link.download = "ontology-graph-" + timestamp + ".png";
              link.href = exportCanvas.toDataURL("image/png");
              document.body.appendChild(link);
              link.click();
              document.body.removeChild(link);
            }

            var button = document.getElementById("wikontic-download-visible-png");
            if (button) {
              button.addEventListener("click", downloadVisibleGraphPng);
            }
          })();
        </script>
    """
    if "</body>" in graph_html:
        return graph_html.replace("</body>", f"{export_controls}\n    </body>", 1)
    return graph_html + export_controls


def _node_title(label: str, item_id: str, detail: str = "") -> str:
    title = f"<b>{html.escape(label)}</b><br>{html.escape(item_id)}"
    if detail:
        title += f"<br>{html.escape(detail)}"
    return title


def _add_entity_node(
    net: Network,
    bundle: dict[str, Any],
    entity_id: str,
    role: str,
    size: int = 18,
) -> None:
    colors = {
        "selected": {"background": "#dbeafe", "border": "#2563eb"},
        "parent": {"background": "#e2e8f0", "border": "#64748b"},
        "subject": {"background": "#ede9fe", "border": "#7c3aed"},
        "object": {"background": "#dcfce7", "border": "#16a34a"},
    }
    label = _label_entity(bundle, entity_id)
    net.add_node(
        entity_id,
        label=_graph_node_label(label, entity_id),
        title=_node_title(label, entity_id, "entity type"),
        color=colors.get(role, colors["object"]),
        shape="box",
        margin=10,
        size=size,
    )


def _add_property_node(
    net: Network,
    bundle: dict[str, Any],
    property_id: str,
    role: str = "property",
    size: int = 16,
) -> None:
    colors = {
        "subject_property": {"background": "#ccfbf1", "border": "#0f766e"},
        "object_property": {"background": "#ffedd5", "border": "#c2410c"},
        "property": {"background": "#fef3c7", "border": "#b45309"},
    }
    label = _label_property(bundle, property_id)
    data_type = bundle["prop_data_type"].get(property_id, "unknown data type")
    net.add_node(
        property_id,
        label=_graph_node_label(label, property_id),
        title=_node_title(label, property_id, f"property, {data_type}"),
        color=colors.get(role, colors["property"]),
        shape="ellipse",
        size=size,
    )


def _render_network(net: Network, height: int = 650) -> None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as tmp_file:
        path = Path(tmp_file.name)
    try:
        net.save_graph(str(path))
        graph_html = _with_png_export_button(path.read_text(encoding="utf-8"))
        st.iframe(graph_html, height=height + 52, width="stretch")
    finally:
        path.unlink(missing_ok=True)


def _entity_graph(
    bundle: dict[str, Any],
    entity_id: str,
    direction: str,
    max_relations: int,
    max_types_per_relation: int,
    include_inherited: bool,
    include_any: bool,
) -> Network:
    net = _network()
    _add_entity_node(net, bundle, entity_id, "selected", size=28)

    parent_ids = _dedupe(bundle["entity_hierarchy"].get(entity_id, []))
    for parent_id in parent_ids[:10]:
        if parent_id == entity_id:
            continue
        _add_entity_node(net, bundle, parent_id, "parent", size=14)
        net.add_edge(entity_id, parent_id, label="parent type", color="#94a3b8")

    prop_sources = _constraint_sources(
        bundle,
        entity_id,
        direction,
        include_inherited=include_inherited,
        include_any=include_any,
    )
    property_ids = _sorted_properties(bundle, prop_sources.keys())[:max_relations]

    for prop_id in property_ids:
        constraints = bundle["prop_constraints"].get(prop_id, {})
        _add_property_node(
            net,
            bundle,
            prop_id,
            "subject_property" if direction == "subject" else "object_property",
        )

        if direction == "subject":
            net.add_edge(entity_id, prop_id, label="may use", color="#0f766e")
            target_types = _sorted_entities(
                bundle,
                constraints.get("Value-type constraint", []),
            )[:max_types_per_relation]
            for target_id in target_types:
                _add_entity_node(net, bundle, target_id, "object", size=12)
                net.add_edge(prop_id, target_id, label="object type", color="#16a34a")
        else:
            net.add_edge(prop_id, entity_id, label="may target", color="#c2410c")
            source_types = _sorted_entities(
                bundle,
                constraints.get("Subject type constraint", []),
            )[:max_types_per_relation]
            for source_id in source_types:
                _add_entity_node(net, bundle, source_id, "subject", size=12)
                net.add_edge(source_id, prop_id, label="subject type", color="#7c3aed")

    return net


def _property_graph(
    bundle: dict[str, Any],
    property_id: str,
    max_subject_types: int,
    max_object_types: int,
) -> Network:
    net = _network()
    _add_property_node(net, bundle, property_id, "property", size=28)

    constraints = bundle["prop_constraints"].get(property_id, {})
    subject_types = _sorted_entities(
        bundle,
        constraints.get("Subject type constraint", []),
    )[:max_subject_types]
    object_types = _sorted_entities(
        bundle,
        constraints.get("Value-type constraint", []),
    )[:max_object_types]

    for subject_id in subject_types:
        _add_entity_node(net, bundle, subject_id, "subject", size=14)
        net.add_edge(subject_id, property_id, label="valid subject", color="#7c3aed")

    for object_id in object_types:
        _add_entity_node(net, bundle, object_id, "object", size=14)
        net.add_edge(property_id, object_id, label="valid object", color="#16a34a")

    return net


def _stats(bundle: dict[str, Any]) -> dict[str, int]:
    entity_labels = bundle["entity_labels"]
    property_labels = bundle["property_labels"]
    return {
        "entity_types": len([key for key in entity_labels if key != "ANY"]),
        "properties": len(property_labels),
        "entity_aliases": sum(len(v) for v in bundle["entity_aliases"].values()),
        "property_aliases": sum(len(v) for v in bundle["property_aliases"].values()),
        "subject_constraint_keys": len(bundle["subj_constraint2prop"]),
        "object_constraint_keys": len(bundle["obj_constraint2prop"]),
        "property_constraint_rows": len(bundle["prop_constraints"]),
        "hierarchy_rows": len(bundle["entity_hierarchy"]),
    }


def _render_stats(bundle: dict[str, Any]) -> None:
    stats = _stats(bundle)
    cols = st.columns(4)
    cols[0].metric("Entity types", f"{stats['entity_types']:,}")
    cols[1].metric("Properties", f"{stats['properties']:,}")
    cols[2].metric("Entity aliases", f"{stats['entity_aliases']:,}")
    cols[3].metric("Property aliases", f"{stats['property_aliases']:,}")

    cols = st.columns(4)
    cols[0].metric("Subject rule keys", f"{stats['subject_constraint_keys']:,}")
    cols[1].metric("Object rule keys", f"{stats['object_constraint_keys']:,}")
    cols[2].metric("Property constraints", f"{stats['property_constraint_rows']:,}")
    cols[3].metric("Hierarchy rows", f"{stats['hierarchy_rows']:,}")


def _render_note(language: str) -> None:
    if language == "tr":
        text = (
            "<strong>TR profile:</strong> labels and aliases come from "
            "<code>ontology_mappings/tr/</code>. Constraint files are shared because "
            "the ontology is keyed by Wikidata Q/P identifiers."
        )
    else:
        text = (
            "<strong>EN profile:</strong> labels and aliases use the root mapping files. "
            "The same shared constraint graph is materialized into the English ontology DB."
        )
    st.markdown(f"<div class=\"ontology-note\">{text}</div>", unsafe_allow_html=True)


def _render_legend() -> None:
    items = [
        ("#dbeafe", "#2563eb", "Selected entity"),
        ("#e2e8f0", "#64748b", "Parent type"),
        ("#fef3c7", "#b45309", "Property"),
        ("#ede9fe", "#7c3aed", "Valid subject type"),
        ("#dcfce7", "#16a34a", "Valid object type"),
    ]
    chips = "".join(
        f"""
        <span class="ontology-chip">
            <span class="ontology-dot" style="background:{bg}; border-color:{border};"></span>
            {label}
        </span>
        """
        for bg, border, label in items
    )
    st.markdown(f"<div class=\"ontology-legend\">{chips}</div>", unsafe_allow_html=True)


def _render_entity_explorer(bundle: dict[str, Any], key_prefix: str) -> None:
    entity_ids = _sorted_entities(bundle, list(bundle["entity_labels"].keys()))
    default_index = entity_ids.index("Q5") if "Q5" in entity_ids else 0

    control_cols = st.columns([2.1, 1, 1, 1])
    entity_id = control_cols[0].selectbox(
        "Entity type",
        entity_ids,
        index=default_index,
        format_func=lambda item: f"{_label_entity(bundle, item)} ({item})",
        key=f"{key_prefix}_entity",
    )
    direction_label = control_cols[1].selectbox(
        "Constraint side",
        ["As subject", "As object"],
        key=f"{key_prefix}_direction",
    )
    include_inherited = control_cols[2].checkbox(
        "Include inherited",
        value=True,
        key=f"{key_prefix}_inherited",
    )
    include_any = control_cols[3].checkbox(
        "Include ANY rules",
        value=False,
        key=f"{key_prefix}_any",
    )

    slider_cols = st.columns(2)
    max_relations = slider_cols[0].slider(
        "Relations in graph",
        min_value=4,
        max_value=40,
        value=14,
        step=2,
        key=f"{key_prefix}_entity_relation_limit",
    )
    max_types_per_relation = slider_cols[1].slider(
        "Type nodes per relation",
        min_value=1,
        max_value=6,
        value=2,
        key=f"{key_prefix}_entity_type_limit",
    )

    direction = "subject" if direction_label == "As subject" else "object"
    prop_sources = _constraint_sources(
        bundle,
        entity_id,
        direction,
        include_inherited=include_inherited,
        include_any=include_any,
    )
    parent_ids = _dedupe(bundle["entity_hierarchy"].get(entity_id, []))

    cols = st.columns(4)
    cols[0].metric("Selected type", entity_id)
    cols[1].metric("Parent types", f"{len(parent_ids):,}")
    cols[2].metric("Allowed relations", f"{len(prop_sources):,}")
    cols[3].metric("Aliases", f"{len(bundle['entity_aliases'].get(entity_id, [])):,}")

    _render_legend()
    net = _entity_graph(
        bundle,
        entity_id,
        direction,
        max_relations=max_relations,
        max_types_per_relation=max_types_per_relation,
        include_inherited=include_inherited,
        include_any=include_any,
    )
    _render_network(net)

    table_tabs = st.tabs(["Allowed relations", "Hierarchy", "Source JSON"])
    with table_tabs[0]:
        row_limit = st.slider(
            "Rows",
            min_value=25,
            max_value=500,
            value=100,
            step=25,
            key=f"{key_prefix}_entity_relation_rows",
        )
        df = _property_rows(bundle, set(prop_sources.keys()), prop_sources, limit=row_limit)
        if df.empty:
            st.info("No relation constraints found for this selection.")
        else:
            st.dataframe(df, width="stretch", hide_index=True)

    with table_tabs[1]:
        parent_df = _entity_rows(bundle, parent_ids, limit=250)
        if parent_df.empty:
            st.info("No parent type hierarchy found for this entity.")
        else:
            st.dataframe(parent_df, width="stretch", hide_index=True)

    with table_tabs[2]:
        st.json(
            {
                "entity_type2label": {
                    entity_id: bundle["entity_labels"].get(entity_id),
                },
                "entity_type2aliases": {
                    entity_id: bundle["entity_aliases"].get(entity_id, []),
                },
                "entity_type2hierarchy": {
                    entity_id: bundle["entity_hierarchy"].get(entity_id, []),
                },
                "subj_constraint2prop": {
                    entity_id: bundle["subj_constraint2prop"].get(entity_id, []),
                    "<ANY SUBJECT>": bundle["subj_constraint2prop"].get("<ANY SUBJECT>", []),
                },
                "obj_constraint2prop": {
                    entity_id: bundle["obj_constraint2prop"].get(entity_id, []),
                    "<ANY OBJECT>": bundle["obj_constraint2prop"].get("<ANY OBJECT>", []),
                },
            },
            expanded=False,
        )


def _render_property_explorer(bundle: dict[str, Any], key_prefix: str) -> None:
    property_ids = _sorted_properties(bundle, list(bundle["property_labels"].keys()))
    default_id = "P19" if "P19" in property_ids else property_ids[0]
    default_index = property_ids.index(default_id)

    control_cols = st.columns([2.2, 1, 1])
    property_id = control_cols[0].selectbox(
        "Property",
        property_ids,
        index=default_index,
        format_func=lambda item: f"{_label_property(bundle, item)} ({item})",
        key=f"{key_prefix}_property",
    )
    max_subject_types = control_cols[1].slider(
        "Subject type nodes",
        min_value=2,
        max_value=40,
        value=12,
        step=2,
        key=f"{key_prefix}_prop_subject_limit",
    )
    max_object_types = control_cols[2].slider(
        "Object type nodes",
        min_value=2,
        max_value=40,
        value=12,
        step=2,
        key=f"{key_prefix}_prop_object_limit",
    )

    constraints = bundle["prop_constraints"].get(property_id, {})
    subject_types = constraints.get("Subject type constraint", [])
    object_types = constraints.get("Value-type constraint", [])
    aliases = bundle["property_aliases"].get(property_id, [])

    cols = st.columns(4)
    cols[0].metric("Property", property_id)
    cols[1].metric("Data type", bundle["prop_data_type"].get(property_id, "-"))
    cols[2].metric("Valid subject types", f"{len(subject_types):,}")
    cols[3].metric("Valid object types", f"{len(object_types):,}")

    if aliases:
        st.caption("Aliases: " + ", ".join(str(alias) for alias in aliases[:14]))

    _render_legend()
    net = _property_graph(
        bundle,
        property_id,
        max_subject_types=max_subject_types,
        max_object_types=max_object_types,
    )
    _render_network(net)

    table_tabs = st.tabs(["Subject types", "Object types", "Source JSON"])
    with table_tabs[0]:
        df = _entity_rows(bundle, subject_types, limit=500)
        if df.empty:
            st.info("No subject type constraints found for this property.")
        else:
            st.dataframe(df, width="stretch", hide_index=True)

    with table_tabs[1]:
        df = _entity_rows(bundle, object_types, limit=500)
        if df.empty:
            st.info("No object/value type constraints found for this property.")
        else:
            st.dataframe(df, width="stretch", hide_index=True)

    with table_tabs[2]:
        st.json(
            {
                "prop2label": {property_id: bundle["property_labels"].get(property_id)},
                "prop2aliases": {property_id: aliases},
                "prop2data_type": {property_id: bundle["prop_data_type"].get(property_id)},
                "prop2constraints": {property_id: constraints},
                "subject_object_constraints": {
                    property_id: bundle["subject_object_constraints"].get(property_id, {})
                },
            },
            expanded=False,
        )


def _runtime_profile_rows(language: str) -> pd.DataFrame:
    rows: list[dict[str, str | int]] = []
    ontology_profiles = [
        profile
        for profile in get_available_ontology_profiles()
        if profile.language == language
    ]
    for ontology_profile in ontology_profiles:
        for embedding_profile in get_available_embedding_profiles():
            if language not in embedding_profile.compatible_languages:
                continue
            runtime_profile = resolve_runtime_profile(
                ontology_profile.profile_id,
                embedding_profile.profile_id,
            )
            rows.append(
                {
                    "runtime_profile": runtime_profile.profile_id,
                    "ontology_db": runtime_profile.ontology_db_name,
                    "triplets_db": runtime_profile.triplets_db_name,
                    "entity_type_aliases": runtime_profile.entity_type_aliases_collection_name,
                    "property_aliases": runtime_profile.property_aliases_collection_name,
                    "entity_aliases": runtime_profile.entity_aliases_collection_name,
                    "embedding_dimension": runtime_profile.embedding_dimension,
                }
            )
    return pd.DataFrame(rows)


def _mapping_file_rows(bundle: dict[str, Any], language: str) -> pd.DataFrame:
    lang_dir = "src/wikontic/utils/ontology_mappings"
    localized_prefix = lang_dir if language == "en" else f"{lang_dir}/{language}"
    rows = [
        {
            "layer": "localized",
            "file": f"{localized_prefix}/entity_type2label.json",
            "records": len(bundle["entity_labels"]),
            "used_for": "entity_types.label",
        },
        {
            "layer": "localized",
            "file": f"{localized_prefix}/entity_type2aliases.json",
            "records": sum(len(v) for v in bundle["entity_aliases"].values()),
            "used_for": "entity_type_aliases__{embedding_key}",
        },
        {
            "layer": "localized",
            "file": f"{localized_prefix}/prop2label.json",
            "records": len(bundle["property_labels"]),
            "used_for": "properties.label",
        },
        {
            "layer": "localized",
            "file": f"{localized_prefix}/prop2aliases.json",
            "records": sum(len(v) for v in bundle["property_aliases"].values()),
            "used_for": "property_aliases__{embedding_key}",
        },
        {
            "layer": "shared",
            "file": f"{lang_dir}/entity_type2hierarchy.json",
            "records": len(bundle["entity_hierarchy"]),
            "used_for": "entity_types.parent_type_ids",
        },
        {
            "layer": "shared",
            "file": f"{lang_dir}/prop2constraints.json",
            "records": len(bundle["prop_constraints"]),
            "used_for": "properties.valid_subject_type_ids / valid_object_type_ids",
        },
        {
            "layer": "shared",
            "file": f"{lang_dir}/subj_constraint2prop.json",
            "records": len(bundle["subj_constraint2prop"]),
            "used_for": "entity_types.valid_subject_property_ids",
        },
        {
            "layer": "shared",
            "file": f"{lang_dir}/obj_constraint2prop.json",
            "records": len(bundle["obj_constraint2prop"]),
            "used_for": "entity_types.valid_object_property_ids",
        },
        {
            "layer": "shared",
            "file": f"{lang_dir}/prop2data_type.json",
            "records": len(bundle["prop_data_type"]),
            "used_for": "property metadata / object value type display",
        },
        {
            "layer": "shared",
            "file": f"{lang_dir}/subject_object_constraints.json",
            "records": len(bundle["subject_object_constraints"]),
            "used_for": "raw Wikidata constraint audit source",
        },
    ]
    return pd.DataFrame(rows)


def _database_materialization_rows(language: str) -> pd.DataFrame:
    ontology_db = f"ontology__{language}"
    return pd.DataFrame(
        [
            {
                "collection": f"{ontology_db}.entity_types",
                "source": "entity_type2label + entity_type2hierarchy + subj/obj constraint maps",
                "key_fields": "entity_type_id, label, parent_type_ids, valid_subject_property_ids, valid_object_property_ids",
            },
            {
                "collection": f"{ontology_db}.properties",
                "source": "prop2label + prop2constraints",
                "key_fields": "property_id, label, valid_subject_type_ids, valid_object_type_ids",
            },
            {
                "collection": f"{ontology_db}.entity_type_aliases__{{embedding_key}}",
                "source": "entity_type2label + entity_type2aliases",
                "key_fields": "entity_type_id, alias_label, alias_text_embedding",
            },
            {
                "collection": f"{ontology_db}.property_aliases__{{embedding_key}}",
                "source": "prop2label + prop2aliases",
                "key_fields": "relation_id, alias_label, alias_text_embedding",
            },
            {
                "collection": "triplets.entity_aliases__{embedding_key}",
                "source": "runtime user KG entity aliases",
                "key_fields": "sample_id, label, entity_type, alias, alias_text_embedding",
            },
            {
                "collection": "triplets.triplets / initial_triplets / filtered_triplets / ontology_filtered_triplets",
                "source": "extraction pipeline output",
                "key_fields": "sample_id, subject, relation, object, subject_type, object_type",
            },
        ]
    )


def _render_source_files(bundle: dict[str, Any], language: str, key_prefix: str) -> None:
    st.subheader("Mapping files")
    st.dataframe(
        _mapping_file_rows(bundle, language),
        width="stretch",
        hide_index=True,
    )

    st.subheader("Database materialization")
    st.dataframe(
        _database_materialization_rows(language),
        width="stretch",
        hide_index=True,
    )

    st.subheader("Runtime profiles")
    runtime_df = _runtime_profile_rows(language)
    if runtime_df.empty:
        st.info("No available runtime profile found for this ontology language.")
    else:
        st.dataframe(runtime_df, width="stretch", hide_index=True)

    st.download_button(
        "Download ontology bundle JSON",
        data=_bundle_download_json(language),
        file_name=f"wikontic_ontology_{language}_bundle.json",
        mime="application/json",
        key=f"{key_prefix}_download_bundle",
    )


def _render_language_view(language: str) -> None:
    bundle = _load_bundle(language)
    _render_note(language)
    _render_stats(bundle)

    tabs = st.tabs(["Entity graph", "Property graph", "Files and DB"])
    with tabs[0]:
        st.subheader("Entity type rules")
        st.caption(
            "Shows parent types plus the relations this entity type may use as a subject "
            "or receive as an object. Inherited constraints are collected from parent Q-types."
        )
        _render_entity_explorer(bundle, key_prefix=f"{language}_entity")

    with tabs[1]:
        st.subheader("Property constraints")
        st.caption(
            "Shows which entity types are allowed on the subject and object side of a Wikidata property."
        )
        _render_property_explorer(bundle, key_prefix=f"{language}_property")

    with tabs[2]:
        _render_source_files(bundle, language, key_prefix=f"{language}_files")


_style_page()
_render_header()

language_tabs = st.tabs([f"{LANGUAGE_LABELS['en']} ontology", f"{LANGUAGE_LABELS['tr']} ontology"])
with language_tabs[0]:
    _render_language_view("en")
with language_tabs[1]:
    _render_language_view("tr")
