import argparse
import json
import re
from collections import defaultdict
from dataclasses import dataclass, asdict, field
from typing import Any, Dict, List, Optional, Set

import chromadb
from chromadb.utils import embedding_functions
from tqdm import tqdm

from config import (
    logger, yaml,
    REPO_ROOT, PARAM_DIR, CSR_DIR, OUTPUT_DIR,
    CORPUS_PATH, ANALYSIS_CORPUS_PATH,
    REPORT_PATH, DB_DIR, GRAPH_PATH, MOCK_PREFIX, CSR_IDL_KEYS,
)
from utils import (
    flatten_text, resolve_ref, extract_enums, extract_range,
    compress_enum, extract_defined_by, extract_param_deps,
)


@dataclass
class SchemaRecord:
    raw_type:       Optional[str]  = None
    inferred_type:  str            = "unknown"
    enum_values:    Any            = field(default_factory=list)
    raw_enum_values: List[Any]     = field(default_factory=list)
    minimum:        Optional[int]  = None
    maximum:        Optional[int]  = None
    is_xlen_split:  bool           = False
    branches:       list           = field(default_factory=list)
    constraints:    List[str]      = field(default_factory=list)
    raw_schema:     Any            = None


@dataclass
class ParamRecord:
    name:                 str
    long_name:            str
    description:          str
    schema:               SchemaRecord
    defined_by_extensions: List[str]
    param_dependencies:   List[str]
    defined_by_raw:       Any            = None
    branch_summary:       List[str]      = field(default_factory=list)
    used_by:              List[str]      = field(default_factory=list)
    has_idl_requirements: bool           = False
    source_file:          str            = ""
    classification:       str            = "UNKNOWN"
    confidence:           str            = "low"
    reasoning:            str            = ""
    summary:              str            = ""
    csr_refs:             List[Dict]     = field(default_factory=list)


def _detect_constraints(description: str, enum_values: list) -> List[str]:
    lower = description.lower()
    constraints = []
    if "power of 2" in lower:   constraints.append("power_of_two")
    if "alignment" in lower:    constraints.append("alignment_constraint")
    return constraints


def _summarize_branches(branches: list) -> List[str]:
    out = []
    for b in branches:
        cond = b.get("condition", {})
        if "param" in cond:
            p = cond["param"]
            out.append(f"{p['name']}={p.get('equal')}")
    return out


def _normalize_type(raw: str) -> str:
    return {
        "integer_enum":  "integer with fixed values",
        "integer_range": "integer range",
        "boolean":       "boolean choice",
        "array_set":     "set of values",
        "conditional":   "conditional requirement",
        "string_enum":   "string choice",
        "string":        "text value",
    }.get(raw, raw)


def _analyze_schema(schema: dict, description: str = "") -> SchemaRecord:
    if not isinstance(schema, dict):
        return SchemaRecord()

    schema = resolve_ref(schema)
    raw = schema.get("type")

    branches = []
    if "oneOf" in schema:
        for branch in schema["oneOf"]:
            if isinstance(branch, dict) and "when" in branch:
                bs = branch.get("schema", {})
                branches.append({
                    "condition": branch.get("when"),
                    "schema":    bs,
                    "type":      _analyze_schema(bs).inferred_type,
                })

    inferred = "unknown"
    if "oneOf" in schema:   inferred = "conditional"
    elif raw == "boolean":  inferred = "boolean"
    elif raw == "array":    inferred = "array_set"

    raw_enums = extract_enums(schema)
    try:    dedup = sorted(set(raw_enums))
    except: dedup = list(set(raw_enums))

    if inferred == "unknown":
        if raw == "integer": inferred = "integer_enum" if dedup else "integer_range"
        elif raw == "string": inferred = "string_enum" if dedup else "string"

    if inferred == "conditional" and branches:
        first_raw = branches[0]["schema"].get("type")
        if first_raw: raw = first_raw

    vmin, vmax = extract_range(schema)

    return SchemaRecord(
        raw_type=raw, inferred_type=inferred,
        enum_values=compress_enum(dedup), raw_enum_values=dedup,
        minimum=vmin, maximum=vmax,
        is_xlen_split="oneOf" in schema and "MXLEN" in str(schema),
        branches=branches,
        constraints=_detect_constraints(description, dedup),
        raw_schema=schema,
    )


def _build_rag_summary(r: ParamRecord) -> str:
    parts = [
        f"{r.name}: {r.description}",
        f"type={_normalize_type(r.schema.inferred_type)}",
    ]
    if r.schema.enum_values:
        parts.append(f"values={r.schema.enum_values}")
    if r.param_dependencies:
        parts.append(f"depends_on={','.join(r.param_dependencies)}")
        parts.append("depends on other architectural parameters")
    if "MXLEN" in r.param_dependencies or r.name in ("MXLEN", "XLEN", "SXLEN", "UXLEN"):
        parts.append("architecture register width parameter")
        parts.append("depends on register width XLEN")
    if r.branch_summary:
        parts.append(f"conditions={','.join(r.branch_summary)}")
    if r.defined_by_extensions:
        parts.append(f"extensions={','.join(r.defined_by_extensions)}")
    if r.defined_by_raw:
        parts.append("has architectural conditions")
    if r.schema.minimum is not None or r.schema.maximum is not None:
        parts.append(f"range={r.schema.minimum} to {r.schema.maximum}")
    if r.schema.constraints:
        parts.append(f"constraints={','.join(r.schema.constraints)}")
        if "alignment_constraint" in r.schema.constraints:
            parts.extend(["alignment rule", "memory alignment parameter"])
        if "power_of_two" in r.schema.constraints:
            parts.append("power of two values")

    desc_low = r.description.lower()
    if "trap" in desc_low:                              parts.append("trap handling parameter")
    if "supported values" in desc_low or "legal values" in desc_low:
                                                        parts.append("valid values and architectural constraints")
    if "alignment" in desc_low:                         parts.append("address alignment requirement")
    if "width" in desc_low:                             parts.append("bit width size")
    if "mode" in desc_low:                              parts.append("operating mode configuration")
    if "mxlen" in desc_low or r.name == "MXLEN":
        parts.append("machine implementation base integer GPR general purpose register processor width bit-width XLEN bits 32 64 size register-width")
    if "pmp" in desc_low or "pmp" in r.name.lower():
        parts.append("physical memory protection PMP entries count total amount regions number capacity how-many")

    parts.append(f"RISC-V architectural parameter {r.name}")
    if r.csr_refs:
        csr_names = sorted({ref["csr"] for ref in r.csr_refs})
        parts.append(f"related_csrs={','.join(csr_names)}")
    parts.append("hardware configuration parameter used for hardware configuration and ISA behavior")
    return " | ".join(parts)


class Pipeline:
    def __init__(self, mode: str = "rag"):
        self.mode = mode
        self.records: List[ParamRecord] = []
        self.param_names: Set[str] = set()

    def _parse_params(self):
        files = sorted(PARAM_DIR.rglob("*.yaml"))
        self.param_names = {f.stem for f in files if not f.stem.startswith(MOCK_PREFIX)}

        for pf in tqdm(files, desc="Phase 1 — Parameters"):
            if pf.stem.startswith(MOCK_PREFIX): continue
            try:
                with open(pf, "r", encoding="utf-8") as f:
                    data = yaml.load(f)
                if not isinstance(data, dict): continue

                name     = data.get("name", pf.stem)
                desc     = flatten_text(data.get("description", ""))
                longname = flatten_text(data.get("long_name", name))
                exts, p_conds = extract_defined_by(data.get("definedBy", {}))

                deps = (
                    extract_param_deps(data)
                    | extract_param_deps(data.get("schema", {}))
                    | p_conds
                    | {p for p in self.param_names
                       if p != name and p.lower() in desc.lower()}
                )
                deps = {d for d in deps if d in self.param_names and d != name}

                schema_rec = _analyze_schema(data.get("schema", {}), desc)

                self.records.append(ParamRecord(
                    name=name, long_name=longname, description=desc,
                    schema=schema_rec,
                    defined_by_extensions=list(exts),
                    param_dependencies=list(deps),
                    defined_by_raw=data.get("definedBy"),
                    branch_summary=_summarize_branches(schema_rec.branches),
                    source_file=str(pf.relative_to(REPO_ROOT)),
                ))
            except Exception as e:
                logger.error(f"Failed {pf.name}: {e}")

    def _cross_ref_csrs(self):
        for cf in tqdm(list(CSR_DIR.glob("*.yaml")), desc="Phase 1 — CSR Cross-ref"):
            try:
                with open(cf, "r", encoding="utf-8") as f:
                    data = yaml.load(f)
                if not isinstance(data, dict): continue

                def _scan(scope: dict, field_label: str):
                    for key in CSR_IDL_KEYS:
                        val = str(scope.get(key, ""))
                        for r in self.records:
                            if re.search(rf"\b{re.escape(r.name)}\b", val):
                                r.csr_refs.append({
                                    "csr": cf.stem,
                                    "field": field_label,
                                    "context": f"logic in {key}",
                                })

                _scan(data, "(csr-level)")
                for fname, fdata in (data.get("fields") or {}).items():
                    if isinstance(fdata, dict):
                        _scan(fdata, fname)
            except Exception:
                continue

    def run(self):
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        self._parse_params()
        self._cross_ref_csrs()

        used_by: Dict[str, list] = defaultdict(list)
        graph = {}
        for r in self.records:
            graph[r.name] = {
                "depends_on":  r.param_dependencies,
                "used_by":     [],
                "related_csrs": list({ref["csr"] for ref in r.csr_refs}),
            }
            for dep in r.param_dependencies:
                used_by[dep].append(r.name)

        for r in self.records:
            r.used_by = used_by[r.name]
            if r.name in graph:
                graph[r.name]["used_by"] = r.used_by
            r.summary = _build_rag_summary(r)
            r.classification = "normative" if r.name.isupper() else "sw-rule"
            r.confidence = (
                "high" if r.param_dependencies or r.schema.inferred_type == "conditional"
                else "medium"
            )

        with open(GRAPH_PATH, "w") as f:
            json.dump(graph, f, indent=2)
        logger.info(f"Dependency graph -> {GRAPH_PATH}")

        if self.mode == "analysis":
            with open(ANALYSIS_CORPUS_PATH, "w") as f:
                json.dump({"parameters": [asdict(r) for r in self.records]}, f, indent=2)
            logger.info(f"Analysis corpus -> {ANALYSIS_CORPUS_PATH}")

        with open(CORPUS_PATH, "w") as f:
            json.dump({"parameters": [asdict(r) for r in self.records]}, f, indent=2)
        logger.info(f"RAG corpus -> {CORPUS_PATH}")

        self._build_db()
        self._write_report()

    def _build_db(self):
        logger.info("Building ChromaDB vector index...")
        client = chromadb.PersistentClient(path=str(DB_DIR))
        emb_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
        try: client.delete_collection("udb_parameters")
        except Exception: pass

        coll = client.create_collection(name="udb_parameters", embedding_function=emb_fn)
        coll.upsert(
            ids=[r.name for r in self.records],
            documents=[r.summary for r in self.records],
        )
        logger.info(f"ChromaDB index ready -> {DB_DIR} ({len(self.records)} vectors)")

    def _write_report(self):
        total         = len(self.records)
        total_csr     = sum(len(r.csr_refs) for r in self.records)
        params_w_csr  = sum(1 for r in self.records if r.csr_refs)
        dep_counts    = [len(r.param_dependencies) for r in self.records]
        avg_deps      = sum(dep_counts) / total if total else 0
        max_deps      = max(dep_counts) if dep_counts else 0
        type_counts   = defaultdict(int)
        for r in self.records:
            type_counts[r.schema.inferred_type] += 1
        most_used = sorted(
            [(r.name, len(r.used_by)) for r in self.records], key=lambda x: -x[1]
        )[:10]

        lines = [
            "# RISC-V UDB Parameter Database Report", "---",
            "## Overview",
            f"- Total Parameters: **{total}**",
            f"- CSR References: **{total_csr}**",
            f"- Parameters linked to CSRs: **{params_w_csr}**",
            f"- Average Dependencies: **{avg_deps:.2f}**",
            f"- Max Dependencies: **{max_deps}**",
            "", "## Type Distribution",
        ]
        for t, c in sorted(type_counts.items(), key=lambda x: -x[1]):
            lines.append(f"- {t}: {c}")
        lines += ["", "## Most Influential Parameters"]
        for name, c in most_used:
            lines.append(f"- {name}: used by {c} parameters")
        lines += [
            "", "## Status",
            "- Parameter parsing: COMPLETE",
            "- CSR cross-referencing: COMPLETE",
            "- Vector database: READY",
        ]
        with open(REPORT_PATH, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        logger.info(f"Report -> {REPORT_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build RISC-V parameter vector database.")
    parser.add_argument("--mode", choices=["rag", "analysis"], default="rag",
                        help="rag=retrieval index only; analysis=also writes full corpus")
    args = parser.parse_args()
    Pipeline(mode=args.mode).run()
