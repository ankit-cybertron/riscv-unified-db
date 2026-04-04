import hashlib
import re
from typing import Any

from config import SCHEMA_DEFS_DATA, CHUNK_KEYWORDS, DIR_TO_CHUNK_TYPE


def flatten_text(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        parts = []
        for item in value:
            parts.append(str(item["text"]) if isinstance(item, dict) and "text" in item else str(item))
        return " ".join(p for p in parts if p).strip()
    return str(value) if value is not None else ""


def normalize_text(text: str) -> str:
    return " ".join(text.split()).strip()


def is_relevant_chunk(text: str) -> bool:
    lower = text.lower()
    return any(kw in lower for kws in CHUNK_KEYWORDS.values() for kw in kws)


def chunk_id(source: str, path: str, index: int) -> str:
    return "chunk_" + hashlib.md5(f"{source}::{path}::{index}".encode()).hexdigest()[:8]


def infer_chunk_type(source_path, yaml_key_path: str) -> str:
    from pathlib import Path
    return DIR_TO_CHUNK_TYPE.get(Path(source_path).parent.name, "unknown")


def resolve_ref(obj: Any) -> Any:
    if isinstance(obj, dict):
        if "$ref" in obj:
            ref = obj["$ref"]
            resolved = {}
            if "#/$defs/" in ref:
                key = ref.split("#/$defs/")[-1]
                resolved = SCHEMA_DEFS_DATA.get("$defs", {}).get(key, {})
            obj = {**resolve_ref(resolved), **obj}
            del obj["$ref"]
        return {k: resolve_ref(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [resolve_ref(i) for i in obj]
    return obj


def extract_enums(obj: Any) -> list:
    enums = []
    if isinstance(obj, dict):
        if "enum" in obj:
            enums.extend(obj["enum"])
        if "$ref" in obj:
            ref = obj["$ref"]
            if "#/$defs/" in ref:
                key = ref.split("#/$defs/")[-1]
                enums.extend(extract_enums(SCHEMA_DEFS_DATA.get("$defs", {}).get(key, {})))
        for k in ("allOf", "anyOf", "oneOf", "items"):
            if k in obj:
                node = obj[k]
                targets = node if isinstance(node, list) else [node]
                for item in targets:
                    enums.extend(extract_enums(item))
    elif isinstance(obj, list):
        for item in obj:
            enums.extend(extract_enums(item))
    return enums


def extract_range(obj: Any):
    mins, maxs = [], []
    def _walk(node):
        if isinstance(node, dict):
            if "minimum" in node: mins.append(node["minimum"])
            if "maximum" in node: maxs.append(node["maximum"])
            for v in node.values(): _walk(v)
        elif isinstance(node, list):
            for i in node: _walk(i)
    _walk(obj)
    return (min(mins) if mins else None, max(maxs) if maxs else None)


def compress_enum(enum_list: list):
    if len(enum_list) > 20:
        try:
            return f"{len(enum_list)} values, min={min(enum_list)}, max={max(enum_list)}"
        except Exception:
            return f"{len(enum_list)} values"
    return enum_list


def extract_defined_by(obj: Any):
    exts, params = set(), set()
    if isinstance(obj, dict):
        if "extension" in obj:
            ext = obj["extension"]
            if isinstance(ext, str): exts.add(ext)
            elif isinstance(ext, dict) and "name" in ext: exts.add(ext["name"])
        if "param" in obj and isinstance(obj["param"], dict):
            name = obj["param"].get("name")
            if name: params.add(name)
        for v in obj.values():
            e, p = extract_defined_by(v)
            exts |= e; params |= p
    elif isinstance(obj, list):
        for item in obj:
            e, p = extract_defined_by(item)
            exts |= e; params |= p
    return exts, params


def extract_param_deps(obj: Any) -> set:
    deps = set()
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k == "param" and isinstance(v, dict) and v.get("name"):
                deps.add(v["name"])
            else:
                deps |= extract_param_deps(v)
    elif isinstance(obj, list):
        for item in obj: deps |= extract_param_deps(item)
    return deps
