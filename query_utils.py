# query_utils.py
import os
import json
import ast
import re
import hashlib
import datetime
from typing import Any, Tuple, Optional
import pandas as pd
import streamlit as st
import google.generativeai as genai
from pymongo import MongoClient
from bson import ObjectId

# ------------------ QUERY CACHE ------------------
CACHE_FILE = "query_cache.json"
if os.path.exists(CACHE_FILE):
    try:
        with open(CACHE_FILE, "r") as f:
            query_cache = json.load(f)
    except Exception:
        query_cache = {}
else:
    query_cache = {}

def save_cache():
    try:
        with open(CACHE_FILE, "w") as f:
            json.dump(query_cache, f, indent=2)
    except Exception as e:
        try:
            st.warning(f"Could not save query cache: {e}")
        except Exception:
            pass

# ------------------ Mongo Client ------------------
@st.cache_resource
def get_mongo_client(uri: str) -> Optional[MongoClient]:
    try:
        client = MongoClient(uri, serverSelectionTimeoutMS=5000)
        client.server_info()
        return client
    except Exception as e:
        try:
            st.error(f"MongoDB connection error: {e}")
        except Exception:
            pass
        return None

# ------------------ LLM MODEL ------------------
@st.cache_resource
def load_model():
    # Keep the same model string you used in the app
    return genai.GenerativeModel("gemini-1.5-flash")

# ------------------ Query Complexity ------------------
def is_complex_query(user_input: str) -> bool:
    keywords = ["average", "sum", "top", "group by", "group_by", "count", "sort", "max", "min", "between", "aggregate", "join", "lookup"]
    u = user_input.lower()
    return any(k in u for k in keywords)

# ------------------ sanitize LLM output ------------------
def sanitize_llm_output(raw: str) -> str:
    if not raw:
        return raw
    text = raw.strip()

    # Remove code fences
    if text.startswith("```"):
        parts = text.split("\n", 1)
        if len(parts) > 1:
            text = parts[1]
    if text.endswith("```"):
        parts = text.rsplit("\n", 1)
        text = parts[0]

    cleaned_lines = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("import ") or stripped.startswith("from "):
            continue
        if stripped.startswith("#"):
            continue
        # naive inline comment removal outside quotes:
        if "#" in line:
            quote_count = line.count("'") + line.count('"')
            if quote_count % 2 == 0:
                line = line.split("#", 1)[0].rstrip()
        cleaned_lines.append(line)

    cleaned = "\n".join(cleaned_lines)
    # Normalize smart quotes
    cleaned = cleaned.replace("“", "\"").replace("”", "\"").replace("‘", "'").replace("’", "'")
    # Remove control characters except \n, \r, \t
    cleaned = "".join(ch for ch in cleaned if (ord(ch) >= 32 or ch in "\n\r\t"))
    # Trim repeated whitespace
    cleaned = re.sub(r"[ \t]+", " ", cleaned).strip()
    return cleaned

# ------------------ extract_first_bracketed ------------------
def extract_first_bracketed(s: str) -> str:
    s = s.strip()
    start_index = None
    start_char = None
    for i, ch in enumerate(s):
        if ch in "{[":
            start_index = i
            start_char = ch
            break
    if start_index is None:
        return s
    stack = [start_char]
    for j in range(start_index + 1, len(s)):
        c = s[j]
        if c in "{[":
            stack.append(c)
        elif c in "]}":
            if not stack:
                break
            stack.pop()
            if not stack:
                return s[start_index:j+1]
    return s

# ------------------ parse_query_string ------------------
def parse_query_string(sanitized: str) -> Tuple[Optional[Any], Optional[str]]:
    if not sanitized:
        return None, "Empty sanitized string."

    try_candidate = extract_first_bracketed(sanitized)
    # Try json.loads
    try:
        parsed = json.loads(try_candidate)
        return parsed, None
    except Exception:
        pass

    # Try ast.literal_eval
    try:
        parsed = ast.literal_eval(try_candidate)
        return parsed, None
    except Exception as e:
        # Fallback: replace Python True/False/None with JSON equivalents and try json.loads
        try_json = try_candidate.replace("None", "null").replace("True", "true").replace("False", "false")
        if "'" in try_json and '"' not in try_json:
            try_json = try_json.replace("'", "\"")
        try:
            parsed = json.loads(try_json)
            return parsed, None
        except Exception:
            return None, f"Failed to parse query. Last attempted parse error: {e}. Sanitized preview: {sanitized[:1000]}"

# ------------------ add_dollar_to_stage_and_ops ------------------
def add_dollar_to_stage_and_ops(parsed):
    mongo_stage_names = {
        "addFields", "match", "project", "group", "sort", "limit", "unwind", "lookup", "replaceRoot", "set", "unset", "count", "bucket", "facet"
    }
    mongo_ops = {"gt", "gte", "lt", "lte", "eq", "ne", "in", "nin", "exists", "regex", "size", "all", "push", "addToSet", "sum", "avg", "first", "last"}

    def fix_dict(d):
        if not isinstance(d, dict):
            return d
        new = {}
        for k, v in d.items():
            new_k = k
            if k in mongo_stage_names and not k.startswith("$"):
                new_k = f"${k}"
            if k in mongo_ops and not k.startswith("$"):
                new_k = f"${k}"
            if isinstance(v, dict):
                new_v = fix_dict(v)
            elif isinstance(v, list):
                new_v = [fix_dict(x) if isinstance(x, dict) else x for x in v]
            else:
                new_v = v
            new[new_k] = new_v
        return new

    if isinstance(parsed, list):
        return [fix_dict(stage) if isinstance(stage, dict) else stage for stage in parsed]
    elif isinstance(parsed, dict):
        if "filter" in parsed and isinstance(parsed["filter"], dict):
            return parsed["filter"]
        if "pipeline" in parsed and isinstance(parsed["pipeline"], list):
            return parsed["pipeline"]
        return fix_dict(parsed)
    else:
        return parsed

# ------------------ BSON & datetime conversion ------------------
def convert_bson_types(obj):
    if isinstance(obj, ObjectId):
        return str(obj)
    if isinstance(obj, datetime.datetime):
        return obj.isoformat()
    if isinstance(obj, datetime.date) and not isinstance(obj, datetime.datetime):
        return obj.isoformat()
    if isinstance(obj, dict):
        return {k: convert_bson_types(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [convert_bson_types(v) for v in obj]
    return obj

# ------------------ prepare_dataframe ------------------
def prepare_dataframe(docs: list) -> pd.DataFrame:
    cleaned_docs = [convert_bson_types(d) for d in docs]
    try:
        df = pd.json_normalize(cleaned_docs, sep='.')
    except Exception:
        df = pd.DataFrame(cleaned_docs)

    for col in df.columns:
        if col.endswith("_preview"):
            continue
        try:
            non_null = df[col].dropna()
            if non_null.empty:
                continue
            sample = non_null.iloc[0]
            if isinstance(sample, list):
                def preview_cell(val):
                    if not isinstance(val, list):
                        return val
                    names = []
                    for item in val:
                        if isinstance(item, dict):
                            for k in ("ProductName", "product_name", "name", "title"):
                                if k in item:
                                    names.append(str(item.get(k)))
                                    break
                            else:
                                names.append(json.dumps(item, ensure_ascii=False))
                        else:
                            names.append(str(item))
                    preview = ", ".join(names[:6])
                    if len(names) > 6:
                        preview += "..."
                    return preview
                df[f"{col}_preview"] = df[col].apply(preview_cell)
        except Exception:
            continue

    return df

# ------------------ Generate Query (LLM prompts) ------------------
def generate_mongo_query(user_input: str, schema_info: str) -> str:
    model = load_model()
    prompt = f"""
    You are an expert MongoDB query generator.
    Convert the following user request into a valid MongoDB query or aggregation pipeline.
    - If it's a simple filter, return a Python dict (e.g. {{ "price": {{ "$gt": 10 }} }}).
    - If it's analytical/aggregation, return a Python list representing the aggregation pipeline (e.g. [{{ "$match": {{...}} }}, {{ "$group": {{...}} }}]).
    IMPORTANT: Do not include code fences, comments, or import statements.
    Use ISO-8601 strings for date literals (e.g. "2022-01-01T00:00:00").
    Only return a Python literal (dict or list) composed of simple literals (strings, numbers, booleans, lists, dicts). Do not return Python datetime objects or function calls.
    Dataset schema: {schema_info}
    User request: "{user_input}"
    """
    response = model.generate_content(prompt)
    return response.text.strip()

def generate_mongo_query_complex(user_input: str, schema_info: str) -> str:
    model = load_model()
    prompt = f"""
    You are an expert MongoDB query generator.
    The user request is complex. If possible provide both:
      1) a filter dict (for a find) under the key 'filter'
      2) an aggregation pipeline list under the key 'pipeline'
    BUT if only one is suitable return only that (dict or list).
    IMPORTANT: Do not include code fences, comments, or import statements.
    Use ISO-8601 strings for date literals (e.g. "2022-01-01T00:00:00").
    Return only Python literals (dict/list) with simple types.
    Dataset schema: {schema_info}
    User request: "{user_input}"
    """
    response = model.generate_content(prompt)
    return response.text.strip()

def generate_mongo_query_auto(user_input: str, schema_info: str) -> str:
    if is_complex_query(user_input):
        return generate_mongo_query_complex(user_input, schema_info)
    else:
        return generate_mongo_query(user_input, schema_info)

# ------------------ Cached Query Retrieval ------------------
def get_cached_or_generate_query(user_input: str, schema_info: str, db_name_s: str, collection_name_s: str, force=False):
    key_str = f"{db_name_s}_{collection_name_s}_{user_input}"
    key_hash = hashlib.md5(key_str.encode()).hexdigest()
    if (not force) and key_hash in query_cache:
        return query_cache[key_hash], True
    else:
        query_text = generate_mongo_query_auto(user_input, schema_info)
        query_cache[key_hash] = query_text
        save_cache()
        return query_text, False

# ------------------ Execute Query ------------------
def execute_query(query_str: str, mode: str = "Auto", mongo_uri: str = None, db_name: str = None, collection_name: str = None):
    try:
        if not query_str or not isinstance(query_str, str):
            return {"error": "Empty or invalid query string."}

        sanitized = sanitize_llm_output(query_str)

        parsed, parse_err = parse_query_string(sanitized)
        if parse_err:
            return {"error": parse_err}

        parsed_fixed = add_dollar_to_stage_and_ops(parsed)

        client = get_mongo_client(mongo_uri)
        if not client:
            return {"error": "Cannot connect to MongoDB."}
        db = client[db_name]
        col = db[collection_name]

        results = []

        def is_pipeline(obj):
            stage_keys = {"$match", "$group", "$project", "$sort", "$limit", "$unwind", "$lookup", "$addFields", "$set", "$unset", "$replaceRoot", "$count", "$facet", "$bucket"}
            if isinstance(obj, list):
                return True
            if isinstance(obj, dict) and any(k in obj for k in stage_keys):
                return True
            return False

        if mode == "Find":
            if isinstance(parsed_fixed, dict) and any(k.startswith("$") for k in parsed_fixed.keys()):
                if "$match" in parsed_fixed and isinstance(parsed_fixed["$match"], dict):
                    filter_doc = parsed_fixed["$match"]
                else:
                    return {"error": "Requested Find mode, but parsed query looks like an aggregation stage. Please use Auto or Aggregate mode."}
            elif isinstance(parsed_fixed, dict):
                filter_doc = parsed_fixed
            else:
                return {"error": "Requested Find mode but parsed query is not a dict filter."}

            for doc in col.find(filter_doc):
                results.append(convert_bson_types(doc))

        elif mode == "Aggregate":
            if isinstance(parsed_fixed, list):
                pipeline = parsed_fixed
            elif isinstance(parsed_fixed, dict):
                if "pipeline" in parsed_fixed and isinstance(parsed_fixed["pipeline"], list):
                    pipeline = parsed_fixed["pipeline"]
                else:
                    pipeline = [parsed_fixed]
            else:
                return {"error": "Parsed query isn't a list or dict suitable for aggregate."}

            for doc in col.aggregate(pipeline):
                results.append(convert_bson_types(doc))

        else:  # Auto
            if isinstance(parsed_fixed, dict):
                if is_pipeline(parsed_fixed):
                    pipeline = [parsed_fixed]
                    for doc in col.aggregate(pipeline):
                        results.append(convert_bson_types(doc))
                else:
                    for doc in col.find(parsed_fixed):
                        results.append(convert_bson_types(doc))
            elif isinstance(parsed_fixed, list):
                for doc in col.aggregate(parsed_fixed):
                    results.append(convert_bson_types(doc))
            else:
                return {"error": "Invalid query format after parsing (not dict or list)."}

        if not results:
            return {"error": "No data found for this query."}
        return results

    except Exception as e:
        return {"error": str(e)}

# ------------------ Get Schema ------------------
def get_schema_info(mongo_uri: str, db_name: str, collection_name: str):
    client = get_mongo_client(mongo_uri)
    if not client:
        return "{}"
    try:
        db = client[db_name]
        col = db[collection_name]
        sample_doc = col.find_one()
        if sample_doc and "_id" in sample_doc:
            sample_doc["_id"] = str(sample_doc["_id"])
        return json.dumps(sample_doc, indent=2) if sample_doc else "{}"
    except Exception as e:
        return f"Error retrieving schema: {e}"
