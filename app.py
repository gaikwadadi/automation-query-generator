# app.py
import os
import sys
import json
import hashlib
import datetime
import pandas as pd
import streamlit as st
import google.generativeai as genai
from pymongo import MongoClient

# UTF-8 encoding fix
os.environ['PYTHONIOENCODING'] = 'utf-8'
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

# Page config
st.set_page_config(page_title="MongoDB Query Generator", page_icon="🤖", layout="wide")
st.title("🤖 AI-powered MongoDB Query Generator")

# User API Key Input
api_key = st.text_input("🔑 Enter your Google API Key:", type="password")
if api_key:
    genai.configure(api_key=api_key)

# --- MongoDB Settings ---
# Defaults from Streamlit Secrets
mongo_uri_default = st.secrets.get("MONGO_URI", "mongodb://localhost:27017/")
db_name_default = st.secrets.get("MONGO_DB", "mydb")
collection_name_default = st.secrets.get("COLLECTION_NAME", "mycollection")

# Sidebar inputs (optional overrides)
st.sidebar.header("⚙️ MongoDB Settings")
mongo_uri = st.sidebar.text_input("MongoDB URI", mongo_uri_default)
db_name = st.sidebar.text_input("Database Name", db_name_default)
collection_name = st.sidebar.text_input("Collection Name", collection_name_default)
exec_mode = st.sidebar.selectbox("Execution Mode (Auto recommended)", ["Auto", "Find", "Aggregate"])

# --- Import helpers from query_utils ---
from query_utils import (
    query_cache,
    save_cache,
    get_mongo_client,
    load_model,
    is_complex_query,
    sanitize_llm_output,
    extract_first_bracketed,
    parse_query_string,
    add_dollar_to_stage_and_ops,
    convert_bson_types,
    prepare_dataframe,
    generate_mongo_query_auto,
    generate_mongo_query,
    get_cached_or_generate_query,
    execute_query,
    get_schema_info,
)

# --- CSV Upload & Import ---
st.subheader("📂 Upload CSV to MongoDB")
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if st.button("Clear Collection"):
    client = get_mongo_client(mongo_uri)
    if client:
        db = client[db_name]
        col = db[collection_name]
        col.delete_many({})
        st.success("🗑️ Collection cleared. You can now upload a new CSV.")

if uploaded_file is not None and st.button("Import CSV to MongoDB"):
    try:
        df = pd.read_csv(uploaded_file)
        client = get_mongo_client(mongo_uri)
        if not client:
            st.error("Cannot connect to MongoDB.")
        else:
            db = client[db_name]
            col = db[collection_name]

            existing_docs = list(col.find({}, {"_id": 0}))
            existing_hashes = set(
                hashlib.md5(json.dumps(d, sort_keys=True, default=str).encode()).hexdigest() for d in existing_docs
            )

            data_to_insert = []
            duplicates_skipped = 0
            for row in df.to_dict(orient="records"):
                row_hash = hashlib.md5(json.dumps(row, sort_keys=True, default=str).encode()).hexdigest()
                if row_hash not in existing_hashes:
                    data_to_insert.append(row)
                else:
                    duplicates_skipped += 1

            if data_to_insert:
                col.insert_many(data_to_insert)
                st.success(f"✅ Inserted {len(data_to_insert)} new rows (duplicates skipped: {duplicates_skipped}).")
                st.dataframe(pd.DataFrame(data_to_insert).head())
            else:
                st.warning(f"⚠️ All rows already exist. No new data inserted. Duplicates skipped: {duplicates_skipped}")

    except Exception as e:
        st.error(f"❌ Error importing CSV: {e}")

# --- Query section ---
st.subheader("💬 Ask in Natural Language")
user_input = st.text_area("Enter your question:", placeholder="e.g. Which products have a rating above 4.5 and more than 100 reviews?")
show_query = st.checkbox("Show Generated Query")
force_regen = st.checkbox("Force regenerate (ignore cache)")

if st.button("Generate & Run Query"):
    if not api_key:
        st.error("⚠️ Please enter your Google API Key")
    elif not user_input or not user_input.strip():
        st.error("⚠️ Enter a natural language question first.")
    else:
        schema_info = get_schema_info(mongo_uri, db_name, collection_name)
        with st.spinner("Generating MongoDB query..."):
            if force_regen:
                mongo_query = generate_mongo_query_auto(user_input, schema_info)
                key = hashlib.md5(f"{db_name}_{collection_name}_{user_input}".encode()).hexdigest()
                query_cache[key] = mongo_query
                save_cache()
                cached = False
            else:
                mongo_query, cached = get_cached_or_generate_query(user_input, schema_info, db_name, collection_name)

        if cached:
            st.info("💡 Using cached query (from previous request)")

        if show_query:
            st.subheader("📝 Generated Query (sanitized view)")
            st.code(sanitize_llm_output(mongo_query), language="python")

        # Execute query with chosen mode
        with st.spinner("Executing query..."):
            results = execute_query(mongo_query, mode=exec_mode, mongo_uri=mongo_uri, db_name=db_name, collection_name=collection_name)

        if isinstance(results, dict) and "error" in results:
            st.error(results["error"])
            st.subheader("🔍 Debug Info")
            st.write("Sanitized LLM output preview:")
            st.text(sanitize_llm_output(mongo_query)[:2000])
        else:
            st.success("✅ Query Executed Successfully!")
            if isinstance(results, list):
                df_preview = prepare_dataframe(results)
                st.subheader("📊 Results Table")
                st.dataframe(df_preview)
                csv = df_preview.to_csv(index=False).encode('utf-8')
                st.download_button("Download results as CSV", csv, "results.csv", "text/csv")
