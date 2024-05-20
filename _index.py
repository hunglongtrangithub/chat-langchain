import os
import json
import csv
import requests
from uuid import uuid4
from bson import ObjectId
import meilisearch
from config import settings

EMBEDDER_NAME = "default"
INDEX_NAME = "clinical_docs"
# EMBEDDINGS_MODEL_NAME = "BAAI/bge-base-en-v1.5"
EMBEDDINGS_MODEL_NAME = "Salesforce/SFR-Embedding-Mistral"
MEILI_API_URL = "http://" + (settings.meili_http_addr or "localhost:7700")


def get_api_keys():
    master_client = meilisearch.Client(
        url=MEILI_API_URL,
        api_key=settings.meili_master_key.get_secret_value(),
    )

    search_actions = ["search"]
    search_indexes = [INDEX_NAME]
    admin_actions = ["*"]
    admin_indexes = ["*"]
    search_key = None
    admin_key = None

    # Fetch all existing keys
    keys = master_client.get_keys().results

    # Check if keys with the specified actions and indexes exist
    for key in keys:
        if search_key and admin_key:
            break
        elif key.actions == search_actions and key.indexes == search_indexes:
            print("Found existing search key.")
            search_key = key
        elif key.actions == admin_actions and key.indexes == admin_indexes:
            print("Found existing admin key.")
            admin_key = key

    # Create keys if they do not exist
    if not search_key:
        print("Creating search key...")
        search_uid = str(uuid4())
        master_client.create_key(
            options={
                "uid": search_uid,
                "description": "API key for the clinical docs search app",
                "actions": search_actions,
                "indexes": search_indexes,
                "expiresAt": None,
            }
        )
        search_key = master_client.get_key(search_uid)

    if not admin_key:
        print("Creating admin key...")
        admin_uid = str(uuid4())
        master_client.create_key(
            options={
                "uid": admin_uid,
                "description": "API key for the clinical docs admin app",
                "actions": admin_actions,
                "indexes": admin_indexes,
                "expiresAt": None,
            }
        )
        admin_key = master_client.get_key(admin_uid)

    return search_key.key, admin_key.key


SEARCH_API_KEY, ADMIN_API_KEY = get_api_keys()


docs_folder = "docs"
csv_file_path = f"{docs_folder}/processed_docs.csv"


def process_documents():
    with open(csv_file_path, mode="w", newline="", encoding="utf-8") as csv_file:
        fieldnames = ["id", "title", "source", "page_content"]
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()  # Write the header only once

        doc_count = 0
        for patient_folder in os.listdir(docs_folder):
            patient_path = os.path.join(docs_folder, patient_folder)
            if os.path.isdir(patient_path):
                for txt_file in os.listdir(patient_path):
                    if txt_file.endswith(".txt"):
                        file_path = os.path.join(patient_path, txt_file)
                        with open(file_path, "r", encoding="utf-8") as f:
                            page_content = f.read()

                        document = {
                            "id": str(ObjectId()),  # Generate MongoDB-style ObjectID
                            "title": txt_file,
                            "source": f"{docs_folder}/{patient_folder}/{txt_file}",
                            "page_content": page_content,
                        }
                        writer.writerow(document)  # Append document to CSV file
                        doc_count += 1
                        print(document["source"])
        print(f"Processed {doc_count} documents.")


def index_documents_to_meili():
    admin_client = meilisearch.Client(url=MEILI_API_URL, api_key=ADMIN_API_KEY)
    admin_client.create_index(uid=INDEX_NAME, options={"primaryKey": "id"})
    index = admin_client.index(INDEX_NAME)

    # enable vector search
    data = {"vectorStore": True}
    response = requests.patch(
        url=f"{MEILI_API_URL}/experimental-features",
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {ADMIN_API_KEY}",
        },
        data=json.dumps(data),
    )
    if response.status_code == 200:
        print("Vector search enabled successfully!")
    else:
        print("Error enabling vector search:", response.status_code)
        return

    # add the embedder
    settings = {
        "embedders": {
            EMBEDDER_NAME: {
                "source": "huggingFace",
                "model": EMBEDDINGS_MODEL_NAME,
                "documentTemplate": "A clinical document titled {{doc.title}} whose content is {{doc.page_content}}",
            }
        },
        "filterableAttributes": ["title", "source"],
        "sortableAttributes": ["title", "source"],
        "searchableAttributes": ["title", "source", "page_content"],
    }
    index.update_settings(settings)

    index.delete_all_documents()  # Clear the index before adding new documents
    with open(csv_file_path, mode="r", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            index.add_documents([row])  # Index each document individually


def index_docs():
    process_documents()
    print(f"Documents processed and saved to {csv_file_path}.")
    index_documents_to_meili()
    print("Documents indexed to Meilisearch.")


def test_search_meili():
    index = meilisearch.Client(
        url=MEILI_API_URL,
        api_key=SEARCH_API_KEY,
    ).index(INDEX_NAME)

    def search_documents(query):
        opt_params = {
            "hybrid": {
                "semanticRatio": 0.5,
                "embedder": EMBEDDER_NAME,
            },
            "limit": 5,
        }
        return index.search(query, opt_params)["hits"]

    # Example query
    query = "Fake Patient1"
    documents = search_documents(query)
    print(f"Found {len(documents)} documents for query: {query}")
    # Print the results
    for doc in documents:
        print(f"ID: {doc['id']}")
        print(f"Title: {doc['title']}")
        print(f"Source: {doc['source']}")
        print("-" * 40)


if __name__ == "__main__":
    index_docs()
    # test_search_meili()
    # get_api_keys()
    # process_documents()
    pass
