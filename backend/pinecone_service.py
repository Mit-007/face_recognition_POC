import os
from pinecone import Pinecone, ServerlessSpec

from face_service import EMBEDDING_DIM

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "face-embeddings-poc")
PINECONE_CLOUD = os.getenv("PINECONE_CLOUD", "aws")
PINECONE_REGION = os.getenv("PINECONE_REGION", "us-east-1")

_pc = None
_index = None


def get_index():
    global _pc, _index
    if _index is not None:
        return _index

    if not PINECONE_API_KEY:
        raise RuntimeError(
            "PINECONE_API_KEY is not set. Fill in backend/.env before calling this endpoint."
        )

    _pc = Pinecone(api_key=PINECONE_API_KEY)

    existing = [idx["name"] for idx in _pc.list_indexes()]
    if PINECONE_INDEX_NAME not in existing:
        _pc.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=EMBEDDING_DIM,
            metric="cosine",
            spec=ServerlessSpec(cloud=PINECONE_CLOUD, region=PINECONE_REGION),
        )

    _index = _pc.Index(PINECONE_INDEX_NAME)
    return _index


def upsert_face(vector_id: str, embedding, name: str, image_name: str):
    index = get_index()
    index.upsert(
        vectors=[
            {
                "id": vector_id,
                "values": embedding.tolist(),
                "metadata": {"name": name, "image_name": image_name},
            }
        ]
    )


def search_face(embedding, top_k: int = 1):
    index = get_index()
    result = index.query(
        vector=embedding.tolist(),
        top_k=top_k,
        include_metadata=True,
        include_values=False,
    )
    return result.get("matches", [])


def delete_face(vector_id: str):
    try:
        index = get_index()
        index.delete(ids=[vector_id])
    except Exception:
        pass
