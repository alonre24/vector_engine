
# Redis Configuration for Vector Similarity
import os
from redisvl.utils.vectorize import HFTextVectorizer
from redisvl.extensions.cache.embeddings import EmbeddingsCache

# Redis connection settings
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = os.getenv("REDIS_PORT", "6379")
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD", "")
REDIS_DB = os.getenv("REDIS_DB", "0")

# Construct Redis URL
if REDIS_PASSWORD:
    REDIS_URL = f"redis://:" + REDIS_PASSWORD + f"@" + REDIS_HOST + ":" + REDIS_PORT + "/" + REDIS_DB
else:
    REDIS_URL = f"redis://" + REDIS_HOST + ":" + REDIS_PORT + "/" + REDIS_DB

# Default vectorizer configuration
DEFAULT_VECTORIZER = HFTextVectorizer(
    model="sentence-transformers/all-MiniLM-L6-v2",
    cache=EmbeddingsCache(
        name="embedcache",
        ttl=600,
        redis_url=REDIS_URL
    )
)

# Vector dimensions for different models
VECTOR_DIMENSIONS = {
    "sentence-transformers/all-MiniLM-L6-v2": 384,
    "sentence-transformers/all-mpnet-base-v2": 768,
    "text-embedding-ada-002": 1536
}
