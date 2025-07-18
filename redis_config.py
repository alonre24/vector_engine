import os
import logging
from redisvl.utils.vectorize import HFTextVectorizer
from redisvl.extensions.cache.embeddings import EmbeddingsCache
from redisvl.extensions.llmcache import SemanticCache
from functools import wraps

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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

# Setup semantic cache
cache = SemanticCache(
    name="llm_cache",
    vectorizer=DEFAULT_VECTORIZER,
    redis_url=REDIS_URL,
    distance_threshold=0.1  # Adjust based on similarity requirements
)

# Cache decorator
def semantic_cache_decorator(cache_instance):
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Create cache key from function arguments
            cache_key = f"{func.__name__}:{str(args)}:{str(kwargs)}"
            
            try:
                # Try to get from cache
                cached_result = cache_instance.check(cache_key)
                if cached_result:
                    logger.info(f"Cache hit for key: {cache_key}")
                    return cached_result
                
                # Execute function and cache result
                result = await func(*args, **kwargs)
                cache_instance.store(cache_key, result)
                logger.info(f"Cache store for key: {cache_key}")
                return result
            except Exception as e:
                logger.error(f"Error during caching operation: {e}")
                # Fallback to direct function call in case of error
                return await func(*args, **kwargs)
        return wrapper
    return decorator

@semantic_cache_decorator(cache)
async def cached_llm_call(prompt: str, llm_client):
    response = await llm_client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content