import numpy as np
import logging
from redisvl.query import VectorQuery
from redisvl.index import SearchIndex
from redisvl.vectorizer import Vectorizer
from redis import Redis
from redisvl.extensions.llmcache import SemanticCache
from functools import wraps

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Redis connection configuration
REDIS_URL = 'redis://localhost:6379/0'
redis_client = Redis.from_url(REDIS_URL)
index = SearchIndex(redis_client, index_name='vector_index')
vectorizer = Vectorizer()

# Setup semantic cache
cache = SemanticCache(
    name="llm_cache",
    vectorizer=vectorizer,
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
            
            # Try to get from cache
            cached_result = cache_instance.check(cache_key)
            if cached_result:
                return cached_result
            
            # Execute function and cache result
            result = await func(*args, **kwargs)
            cache_instance.store(cache_key, result)
            return result
        return wrapper
    return decorator

@semantic_cache_decorator(cache)
async def cached_llm_call(prompt: str, llm_client):
    response = await llm_client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

def vector_search(query, model, index, num_results=10):
    """Transforms query to vector using a pretrained, sentence-level 
    DistilBERT model and finds similar vectors using Redis vector similarity.
    Args:
        query (str): User query that should be more than a sentence long.
        model (sentence_transformers.SentenceTransformer.SentenceTransformer)
        index (`SearchIndex`): Redis SearchIndex for vector operations.
        num_results (int): Number of results to return.
    Returns:
        results (list): List of search results with content, title, and metadata.
    """
    try:
        # Generate query embedding
        vector = model.encode(list(query))
        
        # Create vector query for Redis
        vector_query = VectorQuery(
            vector=vector,
            vector_field_name="text_embedding",
            num_results=num_results,
            return_fields=["content", "title", "metadata"],
            return_score=True
        )
        
        # Execute search using Redis
        results = index.query(vector_query)
        return results
    except Exception as e:
        logger.error(f"Error during vector search: {e}")
        return []

def id2details(df, I, column):
    """Returns the paper titles based on the paper index."""
    return [list(df[df.id == idx][column]) for idx in I[0]]