import pickle
import pandas as pd
import streamlit as st
from sentence_transformers import SentenceTransformer
from redisvl.query import VectorQuery
from redisvl.index import SearchIndex
from redis import Redis
import logging
from redisvl.extensions.llmcache import SemanticCache
from functools import wraps

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Redis connection configuration
REDIS_URL = 'redis://localhost:6379/0'
redis_client = Redis.from_url(REDIS_URL)

# Setup semantic cache
vectorizer = SentenceTransformer("distilbert-base-nli-stsb-mean-tokens")
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

@st.cache
def read_data(data="data/misinformation_papers.csv"):
    """Read the data from local."""
    return pd.read_csv(data)

@st.cache(allow_output_mutation=True)
def load_bert_model(name="distilbert-base-nli-stsb-mean-tokens"):
    """Instantiate a sentence-level DistilBERT model."""
    return SentenceTransformer(name)

@st.cache(allow_output_mutation=True)
def load_redis_index(index_name="misinformation_index"):
    """Load the Redis index."""
    try:
        index = SearchIndex(redis_client, index_name)
        logger.info("Redis index loaded successfully.")
        return index
    except Exception as e:
        logger.error(f"Error loading Redis index: {e}")
        raise

def semantic_search(query: str, index: SearchIndex, vectorizer, filters=None):
    """Perform semantic search using Redis."""
    try:
        query_vector = vectorizer.encode(query)
        vector_query = VectorQuery(
            vector=query_vector,
            vector_field_name="text_embedding",
            num_results=10,
            return_fields=["content", "title", "metadata"],
            return_score=True
        )
        if filters:
            vector_query.set_filter(filters)
        results = index.query(vector_query)
        return results
    except Exception as e:
        logger.error(f"Error during semantic search: {e}")
        return []

def main():
    # Load data and models
    data = read_data()
    model = load_bert_model()
    redis_index = load_redis_index()

    st.title("Vector-based searches with Sentence Transformers and Redis")

    # User search
    user_input = st.text_area("Search box", "covid-19 misinformation and social media")

    # Filters
    st.sidebar.markdown("**Filters**")
    filter_year = st.sidebar.slider("Publication year", 2010, 2021, (2010, 2021), 1)
    filter_citations = st.sidebar.slider("Citations", 0, 250, 0)
    num_results = st.sidebar.slider("Number of search results", 10, 50, 10)

    # Fetch results
    if user_input:
        filters = {
            "year": {"$gte": filter_year[0], "$lte": filter_year[1]},
            "citations": {"$gte": filter_citations}
        }
        results = semantic_search(user_input, redis_index, model, filters)
        for result in results:
            st.write(
                f"""**{result['title']}**  
                **Citations**: {result['metadata']['citations']}  
                **Publication year**: {result['metadata']['year']}  
                **Abstract**
                {result['content']}
                """
            )

if __name__ == "__main__":
    main()