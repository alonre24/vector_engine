import numpy as np
import logging
from redisvl.query import VectorQuery
from redisvl.index import SearchIndex
from redisvl.vectorizer import Vectorizer
from redis import Redis

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Redis connection configuration
redis_client = Redis(host='localhost', port=6379, db=0)
index = SearchIndex(redis_client, index_name='vector_index')
vectorizer = Vectorizer()

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