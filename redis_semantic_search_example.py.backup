
# Redis Semantic Search Integration Example


# Semantic Search Implementation
from redisvl.query import VectorQuery, RangeQuery
from redisvl.query.filter import Text, Num

def semantic_search(query: str, index: SearchIndex, vectorizer, filters=None):
    # Generate query embedding
    query_vector = vectorizer.embed(query)
    
    # Create vector query
    vector_query = VectorQuery(
        vector=query_vector,
        vector_field_name="text_embedding",
        num_results=10,
        return_fields=["content", "title", "metadata"],
        return_score=True
    )
    
    # Add filters if provided
    if filters:
        vector_query.set_filter(filters)
    
    # Execute search
    results = index.query(vector_query)
    return results

# Range-based similarity search
def similarity_range_search(query: str, index: SearchIndex, vectorizer, distance_threshold=0.8):
    query_vector = vectorizer.embed(query)
    
    range_query = RangeQuery(
        vector=query_vector,
        vector_field_name="text_embedding",
        distance_threshold=distance_threshold,
        return_fields=["content", "title"],
        return_score=True
    )
    
    return index.query(range_query)


# Example usage:
if __name__ == "__main__":
    # This is an example of how to use the Redis semantic_search integration
    # Modify this code according to your specific use case
    
    print("Redis semantic_search integration example")
    print("Please refer to the documentation for detailed usage instructions")
