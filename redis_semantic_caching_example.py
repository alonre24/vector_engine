
# Redis Semantic Caching Integration Example


# Semantic Caching with RedisVL
from redisvl.extensions.llmcache import SemanticCache
from functools import wraps

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


# Example usage:
if __name__ == "__main__":
    # This is an example of how to use the Redis semantic_caching integration
    # Modify this code according to your specific use case
    
    print("Redis semantic_caching integration example")
    print("Please refer to the documentation for detailed usage instructions")
