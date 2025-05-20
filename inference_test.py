import os
import asyncio
import time
import argparse
from clients.base import LLMClient
from clients import PROVIDERS

# Number of concurrent queries to run
NUM_QUERIES = 5
TEST_PROMPTS = [
    "Explain quantum computing in simple terms",
    "Write a short poem about AI",
    "What are the benefits of exercise?",
    "Describe the taste of chocolate",
    "How does a car engine work?",
    "Give me a recipe for banana bread",
    "Explain the concept of blockchain",
    "What are some tips for better sleep?",
    "Tell me about the history of the internet",
    "What causes rainbows to appear?"
]

async def process_completion_stream(client, prompt):
    """Process a completion stream and collect metrics
    
    Returns:
        tuple: (full_content, first_token_time, stream_start_time)
    """
    # Record the time just before making the API call
    stream_start_time = asyncio.get_event_loop().time()
    
    stream = await client.get_completion_stream(prompt)
    
    full_content = ""
    first_token_time = None
    first_token_received = False
    
    async for chunk in stream:
        content_piece = client.extract_content_from_chunk(chunk)
        if content_piece:
            if not first_token_received:
                # Record the time when we receive the first token
                first_token_time = asyncio.get_event_loop().time()
                first_token_received = True
            full_content += content_piece
    
    return full_content, first_token_time, stream_start_time

async def run_query(client, prompt, query_id):
    """Run a single query and collect metrics"""
    print(f"Starting query {query_id}: {prompt[:30]}...")
    
    # Record the time before starting the query
    start_time = asyncio.get_event_loop().time()
    
    try:
        # Process the completion stream and get timing information
        full_content, first_token_time, stream_start_time = await process_completion_stream(client, prompt)
        
        # Record the time after the query completes
        end_time = asyncio.get_event_loop().time()
        
        # Calculate total time for the query
        total_time = end_time - start_time
        
        # Get network latency (round-trip time)
        network_latency = getattr(client, 'network_latency', 0)
        
        # Calculate time to first token (TTFT)
        ttft = first_token_time - stream_start_time if first_token_time else None
        
        # For TTFT, we subtract one network latency since it's dominated by the initial request
        # This gives us the server-side processing time to generate the first token
        adjusted_ttft = ttft - network_latency if ttft and network_latency else ttft
        adjusted_ttft = max(adjusted_ttft, 0.001) if adjusted_ttft else None
        
        # For total time, we subtract one network latency to account for the initial request
        # The streaming latency is already included in the total time and is unavoidable
        adjusted_time = max(total_time - network_latency, 0.001)  # Ensure positive time
        
        if full_content:
            # Calculate tokens and throughput
            tokens = len(full_content.split())
            chars = len(full_content)
            throughput = tokens / total_time if total_time > 0 else 0
            adjusted_throughput = tokens / adjusted_time if adjusted_time > 0 else 0
            
            print(f"✅ Query {query_id} completed: {chars} chars, {tokens} tokens, {total_time:.2f}s (adj: {adjusted_time:.2f}s), {adjusted_throughput:.2f} tokens/sec")
            return query_id, {
                "content": full_content,
                "tokens": tokens,
                "chars": chars,
                "total_time": total_time,
                "adjusted_time": adjusted_time,
                "ttft": ttft,
                "adjusted_ttft": adjusted_ttft,
                "throughput": throughput,
                "adjusted_throughput": adjusted_throughput,
                "network_latency": network_latency
            }
        else:
            print(f"❌ Query {query_id} failed: Empty response")
            return query_id, {
                "content": "Error: Empty response",
                "tokens": 0,
                "chars": 0,
                "total_time": total_time,
                "adjusted_time": adjusted_time,
                "ttft": None,
                "adjusted_ttft": None,
                "throughput": 0,
                "adjusted_throughput": 0,
                "network_latency": network_latency
            }
    except Exception as e:
        end_time = asyncio.get_event_loop().time()
        total_time = end_time - start_time
        network_latency = getattr(client, 'network_latency', 0)
        adjusted_time = max(total_time - network_latency, 0.001)
        
        print(f"❌ Query {query_id} failed: {str(e)}")
        return query_id, {
            "content": f"Error: {str(e)}",
            "tokens": 0,
            "chars": 0,
            "total_time": total_time,
            "adjusted_time": adjusted_time,
            "ttft": None,
            "adjusted_ttft": None,
            "throughput": 0,
            "adjusted_throughput": 0,
            "network_latency": network_latency
        }

async def run_benchmark(client_class, model=None, num_queries=NUM_QUERIES, prompts=None):
    """Run a benchmark test for a given LLM client"""
    # Use default prompts if none provided
    prompts = prompts or TEST_PROMPTS
    
    # Ensure we have enough prompts
    if len(prompts) < num_queries:
        # Repeat prompts if needed
        prompts = prompts * (num_queries // len(prompts) + 1)
    
    # Initialize client
    client = client_class(model=model)
    await client.initialize()
    
    # Prepare results storage
    results = []
    start_time = time.time()
    
    # Create tasks for each query
    tasks = []
    for i in range(num_queries):
        prompt = prompts[i % len(prompts)]
        task = asyncio.create_task(run_query(client, prompt, i))
        tasks.append(task)
    
    # Wait for all tasks to complete
    completed_results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Process results
    for result in completed_results:
        if isinstance(result, Exception):
            # Handle exceptions
            print(f"❌ Query failed: {str(result)}")
            results.append({
                "success": False,
                "error": str(result),
                "tokens": 0,
                "chars": 0,
                "time": 0,
                "ttft": 0,
                "adjusted_time": 0,
                "adjusted_ttft": 0,
                "network_latency": client.network_latency
            })
        else:
            # Extract the query_id and result_dict from the tuple
            _, result_dict = result
            
            # Add success flag to the result dictionary
            result_dict["success"] = result_dict.get("tokens", 0) > 0
            
            # Add the result to our list
            results.append(result_dict)
    
    # Calculate end time
    end_time = time.time()
    total_time = end_time - start_time
    
    # Filter successful queries
    successful_queries = [r for r in results if r.get("tokens", 0) > 0]
    
    # Calculate metrics
    total_tokens = sum(r.get("tokens", 0) for r in successful_queries)
    total_chars = sum(r.get("chars", 0) for r in successful_queries)
    avg_tokens_per_query = total_tokens / len(successful_queries) if successful_queries else 0
    avg_ttft = sum(r.get("ttft", 0) for r in successful_queries if r.get("ttft")) / len(successful_queries) if successful_queries else 0
    avg_adjusted_ttft = sum(r.get("adjusted_ttft", 0) for r in successful_queries if r.get("adjusted_ttft")) / len(successful_queries) if successful_queries else 0
    
    # Get average network latency
    avg_network_latency = sum(r.get("network_latency", 0) for r in successful_queries) / len(successful_queries) if successful_queries else 0
    
    # Calculate throughput metrics
    tokens_per_second = total_tokens / total_time if total_time > 0 else 0
    
    # For adjusted metrics, we need to account for network latency
    # We should adjust the total time directly, not sum individual adjusted times
    total_adjusted_time = max(total_time - avg_network_latency, 0.001)  # Ensure positive time
    adjusted_tokens_per_second = total_tokens / total_adjusted_time if total_adjusted_time > 0 else 0
    
    # Calculate queries per second
    queries_per_second = len(successful_queries) / total_time if total_time > 0 else 0
    adjusted_queries_per_second = len(successful_queries) / total_adjusted_time if total_adjusted_time > 0 else 0
    
    print(f"DEBUG: total_time={total_time:.4f}, avg_network_latency={avg_network_latency:.4f}, total_adjusted_time={total_adjusted_time:.4f}")
    
    # Prepare summary results
    summary = {
        "client": client.name,
        "model": model or "default",
        "total_queries": num_queries,
        "successful_queries": len(successful_queries),
        "total_time": total_time,
        "network_latency": avg_network_latency,
        "total_tokens": total_tokens,
        "total_chars": total_chars,
        "avg_tokens_per_query": avg_tokens_per_query,
        "avg_ttft": avg_ttft,
        "avg_adjusted_ttft": avg_adjusted_ttft,
        "tokens_per_second": tokens_per_second,
        "adjusted_tokens_per_second": adjusted_tokens_per_second,
        "queries_per_second": queries_per_second,
        "adjusted_queries_per_second": adjusted_queries_per_second,
        "results": results
    }
    
    # Print summary
    print("\n" + "="*50)
    print(f"THROUGHPUT METRICS FOR {client.name}")
    print("="*50)
    print(f"Total queries: {num_queries}")
    print(f"Successful queries: {len(successful_queries)}")
    print(f"Total time: {total_time:.2f} seconds")
    print(f"Network latency: {avg_network_latency*1000:.2f} ms")
    print(f"Total tokens generated: {total_tokens}")
    print(f"Total characters: {total_chars}")
    print(f"Average tokens per query: {avg_tokens_per_query:.2f}")
    print(f"Average time to first token: {avg_ttft:.2f} seconds (raw) / {avg_adjusted_ttft:.2f} seconds (adjusted)")
    print(f"Throughput: {tokens_per_second:.2f} tokens/second (raw) / {adjusted_tokens_per_second:.2f} tokens/second (adjusted)")
    print(f"Query rate: {queries_per_second:.2f} queries/second (raw) / {adjusted_queries_per_second:.2f} queries/second (adjusted)")
    print("="*50)
    
    # Print individual query results
    for i, result in enumerate(results):
        if result.get("tokens", 0) > 0:
            print(f"\nQuery {i} result: {result.get('content', '')[:100]}...")
        else:
            print(f"\nQuery {i} result: Error: {result.get('error', 'Unknown error')[:100]}...")
    
    # Save detailed results to file
    safe_name = client.name.replace(" ", "_").replace("/", "_").lower()
    results_file = f"results/{safe_name}_results.txt"
    with open(results_file, "w") as f:
        f.write(f"BENCHMARK RESULTS FOR {client.name}\n")
        f.write("="*50 + "\n\n")
        
        f.write("SUMMARY:\n")
        f.write(f"Total queries: {num_queries}\n")
        f.write(f"Successful queries: {len(successful_queries)}\n")
        f.write(f"Total time: {total_time:.2f} seconds\n")
        f.write(f"Network latency: {avg_network_latency*1000:.2f} ms\n")
        f.write(f"Total tokens generated: {total_tokens}\n")
        f.write(f"Average tokens per query: {avg_tokens_per_query:.2f}\n")
        f.write(f"Throughput: {tokens_per_second:.2f} tokens/second (raw) / {adjusted_tokens_per_second:.2f} tokens/second (adjusted)\n\n")
        
        f.write("DETAILED RESULTS:\n\n")
        for i, result in enumerate(results):
            f.write(f"Query {i}:\n")
            if result.get("tokens", 0) > 0:
                f.write(f"Time: {result.get('total_time', 0):.2f} seconds\n")
                f.write(f"Tokens: {result.get('tokens', 0)}\n")
                f.write(f"Text: {result.get('content', '')}\n\n")
            else:
                f.write(f"Error: {result.get('error', 'Unknown error')}\n\n")
    
    print(f"\nFull results saved to {results_file}")
    
    return summary

async def main():
    parser = argparse.ArgumentParser(description='LLM Throughput Testing')
    parser.add_argument('--provider', type=str, default='fireworks', 
                        choices=list(PROVIDERS.keys()) + ['all'],
                        help='LLM provider to test')
    parser.add_argument('--model', type=str, default=None, 
                        help='Model to use (provider-specific)')
    parser.add_argument('--queries', type=int, default=NUM_QUERIES,
                        help='Number of concurrent queries to run')
    args = parser.parse_args()
    
    if args.provider == 'all':
        # Run benchmarks for all implemented providers
        results = []
        for provider_name, client_factory in PROVIDERS.items():
            print(f"\nRunning benchmark for {provider_name}...")
            result = await run_benchmark(client_factory, model=args.model, num_queries=args.queries)
            results.append(result)
        
        # Compare results
        print("\n" + "="*50)
        print("COMPARISON OF PROVIDERS")
        print("="*50)
        for result in sorted(results, key=lambda x: x['tokens_per_second'], reverse=True):
            print(f"{result['client']}: {result['tokens_per_second']:.2f} tokens/sec, {result['queries_per_second']:.2f} queries/sec")
        print("="*50)
    else:
        # Run benchmark for the specified provider
        if args.provider not in PROVIDERS:
            print(f"Provider {args.provider} not implemented yet")
            return
        
        await run_benchmark(PROVIDERS[args.provider], model=args.model, num_queries=args.queries)

if __name__ == "__main__":
    asyncio.run(main())





