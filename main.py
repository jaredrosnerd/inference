#!/usr/bin/env python3
import os
import asyncio
import argparse
import json
import csv
from datetime import datetime
from inference_test import run_benchmark, NUM_QUERIES, TEST_PROMPTS
from clients import PROVIDERS
from clients.models import (
    MODELS, DEFAULT_MODELS, 
    list_models_for_provider, get_model_id, 
    list_all_friendly_models, get_default_model
)

def save_results_to_csv(results, filename="results/benchmark_results.csv"):
    """Save benchmark results to a CSV file
    
    Args:
        results: List of benchmark result dictionaries
        filename: Path to the CSV file
    """
    # Define the CSV columns
    fieldnames = [
        'timestamp', 
        'client', 
        'model', 
        'total_queries', 
        'successful_queries', 
        'total_time', 
        'network_latency_ms',  # Added network latency in ms
        'total_tokens', 
        'total_chars', 
        'avg_tokens_per_query',
        'avg_ttft',  # Raw time to first token
        'avg_adjusted_ttft',  # Adjusted time to first token
        'tokens_per_second',  # Raw throughput
        'adjusted_tokens_per_second',  # Adjusted throughput
        'queries_per_second',  # Raw query rate
        'adjusted_queries_per_second'  # Adjusted query rate
    ]
    
    # Check if file exists to determine if we need to write headers
    file_exists = os.path.isfile(filename)
    
    # Get current timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    with open(filename, mode='a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        # Write header if file doesn't exist
        if not file_exists:
            writer.writeheader()
        
        # Write each result as a row
        for result in results:
            # Create a new dict with just the fields we want
            row = {
                'timestamp': timestamp,
                'client': result['client'],
                'model': result['model'],
                'total_queries': result['total_queries'],
                'successful_queries': result['successful_queries'],
                'total_time': result['total_time'],
                'network_latency_ms': result['network_latency'] * 1000,  # Convert to ms
                'total_tokens': result['total_tokens'],
                'total_chars': result['total_chars'],
                'avg_tokens_per_query': result['avg_tokens_per_query'],
                'avg_ttft': result['avg_ttft'],
                'avg_adjusted_ttft': result['avg_adjusted_ttft'],
                'tokens_per_second': result['tokens_per_second'],
                'adjusted_tokens_per_second': result['adjusted_tokens_per_second'],
                'queries_per_second': result['queries_per_second'],
                'adjusted_queries_per_second': result['adjusted_queries_per_second']
            }
            writer.writerow(row)
    
    print(f"Results appended to CSV file: {filename}")

async def run_multi_model_benchmark(client_class, models, num_queries=NUM_QUERIES):
    """Run benchmarks for multiple models of the same provider"""
    results = []
    
    for model in models:
        print(f"\nRunning benchmark for {client_class.__name__} with model {model}...")
        try:
            result = await run_benchmark(client_class, model=model, num_queries=num_queries)
            results.append(result)
        except Exception as e:
            print(f"Error running benchmark for {model}: {str(e)}")
    
    return results

async def main_async():
    parser = argparse.ArgumentParser(description='LLM Throughput Testing')
    parser.add_argument('--provider', type=str, default='fireworks', 
                        choices=list(PROVIDERS.keys()) + ['all'],
                        help='LLM provider to test')
    parser.add_argument('--model', type=str, default=None, 
                        help='Model to use (friendly name or provider-specific ID)')
    parser.add_argument('--models', action='store_true',
                        help='Test all available models for the provider')
    parser.add_argument('--list_models', action='store_true',
                        help='List available models for the specified provider')
    parser.add_argument('--queries', type=int, default=NUM_QUERIES,
                        help='Number of concurrent queries to run')
    parser.add_argument('--output', type=str, default=None,
                        help='Output JSON file for results')
    parser.add_argument('--csv', type=str, default="results/benchmark_results.csv",
                        help='CSV file to append results to')
    parser.add_argument('--region', type=str, default='us-west-2',
                        help='AWS region for Bedrock (default: us-west-2)')
    args = parser.parse_args()
    
    # Create results directory if it doesn't exist
    os.makedirs('results', exist_ok=True)
    
    # List available models if requested
    if args.list_models:
        print(f"Available models for {args.provider}:")
        if args.provider == 'all':
            for provider in PROVIDERS.keys():
                print(f"\n{provider.upper()} MODELS:")
                for model_name in list_models_for_provider(provider):
                    model_id = get_model_id(model_name, provider)
                    print(f"  - {model_name}: {model_id}")
        else:
            for model_name in list_models_for_provider(args.provider):
                model_id = get_model_id(model_name, args.provider)
                print(f"  - {model_name}: {model_id}")
        return
    
    # If no model specified, use the default for the provider
    if not args.model and not args.models:
        args.model = get_default_model(args.provider)
        print(f"No model specified, using default: {args.model}")
    
    all_results = []
    
    if args.provider == 'all':
        # Run benchmarks for all implemented providers
        for provider_name, client_class in PROVIDERS.items():
            if args.models:
                # Test all available models for this provider
                provider_models = [get_model_id(model, provider_name) for model in list_models_for_provider(provider_name)]
                results = await run_multi_model_benchmark(client_class, provider_models, args.queries)
                all_results.extend(results)
            else:
                # Test with default or specified model
                model_name = args.model or get_default_model(provider_name)
                try:
                    model_id = get_model_id(model_name, provider_name)
                    print(f"\nRunning benchmark for {provider_name} with model {model_name} ({model_id})...")
                    result = await run_benchmark(client_class, model=model_id, num_queries=args.queries)
                    all_results.append(result)
                except ValueError:
                    print(f"Model {model_name} not available for provider {provider_name}, skipping...")
    else:
        # Run benchmark for the specified provider
        if args.provider not in PROVIDERS:
            print(f"Provider {args.provider} not implemented yet")
            return
        
        client_class = PROVIDERS[args.provider]
        
        if args.models:
            # Test all available models for this provider
            provider_models = [get_model_id(model, args.provider) for model in list_models_for_provider(args.provider)]
            results = await run_multi_model_benchmark(client_class, provider_models, args.queries)
            all_results.extend(results)
        else:
            # Test with specified or default model
            model_name = args.model or get_default_model(args.provider)
            try:
                model_id = get_model_id(model_name, args.provider)
                
                if args.provider == 'bedrock':
                    # Pass the region to the Bedrock client
                    result = await run_benchmark(
                        lambda **kwargs: PROVIDERS['bedrock'](region=args.region, **kwargs),
                        model=model_id,
                        num_queries=args.queries
                    )
                else:
                    # For other providers, use the standard approach
                    result = await run_benchmark(client_class, model=model_id, num_queries=args.queries)
                
                all_results.append(result)
            except ValueError:
                # If the model name isn't in our mapping, try using it directly
                print(f"Model {model_name} not found in mappings, trying as direct model ID...")
                if args.provider == 'bedrock':
                    # Pass the region to the Bedrock client
                    result = await run_benchmark(
                        lambda **kwargs: PROVIDERS['bedrock'](region=args.region, **kwargs),
                        model=model_name,
                        num_queries=args.queries
                    )
                else:
                    # For other providers, use the standard approach
                    result = await run_benchmark(client_class, model=model_name, num_queries=args.queries)
                
                all_results.append(result)
    
    # Compare results
    if len(all_results) > 1:
        print("\n" + "="*50)
        print("COMPARISON OF MODELS/PROVIDERS")
        print("="*50)
        for result in sorted(all_results, key=lambda x: x['adjusted_tokens_per_second'], reverse=True):
            print(f"{result['client']}: {result['tokens_per_second']:.2f} tokens/sec (raw) / {result['adjusted_tokens_per_second']:.2f} tokens/sec (adjusted)")
        print("="*50)
    
    # Save results to CSV
    save_results_to_csv(all_results, args.csv)
    
    # Save all results to JSON if requested
    if args.output or len(all_results) > 1:
        output_file = args.output or f"results/benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        # Clean results for JSON serialization
        json_results = []
        for result in all_results:
            # Create a copy without the detailed results
            json_result = result.copy()
            if 'results' in json_result:
                del json_result['results']
            json_results.append(json_result)
        
        # Add timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        output_data = {
            "timestamp": timestamp,
            "results": json_results
        }
        
        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2)
        print(f"\nBenchmark results saved to {output_file}")

def main():
    asyncio.run(main_async())

if __name__ == "__main__":
    main()

