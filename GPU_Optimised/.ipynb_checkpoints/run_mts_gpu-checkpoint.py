#!/usr/bin/env python3
"""
Simple Runner Script for GPU-Accelerated MTS

Usage:
    python run_mts_gpu.py --N 20 --pop 100 --gen 500
    python run_mts_gpu.py --N 50 --pop 200 --gen 1000 --benchmark
    python run_mts_gpu.py --quick  # Quick test
"""

import argparse
import sys
import numpy as np

try:
    import cupy as cp
    GPU_AVAILABLE = True
except ImportError:
    print("ERROR: CuPy not installed. Please install with:")
    print("  pip install cupy-cuda12x --break-system-packages")
    print("Or for CUDA 11.x:")
    print("  pip install cupy-cuda11x --break-system-packages")
    sys.exit(1)

from mts_gpu_optimized import (
    memetic_tabu_search_gpu,
    GPUConfig
)


def run_quick_test():
    """Run a quick test to verify installation"""
    print("\n" + "="*80)
    print("QUICK TEST - Verifying GPU Installation")
    print("="*80)
    
    # Check GPU
    try:
        props = cp.cuda.runtime.getDeviceProperties(0)
        gpu_name = props['name'].decode('utf-8')
        print(f"\n✓ GPU detected: {gpu_name}")
        
        free, total = cp.cuda.Device().mem_info
        print(f"✓ GPU memory: {free/1e9:.1f} GB free / {total/1e9:.1f} GB total")
    except Exception as e:
        print(f"✗ GPU error: {e}")
        return False
    
    # Run small test
    print("\n" + "-"*80)
    print("Running small test (N=20, Pop=20, Gen=20)...")
    print("-"*80)
    
    try:
        config = GPUConfig()
        best_s, best_energy, population = memetic_tabu_search_gpu(
            N=20,
            population_size=20,
            max_generations=20,
            config=config
        )
        
        merit = 20 * 20 / (2.0 * best_energy)
        
        print(f"\n✓ Test completed successfully!")
        print(f"  Best energy: {best_energy}")
        print(f"  Merit factor: {merit:.4f}")
        
        return True
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_mts(args):
    """Run MTS with specified parameters"""
    print("\n" + "="*80)
    print("GPU-ACCELERATED MEMETIC TABU SEARCH")
    print("="*80)
    
    # Set random seed
    if args.seed is not None:
        np.random.seed(args.seed)
        cp.random.seed(args.seed)
        print(f"Random seed: {args.seed}")
    
    # Configure GPU
    config = GPUConfig(
        device_id=args.gpu,
        num_streams=args.streams,
        block_size=args.block_size
    )
    
    print(f"\nProblem configuration:")
    print(f"  Sequence length (N): {args.N}")
    print(f"  Population size: {args.pop}")
    print(f"  Max generations: {args.gen}")
    print(f"  Crossover prob: {args.p_combine}")
    
    if args.target is not None:
        print(f"  Target energy: {args.target}")
    
    # Run optimization
    best_s, best_energy, population = memetic_tabu_search_gpu(
        N=args.N,
        population_size=args.pop,
        max_generations=args.gen,
        p_combine=args.p_combine,
        config=config,
        target_energy=args.target
    )
    
    # Print results
    merit = args.N * args.N / (2.0 * best_energy)
    bitstring = ''.join(['0' if x == 1 else '1' for x in best_s])
    
    print("\n" + "="*80)
    print("FINAL RESULTS")
    print("="*80)
    print(f"Best energy: {best_energy}")
    print(f"Merit factor: {merit:.6f}")
    print(f"Best sequence: {bitstring}")
    
    # Save if requested
    if args.output:
        output_data = {
            'N': args.N,
            'best_energy': int(best_energy),
            'merit_factor': float(merit),
            'sequence': bitstring,
            'parameters': {
                'population_size': args.pop,
                'max_generations': args.gen,
                'p_combine': args.p_combine
            }
        }
        
        import json
        with open(args.output, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"\nResults saved to: {args.output}")


def run_benchmark(args):
    """Run benchmark comparison"""
    print("\n" + "="*80)
    print("BENCHMARK MODE")
    print("="*80)
    
    from benchmark_comparison import PerformanceProfiler
    
    profiler = PerformanceProfiler()
    
    N_values = args.N_values or [20, 40, 60, 80, 100]
    print(f"\nBenchmarking N values: {N_values}")
    print(f"Runs per N: {args.runs}")
    
    profiler.benchmark_scaling(N_values=N_values, runs=args.runs)
    
    # Generate report
    profiler.print_summary()
    profiler.plot_results("benchmark_results.png")
    profiler.save_results("benchmark_results.json")
    
    print("\n✓ Benchmark complete!")
    print("  - Plot saved to: benchmark_results.png")
    print("  - Data saved to: benchmark_results.json")


def main():
    parser = argparse.ArgumentParser(
        description='GPU-Accelerated Memetic Tabu Search for LABS',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick test
  python run_mts_gpu.py --quick
  
  # Run with N=50
  python run_mts_gpu.py --N 50 --pop 100 --gen 500
  
  # Run with target energy
  python run_mts_gpu.py --N 20 --pop 100 --gen 1000 --target 50
  
  # Benchmark
  python run_mts_gpu.py --benchmark --N-values 20 40 60 --runs 3
  
  # Use specific GPU
  python run_mts_gpu.py --N 100 --gpu 1
        """
    )
    
    # Mode selection
    parser.add_argument('--quick', action='store_true',
                       help='Run quick test to verify installation')
    parser.add_argument('--benchmark', action='store_true',
                       help='Run benchmark comparison')
    
    # Problem parameters
    parser.add_argument('--N', type=int, default=20,
                       help='Sequence length (default: 20)')
    parser.add_argument('--pop', type=int, default=100,
                       help='Population size (default: 100)')
    parser.add_argument('--gen', type=int, default=500,
                       help='Max generations (default: 500)')
    parser.add_argument('--p-combine', type=float, default=0.9,
                       help='Crossover probability (default: 0.9)')
    parser.add_argument('--target', type=int, default=None,
                       help='Target energy for early stopping')
    
    # GPU configuration
    parser.add_argument('--gpu', type=int, default=0,
                       help='GPU device ID (default: 0)')
    parser.add_argument('--streams', type=int, default=4,
                       help='Number of CUDA streams (default: 4)')
    parser.add_argument('--block-size', type=int, default=256,
                       help='CUDA block size (default: 256)')
    
    # Benchmark parameters
    parser.add_argument('--N-values', type=int, nargs='+',
                       help='N values for benchmark (e.g., 20 40 60)')
    parser.add_argument('--runs', type=int, default=3,
                       help='Benchmark runs per N (default: 3)')
    
    # Output
    parser.add_argument('--output', '-o', type=str,
                       help='Output JSON file for results')
    parser.add_argument('--seed', type=int,
                       help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    # Execute based on mode
    if args.quick:
        success = run_quick_test()
        sys.exit(0 if success else 1)
    elif args.benchmark:
        run_benchmark(args)
    else:
        run_mts(args)


if __name__ == "__main__":
    main()
