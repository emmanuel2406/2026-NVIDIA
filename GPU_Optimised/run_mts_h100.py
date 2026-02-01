#!/usr/bin/env python3
"""
Runner Script for H100-Optimized MTS

Usage:
    python run_mts_h100.py --N 50 --pop 100 --gen 500
    python run_mts_h100.py --benchmark
    python run_mts_h100.py --compare  # Compare H100 vs standard GPU version
"""

import argparse
import sys
import time
import numpy as np

try:
    import cupy as cp
    GPU_AVAILABLE = True
except ImportError:
    print("ERROR: CuPy not installed. Please install with:")
    print("  pip install cupy-cuda12x")
    sys.exit(1)


def run_quick_test():
    """Run a quick test to verify H100 optimization"""
    print("\n" + "=" * 80)
    print("QUICK TEST - H100 Optimized MTS")
    print("=" * 80)

    from mts_h100_optimized import H100Config, memetic_tabu_search_h100

    try:
        config = H100Config()
        print(f"\n[OK] GPU initialized: {config.device_name}")
        print(f"[OK] Batch tabu size: {config.batch_tabu_size}")
        print(f"[OK] Streams: {config.num_streams}")

        # Run small test
        print("\n" + "-" * 80)
        print("Running test (N=25, Pop=50, Gen=100)...")
        print("-" * 80)

        np.random.seed(42)
        cp.random.seed(42)

        start = time.time()
        best_s, best_energy, _ = memetic_tabu_search_h100(
            N=25,
            population_size=50,
            max_generations=100,
            config=config
        )
        elapsed = time.time() - start

        merit = 25 * 25 / (2.0 * best_energy)

        print(f"\n[OK] Test completed in {elapsed:.2f}s")
        print(f"[OK] Best energy: {best_energy}")
        print(f"[OK] Merit factor: {merit:.4f}")
        print(f"[OK] Throughput: {100/elapsed:.1f} gen/s")

        return True

    except Exception as e:
        print(f"\n[ERROR] Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_benchmark():
    """Run comprehensive benchmark"""
    from mts_h100_optimized import benchmark_h100_mts

    print("\n" + "=" * 80)
    print("H100 MTS COMPREHENSIVE BENCHMARK")
    print("=" * 80)

    results = benchmark_h100_mts(
        N_values=[20, 40, 60, 80, 100],
        population_size=100,
        max_generations=200,
        runs=3
    )

    return results


def run_comparison():
    """Compare H100 optimized vs standard GPU version"""
    print("\n" + "=" * 80)
    print("PERFORMANCE COMPARISON: H100 Optimized vs Standard GPU")
    print("=" * 80)

    from mts_h100_optimized import H100Config, memetic_tabu_search_h100
    from mts_gpu_optimized import GPUConfig, memetic_tabu_search_gpu

    h100_config = H100Config()
    gpu_config = GPUConfig()

    N_values = [20, 40, 60, 80]
    results = []

    for N in N_values:
        print(f"\n{'='*80}")
        print(f"Testing N={N}")
        print(f"{'='*80}")

        pop_size = 100
        max_gen = 100

        # Run H100 version
        np.random.seed(42)
        cp.random.seed(42)

        start = time.time()
        best_s_h100, best_e_h100, _ = memetic_tabu_search_h100(
            N=N,
            population_size=pop_size,
            max_generations=max_gen,
            config=h100_config,
            verbose=False
        )
        h100_time = time.time() - start

        # Run standard GPU version
        np.random.seed(42)
        cp.random.seed(42)

        start = time.time()
        best_s_gpu, best_e_gpu, _ = memetic_tabu_search_gpu(
            N=N,
            population_size=pop_size,
            max_generations=max_gen,
            config=gpu_config
        )
        gpu_time = time.time() - start

        speedup = gpu_time / h100_time

        results.append({
            'N': N,
            'h100_time': h100_time,
            'gpu_time': gpu_time,
            'speedup': speedup,
            'h100_energy': best_e_h100,
            'gpu_energy': best_e_gpu
        })

        print(f"\nN={N} Results:")
        print(f"  H100 Optimized: {h100_time:.3f}s, energy={best_e_h100}")
        print(f"  Standard GPU:   {gpu_time:.3f}s, energy={best_e_gpu}")
        print(f"  Speedup: {speedup:.2f}x")

    # Print summary
    print("\n" + "=" * 80)
    print("COMPARISON SUMMARY")
    print("=" * 80)
    print(f"{'N':>6} {'H100 (s)':>12} {'Std GPU (s)':>12} {'Speedup':>10} {'H100 E':>10} {'GPU E':>10}")
    print("-" * 70)

    for r in results:
        print(f"{r['N']:>6} {r['h100_time']:>12.3f} {r['gpu_time']:>12.3f} "
              f"{r['speedup']:>9.2f}x {r['h100_energy']:>10} {r['gpu_energy']:>10}")

    avg_speedup = np.mean([r['speedup'] for r in results])
    print("-" * 70)
    print(f"{'Average Speedup:':>32} {avg_speedup:.2f}x")

    return results


def run_mts(args):
    """Run MTS with specified parameters"""
    from mts_h100_optimized import H100Config, memetic_tabu_search_h100, sequence_to_bitstring

    print("\n" + "=" * 80)
    print("H100-OPTIMIZED MEMETIC TABU SEARCH")
    print("=" * 80)

    # Set random seed
    if args.seed is not None:
        np.random.seed(args.seed)
        cp.random.seed(args.seed)
        print(f"Random seed: {args.seed}")

    # Configure
    config = H100Config(device_id=args.gpu)

    print(f"\nProblem configuration:")
    print(f"  Sequence length (N): {args.N}")
    print(f"  Population size: {args.pop}")
    print(f"  Max generations: {args.gen}")
    print(f"  Crossover prob: {args.p_combine}")

    if args.target is not None:
        print(f"  Target energy: {args.target}")

    # Run optimization
    best_s, best_energy, population = memetic_tabu_search_h100(
        N=args.N,
        population_size=args.pop,
        max_generations=args.gen,
        p_combine=args.p_combine,
        config=config,
        target_energy=args.target
    )

    # Print results
    merit = args.N * args.N / (2.0 * best_energy)
    bitstring = sequence_to_bitstring(best_s)

    print("\n" + "=" * 80)
    print("FINAL RESULTS")
    print("=" * 80)
    print(f"Best energy: {best_energy}")
    print(f"Merit factor: {merit:.6f}")
    print(f"Best sequence: {bitstring}")

    # Save if requested
    if args.output:
        import json
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

        with open(args.output, 'w') as f:
            json.dump(output_data, f, indent=2)

        print(f"\nResults saved to: {args.output}")


def main():
    parser = argparse.ArgumentParser(
        description='H100-Optimized Memetic Tabu Search for LABS',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick test
  python run_mts_h100.py --quick

  # Run with N=50
  python run_mts_h100.py --N 50 --pop 100 --gen 500

  # Run benchmark
  python run_mts_h100.py --benchmark

  # Compare H100 vs standard GPU
  python run_mts_h100.py --compare
        """
    )

    # Mode selection
    parser.add_argument('--quick', action='store_true',
                        help='Run quick test')
    parser.add_argument('--benchmark', action='store_true',
                        help='Run benchmark')
    parser.add_argument('--compare', action='store_true',
                        help='Compare H100 vs standard GPU')

    # Problem parameters
    parser.add_argument('--N', type=int, default=30,
                        help='Sequence length (default: 30)')
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

    # Output
    parser.add_argument('--output', '-o', type=str,
                        help='Output JSON file for results')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')

    args = parser.parse_args()

    # Execute based on mode
    if args.quick:
        success = run_quick_test()
        sys.exit(0 if success else 1)
    elif args.benchmark:
        run_benchmark()
    elif args.compare:
        run_comparison()
    else:
        run_mts(args)


if __name__ == "__main__":
    main()
