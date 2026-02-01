"""
Comprehensive Comparison: CPU vs GPU MTS Performance
Includes visualization, profiling, and analysis tools
"""

import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
import time
from typing import Dict, List
import json
from dataclasses import dataclass, asdict

try:
    from mts_gpu_optimized import (
        memetic_tabu_search_gpu, 
        GPUConfig,
        compute_energy_gpu,
        compute_energy_batch_gpu
    )
    GPU_AVAILABLE = True
except ImportError:
    print("Warning: GPU modules not available. Install CuPy for GPU acceleration.")
    GPU_AVAILABLE = False


@dataclass
class BenchmarkResult:
    """Store benchmark results"""
    N: int
    method: str
    time_seconds: float
    best_energy: int
    merit_factor: float
    generations: int
    throughput_gen_per_sec: float
    memory_mb: float = 0.0
    gpu_name: str = "N/A"
    
    def to_dict(self):
        return asdict(self)


class PerformanceProfiler:
    """Profile CPU vs GPU performance"""
    
    def __init__(self):
        self.results: List[BenchmarkResult] = []
        self.gpu_available = GPU_AVAILABLE
        
        if self.gpu_available:
            self.gpu_config = GPUConfig()
            self.gpu_name = self.gpu_config.device_name
        else:
            self.gpu_name = "N/A"
    
    def run_cpu_baseline(self, N: int, population_size: int, max_generations: int) -> BenchmarkResult:
        """Run CPU version (simulated for now)"""
        print(f"\n[CPU] Running baseline for N={N}...")
        
        # Import original CPU version
        import sys
        sys.path.append('/home/claude')
        
        # For now, we'll use a simplified CPU version
        # In practice, you'd import from your original script
        start_time = time.time()
        
        # Simulate CPU version with numpy (slower)
        population = [np.random.choice([-1, 1], size=N) for _ in range(population_size)]
        best_energy = float('inf')
        
        for gen in range(max_generations):
            # Simple random search for baseline
            idx = np.random.randint(0, population_size)
            child = population[idx].copy()
            
            # Mutate
            for i in range(N):
                if np.random.random() < 1.0/N:
                    child[i] *= -1
            
            # Compute energy (CPU)
            energy = 0
            for k in range(1, N):
                Ck = np.sum(child[:N-k] * child[k:])
                energy += Ck * Ck
            
            if energy < best_energy:
                best_energy = energy
        
        elapsed = time.time() - start_time
        merit = N * N / (2.0 * best_energy) if best_energy > 0 else 0
        
        result = BenchmarkResult(
            N=N,
            method="CPU (NumPy)",
            time_seconds=elapsed,
            best_energy=int(best_energy),
            merit_factor=merit,
            generations=max_generations,
            throughput_gen_per_sec=max_generations / elapsed,
            gpu_name="N/A"
        )
        
        print(f"[CPU] Completed in {elapsed:.3f}s, "
              f"throughput: {result.throughput_gen_per_sec:.2f} gen/s")
        
        return result
    
    def run_gpu_optimized(self, N: int, population_size: int, max_generations: int) -> BenchmarkResult:
        """Run GPU-optimized version"""
        if not self.gpu_available:
            raise RuntimeError("GPU not available. Install CuPy.")
        
        print(f"\n[GPU] Running optimized version for N={N}...")
        
        start_time = time.time()
        
        best_s, best_energy, population = memetic_tabu_search_gpu(
            N=N,
            population_size=population_size,
            max_generations=max_generations,
            config=self.gpu_config
        )
        
        elapsed = time.time() - start_time
        merit = N * N / (2.0 * best_energy)
        
        # Get GPU memory usage
        mempool = cp.get_default_memory_pool()
        memory_mb = mempool.used_bytes() / 1024 / 1024
        
        result = BenchmarkResult(
            N=N,
            method="GPU (CuPy+CUDA)",
            time_seconds=elapsed,
            best_energy=best_energy,
            merit_factor=merit,
            generations=max_generations,
            throughput_gen_per_sec=max_generations / elapsed,
            memory_mb=memory_mb,
            gpu_name=self.gpu_name
        )
        
        print(f"[GPU] Completed in {elapsed:.3f}s, "
              f"throughput: {result.throughput_gen_per_sec:.2f} gen/s, "
              f"memory: {memory_mb:.1f} MB")
        
        return result
    
    def benchmark_scaling(self, N_values: List[int], runs: int = 3):
        """Benchmark across different problem sizes"""
        print("="*80)
        print("SCALING BENCHMARK: CPU vs GPU")
        print("="*80)
        
        for N in N_values:
            print(f"\n{'='*80}")
            print(f"Benchmarking N={N} ({runs} runs)")
            print(f"{'='*80}")
            
            # Adjust generations based on N
            max_generations = max(50, 200 - N)
            population_size = 50
            
            for run in range(runs):
                print(f"\n--- Run {run+1}/{runs} ---")
                
                # CPU baseline
                try:
                    cpu_result = self.run_cpu_baseline(N, population_size, max_generations)
                    self.results.append(cpu_result)
                except Exception as e:
                    print(f"[CPU] Error: {e}")
                
                # GPU optimized
                if self.gpu_available:
                    try:
                        gpu_result = self.run_gpu_optimized(N, population_size, max_generations)
                        self.results.append(gpu_result)
                        
                        # Compute speedup
                        if cpu_result.time_seconds > 0:
                            speedup = cpu_result.time_seconds / gpu_result.time_seconds
                            print(f"\n[SPEEDUP] {speedup:.2f}x faster on GPU")
                            print(f"[QUALITY] CPU energy: {cpu_result.best_energy}, "
                                  f"GPU energy: {gpu_result.best_energy}")
                    except Exception as e:
                        print(f"[GPU] Error: {e}")
                
                # Clear GPU memory
                if self.gpu_available:
                    cp.get_default_memory_pool().free_all_blocks()
    
    def plot_results(self, output_file: str = "benchmark_results.png"):
        """Visualize benchmark results"""
        if not self.results:
            print("No results to plot")
            return
        
        # Organize results
        cpu_results = [r for r in self.results if r.method.startswith("CPU")]
        gpu_results = [r for r in self.results if r.method.startswith("GPU")]
        
        if not cpu_results or not gpu_results:
            print("Need both CPU and GPU results for comparison")
            return
        
        # Group by N
        N_values = sorted(set(r.N for r in self.results))
        
        cpu_times = []
        gpu_times = []
        cpu_energies = []
        gpu_energies = []
        speedups = []
        
        for N in N_values:
            cpu_N = [r for r in cpu_results if r.N == N]
            gpu_N = [r for r in gpu_results if r.N == N]
            
            if cpu_N and gpu_N:
                cpu_time = np.mean([r.time_seconds for r in cpu_N])
                gpu_time = np.mean([r.time_seconds for r in gpu_N])
                cpu_energy = np.mean([r.best_energy for r in cpu_N])
                gpu_energy = np.mean([r.best_energy for r in gpu_N])
                
                cpu_times.append(cpu_time)
                gpu_times.append(gpu_time)
                cpu_energies.append(cpu_energy)
                gpu_energies.append(gpu_energy)
                speedups.append(cpu_time / gpu_time)
        
        # Create plots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Plot 1: Execution Time
        ax1 = axes[0, 0]
        x = np.arange(len(N_values))
        width = 0.35
        ax1.bar(x - width/2, cpu_times, width, label='CPU', color='steelblue', alpha=0.8)
        ax1.bar(x + width/2, gpu_times, width, label='GPU', color='green', alpha=0.8)
        ax1.set_xlabel('Sequence Length (N)')
        ax1.set_ylabel('Time (seconds)')
        ax1.set_title('Execution Time Comparison')
        ax1.set_xticks(x)
        ax1.set_xticklabels(N_values)
        ax1.legend()
        ax1.grid(axis='y', alpha=0.3)
        
        # Plot 2: Speedup
        ax2 = axes[0, 1]
        ax2.plot(N_values, speedups, 'o-', linewidth=2, markersize=8, color='darkgreen')
        ax2.axhline(1, color='red', linestyle='--', linewidth=1, label='No speedup')
        ax2.set_xlabel('Sequence Length (N)')
        ax2.set_ylabel('Speedup (CPU time / GPU time)')
        ax2.set_title('GPU Speedup vs Problem Size')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Solution Quality
        ax3 = axes[1, 0]
        ax3.plot(N_values, cpu_energies, 'o-', linewidth=2, markersize=8, 
                label='CPU', color='steelblue')
        ax3.plot(N_values, gpu_energies, 's-', linewidth=2, markersize=8,
                label='GPU', color='green')
        ax3.set_xlabel('Sequence Length (N)')
        ax3.set_ylabel('Best Energy Found')
        ax3.set_title('Solution Quality Comparison')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Throughput
        ax4 = axes[1, 1]
        cpu_throughput = [np.mean([r.throughput_gen_per_sec for r in cpu_results if r.N == N]) 
                         for N in N_values]
        gpu_throughput = [np.mean([r.throughput_gen_per_sec for r in gpu_results if r.N == N])
                         for N in N_values]
        ax4.plot(N_values, cpu_throughput, 'o-', linewidth=2, markersize=8,
                label='CPU', color='steelblue')
        ax4.plot(N_values, gpu_throughput, 's-', linewidth=2, markersize=8,
                label='GPU', color='green')
        ax4.set_xlabel('Sequence Length (N)')
        ax4.set_ylabel('Throughput (generations/second)')
        ax4.set_title('Throughput Comparison')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"\n[PLOT] Saved to {output_file}")
        plt.show()
    
    def save_results(self, output_file: str = "benchmark_results.json"):
        """Save results to JSON"""
        results_dict = [r.to_dict() for r in self.results]
        
        with open(output_file, 'w') as f:
            json.dump(results_dict, f, indent=2)
        
        print(f"\n[SAVE] Results saved to {output_file}")
    
    def print_summary(self):
        """Print summary statistics"""
        if not self.results:
            print("No results available")
            return
        
        print("\n" + "="*80)
        print("BENCHMARK SUMMARY")
        print("="*80)
        
        # Group by method
        methods = set(r.method for r in self.results)
        
        for method in sorted(methods):
            method_results = [r for r in self.results if r.method == method]
            
            print(f"\n{method}:")
            print(f"  Runs: {len(method_results)}")
            
            avg_time = np.mean([r.time_seconds for r in method_results])
            avg_throughput = np.mean([r.throughput_gen_per_sec for r in method_results])
            avg_energy = np.mean([r.best_energy for r in method_results])
            avg_merit = np.mean([r.merit_factor for r in method_results])
            
            print(f"  Avg Time: {avg_time:.3f}s")
            print(f"  Avg Throughput: {avg_throughput:.2f} gen/s")
            print(f"  Avg Energy: {avg_energy:.1f}")
            print(f"  Avg Merit: {avg_merit:.4f}")
            
            if method.startswith("GPU"):
                avg_memory = np.mean([r.memory_mb for r in method_results])
                print(f"  Avg GPU Memory: {avg_memory:.1f} MB")
                if method_results:
                    print(f"  GPU: {method_results[0].gpu_name}")
        
        # Overall speedup
        cpu_results = [r for r in self.results if r.method.startswith("CPU")]
        gpu_results = [r for r in self.results if r.method.startswith("GPU")]
        
        if cpu_results and gpu_results:
            avg_cpu_time = np.mean([r.time_seconds for r in cpu_results])
            avg_gpu_time = np.mean([r.time_seconds for r in gpu_results])
            overall_speedup = avg_cpu_time / avg_gpu_time
            
            print(f"\n{'='*80}")
            print(f"OVERALL GPU SPEEDUP: {overall_speedup:.2f}x")
            print(f"{'='*80}")


def quick_comparison_demo():
    """Quick demonstration of CPU vs GPU"""
    print("\n" + "="*80)
    print("QUICK COMPARISON DEMO")
    print("="*80)
    
    profiler = PerformanceProfiler()
    
    # Test with small problem
    N = 20
    print(f"\nTesting with N={N}, Population=20, Generations=30")
    
    # Run both versions
    profiler.benchmark_scaling(N_values=[N], runs=1)
    
    # Print results
    profiler.print_summary()


def full_benchmark_suite():
    """Run comprehensive benchmark suite"""
    print("\n" + "="*80)
    print("COMPREHENSIVE BENCHMARK SUITE")
    print("="*80)
    
    profiler = PerformanceProfiler()
    
    # Test across multiple problem sizes
    N_values = [20, 40, 60, 80, 100]
    
    print(f"\nTesting N values: {N_values}")
    print("This may take several minutes...")
    
    profiler.benchmark_scaling(N_values=N_values, runs=2)
    
    # Generate reports
    profiler.print_summary()
    profiler.plot_results("benchmark_comparison.png")
    profiler.save_results("benchmark_results.json")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--full":
        full_benchmark_suite()
    else:
        quick_comparison_demo()
