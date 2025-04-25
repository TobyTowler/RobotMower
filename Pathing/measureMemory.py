import subprocess
import psutil
import time
import matplotlib.pyplot as plt
import numpy as np
import statistics

def run_with_memory_monitoring(script_path, parameter, interval=0.01):
    """
    Run a Python script with a parameter and monitor its memory usage.
    
    Args:
        script_path: Path to the Python script
        parameter: Parameter to pass to the script
        interval: Sampling interval in seconds
    
    Returns:
        Dictionary with max and average memory usage
    """
    # Start the script as a subprocess
    process = subprocess.Popen(['python', script_path, str(parameter)], 
                              stdout=subprocess.PIPE, 
                              stderr=subprocess.PIPE)
    
    # Monitor memory
    memory_samples = []
    p = psutil.Process(process.pid)
    
    try:
        while process.poll() is None:  # While the process is still running
            mem = p.memory_info().rss / (1024 * 1024)  # MB
            memory_samples.append(mem)
            time.sleep(interval)
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        pass
    
    # Make sure process is terminated
    if process.poll() is None:
        process.terminate()
        process.wait()
    
    # Get stdout and stderr
    stdout, stderr = process.communicate()
    
    if not memory_samples:
        return {"max": 0, "avg": 0, "samples": 0}
    
    return {
        "max": max(memory_samples),
        "avg": statistics.mean(memory_samples),
        "samples": len(memory_samples)
    }

def main():
    # Script to run (replace with your script path)
    script_path = "your_script.py"
    
    # Parameters to test
    parameters = [100, 200, 300, 400, 500, 600, 1000]
    
    # Store results
    max_memory = []
    avg_memory = []
    
    # Run for each parameter
    for param in parameters:
        print(f"Running with parameter: {param}")
        
        # Run multiple times for each parameter value
        param_max = []
        param_avg = []
        runs = 5  # Number of runs per parameter
        
        for i in range(runs):
            print(f"  Run {i+1}/{runs}...")
            result = run_with_memory_monitoring(script_path, param)
            param_max.append(result["max"])
            param_avg.append(result["avg"])
            print(f"    Max: {result['max']:.2f} MB, Avg: {result['avg']:.2f} MB")
        
        # Average across runs
        max_memory.append(statistics.mean(param_max))
        avg_memory.append(statistics.mean(param_avg))
    
    # Create the plot
    plt.figure(figsize=(10, 6))
    
    plt.plot(parameters, max_memory, 'ro-', label='Maximum Memory')
    plt.plot(parameters, avg_memory, 'bo-', label='Average Memory')
    
    plt.xlabel('Parameter Value')
    plt.ylabel('Memory Usage (MB)')
    plt.title('Memory Usage vs Parameter Value')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    # Add data labels
    for i, txt in enumerate(max_memory):
        plt.annotate(f"{txt:.1f}", (parameters[i], max_memory[i]), 
                    textcoords="offset points", xytext=(0,10), ha='center')
    
    for i, txt in enumerate(avg_memory):
        plt.annotate(f"{txt:.1f}", (parameters[i], avg_memory[i]),
                    textcoords="offset points", xytext=(0,-15), ha='center')
    
    plt.tight_layout()
    plt.savefig('memory_usage_plot.png')
    plt.show()

if __name__ == "__main__":
    main()   plt.show()
