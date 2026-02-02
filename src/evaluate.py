import matplotlib.pyplot as plt

def compare_results():
    # Load training times from file
    with open("results/training_times.txt", "r") as f:
        lines = f.readlines()

    serial_time = None
    parallel_time = None
    for line in lines:
        if "Serial" in line:
            serial_time = float(line.split(":")[1].split()[0])
        if "Parallel" in line:
            parallel_time = float(line.split(":")[1].split()[0])

    if serial_time and parallel_time:
        speedup = serial_time / parallel_time
        print(f"Speedup: {speedup:.2f}")

        # Save speedup chart
        labels = ["Serial", "Parallel"]
        times = [serial_time, parallel_time]
        speedups = [1, speedup]

        fig, ax = plt.subplots(1,2, figsize=(10,4))
        ax[0].bar(labels, times, color=['blue','green'])
        ax[0].set_title("Training Time Comparison")
        ax[0].set_ylabel("Seconds")

        ax[1].bar(labels, speedups, color=['blue','green'])
        ax[1].set_title("Speedup Comparison")
        ax[1].set_ylabel("Relative Speedup")

        plt.savefig("results/speedup_chart.png")
        plt.close()

if __name__ == "__main__":
    compare_results()
