import matplotlib.pyplot as plt

def compare_results():
    # Load training times and accuracy from file
    with open("results/training_times.txt", "r") as f:
        lines = f.readlines()

    serial_time = None
    parallel_time = None
    serial_acc = None
    parallel_acc = None

    for line in lines:
        if "Serial" in line:
            parts = line.strip().split("|")
            serial_time = float(parts[0].split(":")[1].split()[0])
            serial_acc = float(parts[1].split(":")[1].replace("%", "").strip())
        if "Parallel" in line:
            parts = line.strip().split("|")
            parallel_time = float(parts[0].split(":")[1].split()[0])
            parallel_acc = float(parts[1].split(":")[1].replace("%", "").strip())

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

    if serial_acc and parallel_acc:
        # Save accuracy comparison chart
        labels = ["Serial", "Parallel"]
        accuracies = [serial_acc, parallel_acc]

        plt.bar(labels, accuracies, color=['blue','green'])
        plt.title("Accuracy Comparison")
        plt.ylabel("Accuracy (%)")
        plt.savefig("results/accuracy_comparison.png")
        plt.close()

        print(f"Serial Accuracy: {serial_acc:.2f}% | Parallel Accuracy: {parallel_acc:.2f}%")

if __name__ == "__main__":
    compare_results()
