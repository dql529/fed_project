import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter


def plot_accuracy_vs_epoch(accuracies, num_epochs, learning_rate, model):
    # 定义一个函数，将y轴刻度转换为百分比格式，保留两位小数
    def to_percent(y, position):
        return f"{100*y:.2f}%"

    formatter = FuncFormatter(to_percent)

    plt.figure(figsize=(10, 6))  # Set the figure size
    plt.plot(
        range(1, num_epochs + 1),
        accuracies,
        marker="o",
        linestyle="-",
        color="b",
        label="Accuracy",
    )  # Plot accuracy
    plt.xlabel("Epoch", fontsize=14)  # Set the label for the x-axis
    plt.ylabel("Accuracy", fontsize=14)  # Set the label for the y-axis
    plt.title("Accuracy vs. Epoch", fontsize=16)  # Set the title
    plt.grid(True)  # Add grid lines
    plt.legend(fontsize=12)  # Add a legend
    plt.xticks(fontsize=12)  # Set the size of the x-axis ticks
    plt.yticks(fontsize=12)  # Set the size of the y-axis ticks
    plt.gca().yaxis.set_major_formatter(formatter)  # Set the formatter for the y-axis
    plt.xlim([1, num_epochs])  # Set the range of the x-axis
    plt.ylim([0, 1])  # Set the range of the y-axis

    # Find the maximum accuracy and its corresponding epoch
    max_accuracy = max(accuracies)
    max_epoch = accuracies.index(max_accuracy) + 1

    # Print the coordinates of the maximum point
    print(
        f"learning rate {learning_rate}, epoch {num_epochs} and dimension {model.num_output_features},Maximum accuracy of {100*max_accuracy:.2f}% at epoch {max_epoch}"
    )

    # Annotate the maximum point
    plt.annotate(
        f"Max Accuracy: {100*max_accuracy:.2f}%",
        xy=(max_epoch, max_accuracy),
        xytext=(max_epoch + 5, max_accuracy - 0.1),
        arrowprops=dict(facecolor="red", shrink=0.05),
    )

    plt.show()
