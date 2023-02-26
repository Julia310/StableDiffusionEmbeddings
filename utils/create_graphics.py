import matplotlib.pyplot as plt

def create_boxplot(values, prompt_aesthetics=None, filename=None):
    # Create the figure and boxplot
    fig, ax = plt.subplots()
    bp = ax.boxplot(values)

    # Add the mean as an empty dot to the plot
    if prompt_aesthetics is not None:
        ax.scatter([1], [prompt_aesthetics], marker='o', color='blue', s=50)

    # Save the figure to disk if a filename is provided
    if filename is not None:
        print(f'save {filename}')
        plt.savefig(filename)

    return fig
