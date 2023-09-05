import os
from utils.create_graphics import plot_scores, create_boxplots
import matplotlib.pyplot as plt
import numpy as np
import shutil
import glob
import numpy as np


def extract_indexed_elements(directory_path, max_idx=500):
    # Initialize a list to store the numbers
    numbers_list = []

    index_0_list = []
    index_24_list = []
    index_49_list = []
    index_99_list = []
    index_199_list = []
    index_299_list = []
    index_499_list = []

    # Iterate through each subdirectory
    for dir_name in os.listdir(directory_path):
        subdir_path = os.path.join(directory_path, dir_name)
        if os.path.isdir(subdir_path):
            for file_name in os.listdir(subdir_path):
                # Check if the file name matches the pattern
                if file_name.endswith("_output.txt"):
                    file_path = os.path.join(subdir_path, file_name)
                    with open(file_path, 'r') as f:
                        lines = f.readlines()
                        lines = [line.strip() for line in lines]  # Remove leading/trailing whitespace and newlines
                        #lines = list(set(lines))
                        lines = [float(num) for num in lines]
                        numbers_list.append(lines)

                        if max_idx == 500:
                            index_0_list.append(lines[0])
                            index_99_list.append(lines[99])
                            index_199_list.append(lines[199])
                            index_299_list.append(lines[299])
                            index_499_list.append(lines[499])
                        else:
                            index_0_list.append(lines[0])
                            index_24_list.append(lines[24])
                            index_49_list.append(lines[49])
    if max_idx == 500:
        return index_0_list, index_99_list, index_199_list, index_299_list, index_499_list
        #index_0_list = [x // 10 for x in index_0_list]
        #index_24_list = [x // 10 for x in index_24_list]
        #index_49_list = [x // 10 for x in index_49_list]

    return index_0_list, index_24_list, index_49_list


import os
import matplotlib.pyplot as plt


def get_score_delta(directory_path, y_label='aesthetic score delta', return_delta=True):
    # Initialize a list to store the numbers
    numbers_list = []

    # Iterate through each subdirectory
    for dir_name in os.listdir(directory_path):
        subdir_path = os.path.join(directory_path, dir_name)
        if os.path.isdir(subdir_path):
            for file_name in os.listdir(subdir_path):
                # Check if the file name matches the pattern
                if file_name.endswith("_output.txt"):
                    file_path = os.path.join(subdir_path, file_name)
                    with open(file_path, 'r') as f:
                        lines = f.readlines()
                        lines = [line.strip() for line in lines]  # Remove leading/trailing whitespace and newlines
                        lines = [float(num) for num in lines]
                        #numbers_list.append(lines[0] - min(lines) )
                        numbers_list.append(max(lines) - lines[0])

    if return_delta:
        return numbers_list

    #fig, ax = plt.subplots()
    fig, ax = plt.subplots(figsize=(4, 6))  # Change the '4' to your desired width in inches
    ax.boxplot(numbers_list, widths=0.3)

    # Add labels
    #ax.set_ylabel(y_label)
    ax.set_xlabel(y_label)

    # Remove y-axis numbers
    ax.set_xticks([])

    # Adjust left margin
    #fig.subplots_adjust(left=0.2)  # Adjust '0.25' as needed

    plt.savefig(f'./output/{y_label.replace(" ", "_")}.png')


def get_best_image(directory_path):
    # Initialize a list to store the numbers
    # Iterate through each subdirectory
    dirs = os.listdir(directory_path)
    for dir_name in os.listdir(directory_path):
        subdir_path = os.path.join(directory_path, dir_name)
        if os.path.isdir(subdir_path):
            for file_name in os.listdir(subdir_path):
                # Check if the file name matches the pattern
                if file_name.endswith("_output.txt"):
                    file_path = os.path.join(subdir_path, file_name)
                    with open(file_path, 'r') as f:
                        lines = f.readlines()
                        lines = [line.strip() for line in lines]  # Remove leading/trailing whitespace and newlines
                        #lines = list(set(lines))
                        lines = [float(num) for num in lines][:300]
                        index_of_max = lines.index(max(lines))
                    # Find the directory starting with 'image'
                    for img_dir_name in os.listdir(subdir_path):
                        if img_dir_name.startswith('image'):
                            img_dir_path = os.path.join(subdir_path, img_dir_name)
                            # Find the corresponding image in the 'image' subdirectory
                            image_files = glob.glob(os.path.join(img_dir_path, f'{index_of_max}_*.jpg'))

                            # If a corresponding image is found, copy it next to the output.txt with a new name
                            if image_files:
                                image_path = image_files[0]
                                image_name = os.path.basename(image_path)
                                new_image_name = 'best_of_first_300_' + image_name
                                new_image_path = os.path.join(subdir_path, new_image_name)
                                shutil.copy(image_path, new_image_path)


def plot_images(directory_path, window_size=15):
    line_cnt = 0

    # Create a figure
    fig, axs = plt.subplots(5, 1, figsize=(10, 20), sharex=True)

    # Iterate through each subdirectory
    for dir_name in os.listdir(directory_path):
        subdir_path = os.path.join(directory_path, dir_name)
        if os.path.isdir(subdir_path):
            for file_name in os.listdir(subdir_path):
                # Check if the file name matches the pattern
                if file_name.endswith("_output.txt"):
                    file_path = os.path.join(subdir_path, file_name)
                    with open(file_path, 'r') as f:
                        lines = f.readlines()
                        lines = [line.strip() for line in lines]  # Remove leading/trailing whitespace and newlines
                        lines = [float(num) for num in lines]
                        smoothed_scores = np.convolve(lines, np.ones(window_size) / window_size, mode='valid')

                        # Plot on the respective subplot
                        axs[line_cnt % 5].plot(range(window_size - 1, len(lines)), smoothed_scores)
                        line_cnt += 1

                        # If 3 lines have been plotted on the same subplot, switch to the next one
                        if line_cnt % 3 == 0:
                            axs[(line_cnt // 3) % 5].set_ylabel('Score')
                            axs[(line_cnt // 3) % 5].set_xlabel('Iterations')

                        # If 15 lines (3 lines on 5 subplots) have been plotted, save and clear the figure
                        if line_cnt % 15 == 0 and line_cnt != 0:
                            fig.savefig(f'./output/{int(line_cnt/15)}_plot.png')
                            plt.close(fig)  # close the figure

                            # Recreate the figure for the next 5 subplots
                            fig, axs = plt.subplots(5, 1, figsize=(10, 20), sharex=True)

    # If the function ends and there are still unprinted subplots, save them
    if line_cnt % 15 != 0:
        fig.savefig(f'./output/{int(line_cnt/15)+1}_plot.png')
        plt.close(fig)


def compute_statistics(data):
    data = np.array(data)  # Convert data to a numpy array
    # Compute the median
    median = np.median(data)

    # Compute the quantiles
    q1 = np.percentile(data, 25)  # 25th percentile
    q3 = np.percentile(data, 75)  # 75th percentile

    # Compute the IQR (Interquartile Range)
    iqr = q3 - q1

    # Determine the lower and upper whiskers
    lower_whisker = np.min(data[data >= q1 - 1.5 * iqr])
    upper_whisker = np.max(data[data <= q3 + 1.5 * iqr])

    return q1, median, q3, lower_whisker, upper_whisker


if __name__ == '__main__':
    directory_path = r"./output/evaluation2/aesthetic_pred/images"
    metric_delta = get_score_delta(directory_path)
    delta_labels = ['aesthetic delta']

    index_0_list, index_99_list, index_199_list, index_299_list, index_499_list = extract_indexed_elements(directory_path)
    data = [index_0_list, index_99_list, index_199_list, index_299_list, index_499_list]
    labels = ['0', '99', '199', '299', '499']
    #create_boxplots(data, labels, x_label='iterations', y_label='aesthetic score')
    #create_boxplots(data, labels, metric_delta, delta_labels, x_label='iterations', y_label='aesthetic score')
    create_boxplots(data, labels, x_label='iterations', y_label='aesthetic score')
    for i in range(len(labels)):
        print(labels[i])
        q1, median, q3, lower_whisker, upper_whisker = compute_statistics(data[i])

        print(f"1st Quartile (Q1): {q1}")
        print(f"Median: {median}")
        print(f"3rd Quartile (Q3): {q3}")
        print(f"Lower Whisker (without outliers): {lower_whisker}")
        print(f"Upper Whisker (without outliers): {upper_whisker}")


    delta_labels = ['sharpness delta']
    labels = ['0', '24', '49']
    directory_path = r"./output/evaluation2/sharpness/images"
    metric_delta = get_score_delta(directory_path, y_label='sharpness delta')
    index_0_list, index_24_list, index_49_list = extract_indexed_elements(directory_path, max_idx=50)
    index_0_list = [x / 10 for x in index_0_list]
    index_24_list = [x / 10 for x in index_24_list]
    index_49_list = [x / 10 for x in index_49_list]
    #metric_delta = [x // 10 for x in metric_delta]
    data = [index_0_list, index_24_list, index_49_list]
    labels = ['0', '24', '49']
    #create_boxplots(data, labels, metric_delta, delta_labels, x_label='iterations', y_label='sharpness x 10')
    create_boxplots(data, labels, x_label='iterations', y_label='sharpness')
    for i in range(len(labels)):
        print(labels[i])
        q1, median, q3, lower_whisker, upper_whisker = compute_statistics(data[i])

        print(f"1st Quartile (Q1): {q1}")
        print(f"Median: {median}")
        print(f"3rd Quartile (Q3): {q3}")
        print(f"Lower Whisker (without outliers): {lower_whisker}")
        print(f"Upper Whisker (without outliers): {upper_whisker}")

    delta_labels = ['blurriness delta']
    directory_path = r"./output/evaluation2/blurriness/images"
    metric_delta = get_score_delta(directory_path, y_label='blurriness delta')
    index_0_list, index_24_list, index_49_list = extract_indexed_elements(directory_path, max_idx=50)
    index_0_list = [x / 10 for x in index_0_list if (x / 10) <= 2]
    print(len(index_0_list))
    index_24_list = [x / 10 for x in index_24_list]
    index_49_list = [x / 10 for x in index_49_list]
    #metric_delta = [x // 10 for x in metric_delta]
    maximum = max(index_0_list)
    data = [index_0_list, index_24_list, index_49_list]
    labels = ['0', '24', '49']
    # create_boxplots(data, labels, metric_delta, delta_labels, x_label='iterations', y_label='blurriness x 10')
    create_boxplots(data, labels, x_label='iterations', y_label='blurriness')
    for i in range(len(labels)):
        print(labels[i])
        q1, median, q3, lower_whisker, upper_whisker = compute_statistics(data[i])

        print(f"1st Quartile (Q1): {q1}")
        print(f"Median: {median}")
        print(f"3rd Quartile (Q3): {q3}")
        print(f"Lower Whisker (without outliers): {lower_whisker}")
        print(f"Upper Whisker (without outliers): {upper_whisker}")



    #index_0_list, index_24_list, index_49_list = extract_indexed_elements(directory_path)

    #data = [index_0_list, index_99_list, index_199_list, index_299_list, index_499_list]
    #data = [index_0_list, index_24_list, index_49_list]
    #labels = ['Index 0', 'Index 99', 'Index 199', 'Index 299', 'Index 499']
    #labels = ['Index 0', 'Index 24', 'Index 49']

    #create_boxplots(data, labels)

    #get_best_image(directory_path)
    #plot_images(directory_path)

    #plot_images(directory_path)

    #print("Means for each position:")
    #print(result)
    #plot_scores(result, './output/evaluation/aesthetic_scores.png')