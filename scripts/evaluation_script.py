import os
from utils.create_graphics import plot_scores, create_boxplots
import matplotlib.pyplot as plt
import numpy as np
import shutil
import glob


def extract_indexed_elements(directory_path):
    # Initialize a list to store the numbers
    numbers_list = []

    index_0_list = []
    index_24_list = []
    index_49_list = []

    #index_99_list = []
    #index_199_list = []
    #index_299_list = []
    #index_499_list = []
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

                        index_0_list.append(lines[0])
                        index_24_list.append(lines[24])
                        index_49_list.append(lines[49])

                        #index_99_list.append(lines[99])
                        #index_199_list.append(lines[199])
                        #index_299_list.append(lines[299])
                        #index_499_list.append(lines[499])
    #return index_0_list, index_99_list, index_199_list, index_299_list, index_499_list
    #index_0_list = [x // 10 for x in index_0_list]
    #index_24_list = [x // 10 for x in index_24_list]
    #index_49_list = [x // 10 for x in index_49_list]

    return index_0_list, index_24_list, index_49_list


def get_score_delta(directory_path):
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
                        #lines = list(set(lines))
                        lines = [float(num) for num in lines]
                        numbers_list.append(max(lines) - lines[0])
    print(numbers_list)
    #numbers_list = [x // 10 for x in numbers_list]
    print(numbers_list)
    fig, ax = plt.subplots()
    ax.boxplot(numbers_list)

    # Add labels and title
    #ax.set_xlabel('Data')
    ax.set_ylabel('Score Delta')
    ax.set_title('Score Delta Boxplot')
    plt.savefig(f'./output/score_increase.png')



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
    cnt = 0
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





if __name__ == '__main__':
    directory_path = r"./output/evaluation2/aesthetic_pred/images"
    #directory_path = r"./output/evaluation2/sharpness/images"

    #index_0_list, index_99_list, index_199_list, index_299_list, index_499_list = extract_indexed_elements(directory_path)
    #index_0_list, index_24_list, index_49_list = extract_indexed_elements(directory_path)

    #data = [index_0_list, index_99_list, index_199_list, index_299_list, index_499_list]
    #data = [index_0_list, index_24_list, index_49_list]
    #labels = ['Index 0', 'Index 99', 'Index 199', 'Index 299', 'Index 499']
    #labels = ['Index 0', 'Index 24', 'Index 49']

    #create_boxplots(data, labels)

    #get_best_image(directory_path)
    #plot_images(directory_path)
    #get_score_delta(directory_path)

    plot_images(directory_path)

    #print("Means for each position:")
    #print(result)
    #plot_scores(result, './output/evaluation/aesthetic_scores.png')