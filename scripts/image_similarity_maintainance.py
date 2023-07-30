import os, re, shutil, glob
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image
import numpy as np
from utils.create_graphics import plot_scores


def image_to_vector(image_path):
    # Open image and convert to grayscale
    image = Image.open(image_path)
    image = np.array(image).flatten()# / 255.0
    # Resize to a fixed size, if necessary
    #image = image.resize((50, 50))
    # Flatten and normalize
    return image


def sorting_key(filename):
    match = re.match(r'^(\d{1,3})_', filename)
    return int(match.group(1)) if match else float('inf')


def compare_images(directory_path):
    # List all files in the directory and sort them
    filenames = sorted(os.listdir(directory_path), key=sorting_key)
    #print(filenames)

    # Read the first image
    first_image = image_to_vector(os.path.join(directory_path, filenames[0]))
    similarities = [1.,]

    for filename in filenames[1:]:
        current_image = image_to_vector(os.path.join(directory_path, filename))

        # Calculate and print the cosine similarity
        similarity = cosine_similarity(first_image.reshape(1, -1), current_image.reshape(1, -1))
        similarities.append(similarity[0][0])
        print(f"Cosine similarity between {filenames[0]} and {filename} is {similarity[0][0]}")
    print(similarities)
    #plot_scores(similarities, save_dir=r'./output/“red racoon holding laser gun standing face t')
    return similarities


def truncate_similarities(similarities, threshold=0.95):
    for i, value in enumerate(similarities):
        if value < threshold:
            del similarities[i:]
            break
    return similarities


# Function to extract the aesthetic score from the filename
def extract_score(filename):
    match = re.search(r'_(\d+.\d+)\.jpg$', filename)
    return float(match.group(1)) if match else 0

# Function to find the filename with the highest score among the first `num_images` images
def find_best_image(directory_path, num_images):
    # List all files in the directory and sort them
    filenames = sorted(os.listdir(directory_path), key=lambda filename: int(filename.split('_')[0]))

    # Make sure we don't try to access more images than exist
    num_images = min(num_images, len(filenames))

    # Get the first `num_images` filenames and their scores
    scores = [(filename, extract_score(filename)) for filename in filenames[:num_images]]

    # Return the filename with the highest score
    return max(scores, key=lambda x: x[1])[0]


def extract_score(filename):
    # The score is the part after the last underscore and before the .jpg
    return float(filename.rsplit('_', 1)[1][:-4])

def find_duplicate_score(directory_path):
    # List all files in the directory and filter those starting with '0_' and sort them
    filenames = sorted(os.listdir(directory_path), key=lambda filename: int(filename.split('_')[0]))
    duplicate_score = float(filenames[0].split('_')[-1].split('.jpg')[0]) * 2

    # Extract the scores from the filenames
    scores = [(filename, extract_score(filename)) for filename in filenames]

    # If no images start with '0_', return None
    if not scores:
        return None

    # Return the filename with the score closest to the duplicate score
    return min(scores, key=lambda x: abs(x[1] - duplicate_score))[0]

# usage


def rename_images(directory):
    # Loop over all files in the given directory
    for filename in os.listdir(directory):
        # Check if the file is an image with .jpg extension
        if filename.endswith('.jpg'):
            # Check if the filename does not start with 'initial_' or an integer
            if re.match(r'^[0-9]', filename):
                # Get the file's current path
                old_file_path = os.path.join(directory, filename)
                # Generate the new path for the file
                new_file_path = os.path.join(directory, '0.95_' + filename)
                # Rename the file
                os.rename(old_file_path, new_file_path)


def remove_images(directory):
    # Loop over all files in the given directory
    for filename in os.listdir(directory):
        # Check if the file is an image with .jpg extension and starts with '0.9_'
        if filename.endswith('.jpg') and filename.startswith('0.9'):
            # Get the file's path
            file_path = os.path.join(directory, filename)
            # Remove the file
            os.remove(file_path)


def find_improved_image(directory_path, threshold = 0.95):
    # Initialize a list to store the numbers
    # Iterate through each subdirectory
    for dir_name in os.listdir(directory_path):
        subdir_path = os.path.join(directory_path, dir_name)
        if os.path.isdir(subdir_path):
            for img_dir_name in os.listdir(subdir_path):
                if img_dir_name.startswith('image'):
                    img_dir_path = os.path.join(subdir_path, img_dir_name)
                    #print(img_dir_path)
                    #similarities = truncate_similarities(compare_images(img_dir_path), threshold=threshold)
                    #best_image = find_best_image(img_dir_path, len(similarities))
                    best_image = find_duplicate_score(img_dir_path)
                    print(f"The best image in {img_dir_path} is {best_image}")

                    # Copy the image one directory higher with a new name
                    source_path = os.path.join(img_dir_path, best_image)
                    #new_image_name = f"{threshold}_{best_image}"
                    new_image_name = f"duplicate_score_{best_image}"
                    target_path = os.path.join(subdir_path, new_image_name)
                    shutil.copy(source_path, target_path)
                    #print(f"Copied {source_path} to {target_path}")

                continue




if __name__ == '__main__':
    #compare_images(r'D:\StableDiffusionEmbeddings\output\evaluation2\additional_images\a beautiful painting of a peaceful lake in th\image1')
    #compare_images(r'D:\StableDiffusionEmbeddings\output\evaluation2\aesthetic_pred\images\chrome and gold wolf, glossy, metallic, neon,\image')
    #compare_images(r'D:\StableDiffusionEmbeddings\output\evaluation2\aesthetic_pred\images\_“red racoon holding laser gun standing face t\image')
    #compare_images(r'D:\StableDiffusionEmbeddings\output\evaluation2\aesthetic_pred\images\ultra realistic illustration, robot brown owl\image')
    #compare_images(r'D:\StableDiffusionEmbeddings\output\evaluation2\additional_images\sun rising in digital art\image1')
    #compare_images(r'D:\StableDiffusionEmbeddings\output\evaluation2\additional_images\An otherworldly landscape with floating islan\image1')
    #compare_images(r'D:\StableDiffusionEmbeddings\output\evaluation2\aesthetic_pred\images\_giant nordic hell dragon attacking a victoria\image')
    #compare_images(r'D:\StableDiffusionEmbeddings\output\evaluation2\aesthetic_pred\images\magnificent forts of indian temples surrounde\image')
    #ompare_images(r'D:\StableDiffusionEmbeddings\output\evaluation2\aesthetic_pred\images\_pencil drawing of a rubber ducky in the style\image')

    #print(find_duplicate_score(r'D:\StableDiffusionEmbeddings\output\evaluation2\sharpness\images\render of a large purple panther at night roa\image1'))
    #print(find_best_image(r'D:\StableDiffusionEmbeddings\output\evaluation2\sharpness\images\render of a large purple panther at night roa\image1', 8))

    #find_improved_image(r'D:\StableDiffusionEmbeddings\output\evaluation2\aesthetic_pred\images')
    #find_improved_image(r'D:\StableDiffusionEmbeddings\output\evaluation2\sharpness\images')
    #find_improved_image(r'D:\StableDiffusionEmbeddings\output\evaluation2\additional_images')
    find_improved_image(r'D:\StableDiffusionEmbeddings\output\evaluation2\sharpness\images')

