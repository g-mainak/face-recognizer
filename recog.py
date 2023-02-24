import os
import cv2
import face_recognition
from multiprocessing import Pool
from PIL import Image
import numpy as np
import pickle
import time
from functools import partial
import shutil

# Set the paths to the folders containing the images
known_folder = "C:/Users/g_mai/Desktop/mainak_pictures"
unknown_folder = "C:/Users/g_mai/Desktop/pictures"
output_folder = "C:/Users/g_mai/Desktop/joy/"
encodings_file = "encodings.pkl"
pictures_file = "pictures.txt"

# Define a function to convert HEIC files to JPEG format
def convert_heic_to_jpeg(file):
    if file.endswith('.HEIC') or file.endswith('.heic'):
        heic_path = os.path.join(unknown_folder, file)
        jpeg_path = os.path.join(unknown_folder, os.path.splitext(file)[0] + '.jpg')
        Image.open(heic_path).save(jpeg_path)
        return jpeg_path
    else:
        return os.path.join(unknown_folder, file)

# Define a worker function to process each image in parallel
def process_image(known_encodings, file):
    image_path = convert_heic_to_jpeg(file)
    image = cv2.imread(image_path)

    # Convert the image from BGR (OpenCV's default) to RGB (face_recognition's default)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Find all the faces in the image
    face_encodings = face_recognition.face_encodings(rgb_image)
    # import pdb; pdb.set_trace()
    # Check if any of the faces in the image match the target person
    if len(face_encodings) > 0:
        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(known_encodings, face_encoding)
            if True in matches:
                shutil.copy(image_path, output_folder + file)
    return file

if __name__ == '__main__':
    # Load the known encodings from the encodings file if it exists
    if os.path.isfile(encodings_file):
        print('Reading encodings file')
        with open(encodings_file, 'rb') as f:
            known_encodings = pickle.load(f)
    else:
        print('No Encodings file. Creating encodings file')
        # Load the images of the target person from the known folder
        known_images = []
        for file in os.listdir(known_folder):
            image_path = os.path.join(known_folder, file)
            image = face_recognition.load_image_file(image_path)
            known_images.append(image)

        # Compute the face encodings for each known image
        known_encodings = [face_recognition.face_encodings(image)[0] for image in known_images]

        # Save the known encodings to the encodings file
        with open(encodings_file, 'wb') as f:
            pickle.dump(known_encodings, f)


    # Use a pool of workers to process the images in parallel
    with Pool(4) as pool:
        start_time = time.time()
        num_images_processed = 0

        files = ([line.strip() for line in open(pictures_file, 'r')] if os.path.exists(pictures_file) else os.listdir(unknown_folder))
        set_files = set(files)
        with open("pictures.txt", "w") as pic_file:
            # import pdb; pdb.set_trace()
            for file in files:
                pic_file.write(str(file) + "\n")
        process_image_mult = partial(process_image, known_encodings)
        # Map the process_image function to each image in the unknown folder
        results = pool.imap_unordered(process_image_mult, files)

        # Iterate over the results and filter out any None results
        for result in results:
            num_images_processed += 1
            set_files.remove(result)

            # Output the number of images processed every 100 images
            if num_images_processed % 10 == 0:
                with open("pictures.txt", "w") as pic_file:
                    for file in set_files:
                        pic_file.write(str(file) + "\n")
                elapsed_time = time.time() - start_time
                print(f"Processed {num_images_processed} images in {elapsed_time:.2f} seconds")

