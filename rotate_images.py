from PIL import Image
import os

def rotate_and_save_image(input_path, output_path, angle):
    # Open the image file
    original_image = Image.open(input_path)

    # Rotate the image
    rotated_image = original_image.rotate(angle)

    # Get the file name and extension
    file_name, file_extension = os.path.splitext(os.path.basename(input_path))

    # Construct the output file name with the rotation angle
    output_file_name = f"{angle}_{file_name}{file_extension}"

    # Save the rotated image
    rotated_image.save(os.path.join(output_path, output_file_name))

if __name__ == "__main__":
    # Replace 'input_directory' with the directory containing your input images
    input_directory = "dataset/NG"

    # Replace 'output_directory' with the desired directory for the rotated images
    output_directory = "dataset/NG"

    # Specify the rotation angles (in degrees)
    rotation_angles = [90, 180, 270]

    # Create the output directory if it doesn't exist
    os.makedirs(output_directory, exist_ok=True)

    # Loop through all files in the input directory
    for filename in os.listdir(input_directory):
        if filename.endswith(('.jpg', '.jpeg', '.png')):
            # Construct the full path to the input file
            input_path = os.path.join(input_directory, filename)

            # Loop through all rotation angles
            for angle in rotation_angles:
                # Rotate and save the image
                rotate_and_save_image(input_path, output_directory, angle)

    print(f"Images rotated by {rotation_angles} degrees and saved to {output_directory}")
