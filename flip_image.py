from PIL import Image
import os

def flip_and_save_image(input_path, output_path, flip_type):
    # Open the image file
    original_image = Image.open(input_path)

    # Flip the image horizontally or vertically
    if "horizontal" in flip_type:
        flipped_horizontal_image = original_image.transpose(Image.FLIP_LEFT_RIGHT)
        # Get the file name and extension
        file_name, file_extension = os.path.splitext(os.path.basename(input_path))
        # Construct the output file name with the flip type
        output_file_name = f"horizontal_flip_{file_name}{file_extension}"
        # Save the flipped image
        flipped_horizontal_image.save(os.path.join(output_path, output_file_name))

    if "vertical" in flip_type:
        flipped_vertical_image = original_image.transpose(Image.FLIP_TOP_BOTTOM)
        # Get the file name and extension
        file_name, file_extension = os.path.splitext(os.path.basename(input_path))
        # Construct the output file name with the flip type
        output_file_name = f"vertical_flip_{file_name}{file_extension}"
        # Save the flipped image
        flipped_vertical_image.save(os.path.join(output_path, output_file_name))

if __name__ == "__main__":
    # Replace 'input_directory' with the directory containing your input images
    input_directory = "dataset/AG"

    # Replace 'output_directory' with the desired directory for the flipped images
    output_directory = "dataset/AG"

    # Specify the flip types ('horizontal', 'vertical', or both)
    flip_types = ["horizontal", "vertical"]

    # Create the output directory if it doesn't exist
    os.makedirs(output_directory, exist_ok=True)

    # Loop through all files in the input directory
    for filename in os.listdir(input_directory):
        if filename.endswith(('.jpg', '.jpeg', '.png')):
            # Construct the full path to the input file
            input_path = os.path.join(input_directory, filename)

            # Flip and save the image for each flip type in the list
            flip_and_save_image(input_path, output_directory, flip_types)

    print(f"Images flipped {flip_types} and saved to {output_directory}")
