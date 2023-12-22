import cv2
import numpy as np
import os

def detect_circles(image_path, min_radius=10, max_radius=100, margin=150):
    # Read the image
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)

    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply GaussianBlur to reduce noise and help with circle detection
    blurred = cv2.GaussianBlur(gray, (9, 9), 2)

    # Use HoughCircles to detect circles in the image
    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=50,
        param1=50,
        param2=30,
        minRadius=min_radius,
        maxRadius=max_radius
    )

    if circles is not None:
        # Convert the (x, y) coordinates and radius of the first circle to integers
        (x, y, r) = np.round(circles[0, 0]).astype("int")

        # Draw the first circle on the original image
        cv2.circle(image, (x, y), r, (0, 255, 0), 4)

        # Crop the image around the detected circle with a margin
        cropped_image = image[y - r - margin:y + r + margin, x - r - margin:x + r + margin]

        # Display the result
        cv2.imshow("Detected Circle", image)
        cv2.imshow("Cropped Image", cropped_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # Save the cropped image
    #     cropped_image_path = image_path.replace(".", f"_cropped_{margin}.")
    #     cv2.imwrite(cropped_image_path, cropped_image)
    #     print(f"Cropped image saved to {cropped_image_path}")
    # else:
    #     print(f"No circles detected in {image_path}.")

# Specify the directory containing your images
images_directory = "test_images/"

# Loop through all images in the directory
for filename in os.listdir(images_directory):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        image_path = os.path.join(images_directory, filename)
        detect_circles(image_path)
