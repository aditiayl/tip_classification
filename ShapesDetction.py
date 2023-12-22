import cv2
import os
import numpy as np

def detect_perfect_circles(image_path):
    # Read the image
    original_image = cv2.imread(image_path, cv2.IMREAD_COLOR)

    # Convert the image to grayscale
    gray = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)

    # Adjust contrast
    contrast_img = cv2.addWeighted(gray, 5, np.zeros_like(gray, dtype=np.uint8), 0, 0)

    # Apply GaussianBlur to reduce noise and help with edge detection
    blurred = cv2.GaussianBlur(contrast_img, (5, 5), 0)

    # Perform edge detection using Canny
    edges = cv2.Canny(blurred, 50, 150)

    # Use HoughCircles on the edges to detect circles
    circles = cv2.HoughCircles(
        edges, 
        cv2.HOUGH_GRADIENT, dp=1, minDist=50, param1=100, param2=30, minRadius=10, maxRadius=100
    )

    # If circles are found, filter based on circularity and draw on the original image
    if circles is not None:
        circles = np.uint16(np.around(circles))

        for i in circles[0, :]:
            # Create a binary image representing the filled circle
            circle_binary = np.zeros_like(edges)
            cv2.circle(circle_binary, (i[0], i[1]), i[2], 255, thickness=-1)

            # Compute circularity
            area = cv2.contourArea(np.array([np.column_stack(np.where(circle_binary == 255))]))
            
            # Check if the area is non-zero before calculating circularity
            if area > 0:
                perimeter = cv2.arcLength(np.array([[(i[0], i[1])]], dtype=np.int32), True)
                circularity = (perimeter ** 2) / (4 * np.pi * area)

                # Draw only if circularity is close to 1 (perfect circle)
                if circularity > 0.8:
                    cv2.circle(original_image, (i[0], i[1]), i[2], (0, 255, 0), 2)
                    cv2.circle(original_image, (i[0], i[1]), 2, (0, 0, 255), 3)

    # Convert the original image to grayscale for thresholding
    gray_original = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)

    # Apply thresholding (adjust the threshold value as needed)
    _, thresholded = cv2.threshold(gray_original, 200, 255, cv2.THRESH_BINARY)

    # Display the result
    cv2.imshow('Perfect Circle Detection with Edge Detection', np.hstack([original_image, cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR), cv2.cvtColor(thresholded, cv2.COLOR_GRAY2BGR)]))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Specify the directory containing your images
    images_directory = "test_images/"

    # Loop through all image files in the directory
    for filename in os.listdir(images_directory):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image_path = os.path.join(images_directory, filename)
            detect_perfect_circles(image_path)
