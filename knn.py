import cv2
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

# Step 1: Load and preprocess your dataset
dataset_path = 'dataset/'
categories = ['GO', 'AG', 'NG']

# Initialize lists to store data
data = []
labels = []

# Load images and labels
for category in categories:
    category_path = os.path.join(dataset_path, category)
    
    # Check if the category path exists
    if not os.path.exists(category_path):
        raise ValueError(f"Category path {category_path} does not exist.")
    
    for filename in os.listdir(category_path):
        if filename.endswith(".png"):
            img_path = os.path.join(category_path, filename)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # Read image in grayscale
            img = cv2.resize(img, (500, 500))  # Resize to a consistent size
            data.append(img)
            labels.append(category)

# Check if any images were loaded
if len(data) == 0:
    raise ValueError("No images loaded. Check your dataset path and categories.")

# Convert lists to NumPy arrays
X = np.array(data)
y = np.array(labels)

# Step 2: Encode labels into numerical values
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Step 3: Data Splitting
X_train, X_temp, y_train, y_temp = train_test_split(X, y_encoded, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Step 5: Scaling Features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train.reshape(X_train.shape[0], -1))
X_val_scaled = scaler.transform(X_val.reshape(X_val.shape[0], -1))

# Step 6: Training KNN
k_value = 3  # Choose an appropriate value for K
knn_model = KNeighborsClassifier(n_neighbors=k_value)
knn_model.fit(X_train_scaled, y_train)

# Step 7: Evaluation
X_test_scaled = scaler.transform(X_test.reshape(X_test.shape[0], -1))

y_test_pred = knn_model.predict(X_test_scaled)

# Step 8: Print accuracy and classification report
accuracy = accuracy_score(y_test, y_test_pred)
print(f'Accuracy on Test Set: {accuracy:.2f}')
print('\nClassification Report:\n', classification_report(y_test, y_test_pred))

# Step 9: Test the Model on New Images in a Folder
test_folder_path = 'test_images/'  # Replace with the path to your test images folder

# Iterate through all images in the test folder
for filename in os.listdir(test_folder_path):
    if filename.endswith(".png"):
        test_image_path = os.path.join(test_folder_path, filename)

        # Read and preprocess the test image
        test_img = cv2.imread(test_image_path, cv2.IMREAD_GRAYSCALE)

        # Apply GaussianBlur to reduce noise and help with circle detection
        blurred = cv2.GaussianBlur(test_img, (9, 9), 2)

        # Use HoughCircles to detect circles in the image
        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=50,
            param1=50,
            param2=30,
            minRadius=10,
            maxRadius=100
        )

        if circles is not None:
            # Convert the (x, y) coordinates and radius of the first circle to integers
            (x, y, r) = np.round(circles[0, 0]).astype("int")

            # Draw the first circle on the original image
            # cv2.circle(test_img, (x, y), r, (0, 255, 0), 4)

            # Crop the image around the detected circle with a margin
            test_img = test_img[max(y - r - 150, 0):min(y + r + 150, test_img.shape[0]),
                                max(x - r - 150, 0):min(x + r + 150, test_img.shape[1])]

        test_img = cv2.resize(test_img, (500, 500))
        
        # Reshape and scale the test image
        test_img_flat = test_img.reshape(1, -1)
        test_img_scaled = scaler.transform(test_img_flat)
        
        # Predict the category of the test image
        predicted_category = label_encoder.inverse_transform(knn_model.predict(test_img_scaled))[0]
        
        # Display the result
        print(f'The predicted category for {filename} is: {predicted_category}')
        
        # Display the image with the predicted category
        cv2.imshow(f'Test Image - Predicted Category: {predicted_category}', test_img)
        cv2.waitKey(0)

# Close all OpenCV windows
cv2.destroyAllWindows()
