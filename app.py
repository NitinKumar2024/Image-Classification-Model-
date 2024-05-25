from tensorflow.keras.models import load_model  # Import the function to load a saved model
import numpy as np  # Import NumPy library for array operations
from tensorflow.keras.preprocessing import image  # Import image preprocessing utilities

# Define the class names based on the CIFAR-10 dataset
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

# Load the saved model
loaded_model = load_model('ann.h5')  # Load the model from the specified directory

# Load and preprocess the image
img_path = 'Dataset\\dog.jpg'  # Path to the input image
img = image.load_img(img_path, target_size=(32, 32))  # Load the image and resize it to match the model input shape
img_array = image.img_to_array(img)  # Convert the image to an array
img_array = np.expand_dims(img_array, axis=0)  # Add a batch dimension to the array
img_array /= 255.0  # Normalize pixel values to range [0, 1]
# Make predictions
predictions = loaded_model.predict(img_array)  # Use the loaded model to make predictions on the input image

# Get the predicted class label
predicted_class_index = np.argmax(predictions)  # Get the index of the class with the highest probability
predicted_class = class_names[predicted_class_index]  # Get the corresponding class name

# Print the predicted class and probabilities
print(f'Predicted class: {predicted_class}')  # Print the predicted class label
print(f'Predicted probabilities: {predictions}')  # Print the predicted probabilities for each class