# Traffic Sign Recognition Project

## Overview
This project focuses on building a deep neural network model that can classify traffic signs into different categories. The model will be trained to recognize various traffic signs, such as speed limits, no entry, traffic signals, and others, to assist autonomous vehicles in understanding and following traffic rules.

## What is Traffic Sign Recognition?
Traffic Sign Recognition (TSR) is the process of identifying and classifying traffic signs from images. There are various types of traffic signs like speed limits, no entry, traffic signals, turn left or right, children crossing, no passing of heavy vehicles, etc. TSR aims to categorize these signs to help vehicles interpret and follow traffic rules.

## Need for Traffic Sign Recognition or Applications
With the rise of self-driving cars, the ability to accurately understand and interpret traffic signs has become a crucial component of autonomous driving systems. For fully autonomous vehicles to achieve Level 5 autonomy, they must be able to read and understand traffic signs and make decisions accordingly. Companies like Tesla, Uber, Google, Mercedes-Benz, Toyota, Ford, and Audi are working to integrate TSR into their self-driving technologies, ensuring that vehicles can navigate safely and follow road rules.

## About the Python Project
In this project, we build a deep neural network model that classifies traffic signs into different categories. The model is trained using the German Traffic Sign Dataset from Kaggle. Once trained, the model can recognize various traffic signs from images and classify them correctly, which is essential for autonomous vehicles to navigate safely.

## Dataset for the Project
The project uses the **German Traffic Sign Dataset** (GTSRB) from Kaggle, which contains over 50,000 images of different traffic signs. The dataset is divided into 43 different classes of traffic signs. Some classes have many images, while others have fewer, offering a balanced set for training a robust model.

- **Dataset Link**: [German Traffic Sign Dataset](https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign)
- **Dataset Size**: Around 300 MB
- **Dataset Structure**:
  - **Train Folder**: Contains images of traffic signs categorized into 43 classes.
  - **Test Folder**: Used for testing the trained model.

## Skills Used
- Python
- Deep Learning (Neural Networks)
- Convolutional Neural Networks (CNNs)
- Data Preprocessing
- TensorFlow/Keras
- Image Classification
- Dataset Handling
- Model Training and Evaluation

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/traffic-sign-recognition.git
   ```
2. Navigate to the project directory:
   ```bash
   cd traffic-sign-recognition
   ```
3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
1. Download the German Traffic Sign Dataset from the Kaggle link above.
2. Extract the dataset and place the image files in the appropriate folders (`train` and `test`).
3. Run the model training script:
   ```bash
   python train_model.py
   ```
4. After training, use the model to classify traffic signs from new images:
   ```bash
   python classify_sign.py --image path_to_image
   ```

## Example
Once the model is trained, it can predict the category of a traffic sign from an image:

```python
# Example of classifying a traffic sign
model = load_model("traffic_sign_model.h5")
image = load_image("path_to_image.jpg")
predicted_class = model.predict(image)
print(f"Predicted Traffic Sign Class: {predicted_class}")
```

## Acknowledgements
- **Kaggle**: For providing the German Traffic Sign Dataset.
- **TensorFlow/Keras**: For building and training the deep learning model.
- **Python Libraries**: NumPy, OpenCV, Matplotlib for data handling and visualization.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
