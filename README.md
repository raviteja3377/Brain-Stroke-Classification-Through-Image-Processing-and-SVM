# Brain-Stroke-Classification-Through-Image-Processing-and-SVM
This project aims to classify brain MRI images as Stroke or Normal using traditional image processing techniques and a Support Vector Machine (SVM) classifier.
Technologies Used
Python
OpenCV
NumPy
scikit-learn
Matplotlib
How It Works
Image Preprocessing: Each image is resized and converted to grayscale.
Feature Extraction: Images are flattened into 1D feature vectors.
Model Training: An SVM with a linear kernel is trained on labeled MRI data.
Prediction: The trained model predicts whether the image is 'Stroke' or 'Normal'.
How to Run
Clone the repo:
git clone https://github.com/yourusername/brain-stroke-classification.git
cd brain-stroke-classification
Install dependencies:
pip install opencv-python scikit-learn numpy matplotlib
Ensure your dataset is in the brain_dataset/ folder.
Run the script:
python stroke_classifier.py
Sample Output
Accuracy Report printed in the terminal
Sample Image displayed with predicted label
Future Improvements
Use deep learning (e.g., CNNs) for better accuracy
Web-based UI for real-time classification
Integration with medical record systems
📧 Contact
Feel free to reach out or contribute!
Author: V. Raviteja, G. Arya, K. Anusha
Email: ravitejavavilla37@gmail.com, aryagiravena2104@gmail.com, kommanaboinaanusha752@gmail.com
GitHub: @raviteja_3377
