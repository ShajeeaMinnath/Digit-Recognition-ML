**Digit Recognition using HOG-SVM & PCA**

A collaborative Machine Learning project designed to classify handwritten digits from the MNIST dataset. This project features a full end-to-end pipeline: from Feature Engineering (HOG) and Dimensionality Reduction (PCA) to a Live Web Deployment using Streamlit.

**Team Members & Roles** 

Member A: Shajeea Minnath A T   

Role: Data Engineer	

Primary Responsibilities: EDA, HOG Feature Extraction, Visualization

Member B: Pavithra

Role: ML Scientist

Primary Responsibilities: PCA Dimensionality Reduction, SVM Training,Serialization

Member C: Amrutha 

Role: DevOps & Deployment	

Primary Responsibilities: Streamlit UI Development, Pipeline Integration

**Technical Stack**

Language: Python 3.x
Libraries: scikit-learn, scikit-image, opencv-python, joblib, streamlit
Version Control: Git & GitHub (Branching & Pull Request Workflow)

**Project Workflow (10 Stages)**
1. Project Planning & Setup
Established a GitHub repository with a branching strategy (feature/eda, feature/modeling) to ensure parallel development and code integrity.

2. Data Acquisition
Utilized the MNIST dataset (70,000 images of handwritten digits).

3. Exploratory Data Analysis (EDA)
Visualized digit distributions and pixel intensity histograms to understand data variance and class balance.

4. Data Pre-processing
Normalized pixel values and prepared images for gradient-based feature extraction.

5. Feature Engineering (HOG)
     Implemented Histogram of Oriented Gradients (HOG):

      Orientations: 9

      Pixels per Cell: 7x7

      Cells per Block: 2x2
      This allowed the model to focus on the shape and structure of the digits rather than raw pixel intensity.

6. Dimensionality Reduction (PCA)
     Applied Principal Component Analysis (PCA) to reduce the feature space to 100 components, retaining over 95% of the variance while significantly boosting training speed.

7. Model Training & Evaluation
     Trained a Support Vector Machine (SVM) with an RBF Kernel.

      Result: Achieved 98% accuracy on the test set.

      Generated Confusion Matrices to identify common digit misclassifications (e.g., 4 vs 9).

8. Model Serialization
Saved the trained model, PCA transformer, and Scaler as .pkl files using joblib for seamless deployment.

9. Deployment (Streamlit)
Developed an interactive web application allowing users to draw digits on a digital canvas and receive real-time predictions.

10. Infrastructure & Governance
      Implemented GitHub Branch Protection Rules and Pull Request Reviews to simulate a professional production environment.

**Installation & Usage**

1. Clone the Repository
Bash
**git clone https://github.com/ShajeeaMinnath/Digit-Recognition-ML.git
cd Digit-Recognition-ML**
2. Install Dependencies
Bash
**pip install streamlit streamlit-drawable-canvas joblib scikit-image opencv-python scikit-learn**
3. Run the Application
Bash
**streamlit run app.py**


**Results Summary**

Feature Descriptor: HOG (9 orientations)

Classifier: SVM (C=10, Gamma='scale')

Accuracy: ~98.2%

Inference Time: < 0.1s per digit
