AI Dental Anomaly Detector
This is a web-based application that uses a YOLOv10 machine learning model to detect potential anomalies in dental panoramic X-ray images. The application provides a simple, user-friendly interface for uploading an image and viewing the AI-analyzed results with bounding boxes drawn around detected issues.

Key Features
AI-Powered Detection: Leverages a fine-tuned YOLOv10 model to identify dental anomalies.

Supported Detections: The model is trained to detect Cavity, Filling, Implant, and Impacted teeth.

Web Interface: A clean, responsive, and easy-to-use interface built with HTML, Tailwind CSS, and JavaScript.

Backend Server: A lightweight Node.js and Express server to handle image uploads and processing.

Side-by-Side Comparison: Displays the original and processed images for clear analysis.

Technology Stack
Machine Learning: Python, PyTorch, Ultralytics YOLOv10, OpenCV

Backend: Node.js, Express.js

Frontend: HTML, Tailwind CSS, JavaScript, SweetAlert2

Getting Started
Follow these instructions to set up and run the project on your local machine.

Prerequisites
Before you begin, ensure you have the following software installed on your system:

Git: To clone the repository. Download Git

Node.js and npm: Required for the web server. Download Node.js (npm is included).

Python: Version 3.9 or higher is recommended. Download Python

Setup and Installation Guide
Step 1: Clone the Repository

Open your terminal or command prompt and clone the project from GitHub:

git clone [https://github.com/your-username/your-repository-name.git](https://github.com/your-username/your-repository-name.git)
cd your-repository-name

(Replace your-username and your-repository-name with your actual GitHub details)

Step 2: Add Model Files

Place your trained model files into the model_files directory:

dental_yolov10_best.pt

class_mapping.pkl

The final structure should look like this:

/dental-detector-app/
|-- 📂 model_files/
|   |-- dental_yolov10_best.pt
|   |-- class_mapping.pkl
...

Step 3: Set Up the Python Environment

It is crucial to create a virtual environment to manage Python dependencies and avoid conflicts.

Create the virtual environment (this creates a venv folder in your project):

python -m venv venv

Activate the virtual environment:

On Windows:

.\venv\Scripts\activate

On macOS/Linux:

source venv/bin/activate

After activation, your terminal prompt should start with (venv).

Install Python dependencies:
With the environment active, install all required libraries from the requirements.txt file.

pip install -r python_inference/requirements.txt

Step 4: Set Up the Node.js Server

Install Node.js dependencies:
In the project's root directory (where package.json is located), run:

npm install

This will download all necessary packages (like Express) and create a node_modules folder.

Running the Application
Ensure the Python virtual environment is active. If you've opened a new terminal, you must activate it again (.\venv\Scripts\activate).

Start the server from the project's root directory:

node server.js

You should see the following message in your terminal:

Server is running on http://localhost:3000

Open your web browser and navigate to http://localhost:3000. You can now use the application.

How to Use
Go to http://localhost:3000 in your browser.

Drag and drop a dental panoramic X-ray image onto the upload area, or click to select a file from your computer.

Click the "Diagnose Image" button.

A loading pop-up will appear while the model processes the image.

The results will be displayed, showing your original image alongside the image with detected anomalies highlighted by bounding boxes.

Click "Analyze Another Image" to reset and upload a new file.

Disclaimer
This is an educational project and is not intended for real-world medical diagnosis. The AI model's predictions are not a substitute for professional medical advice from a qualified healthcare provider.