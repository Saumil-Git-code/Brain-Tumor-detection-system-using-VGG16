# Brain Tumor Detection Web App

A Flask-based web application for classifying brain MRI images into one of four categories:
- **glioma**
- **meningioma**
- **notumor**
- **pituitary**

The app uses a pre-trained CNN model (`VGG16`) to predict tumor type from uploaded MRI images.

## Project Structure

- `app.py` - Flask application entry point
- `model.h5` - Pre-trained TensorFlow/Keras model used for inference
- `Cancer detection.ipynb` - Notebook used for model training and experimentation
- `templates/index.html` - Web page template
- `static/styles.css` - CSS styling for the app
- `uploads/` - Folder where uploaded images are temporarily saved
- `MRI images/` - Dataset folders for training and testing
  - `Training/` and `Testing/` containing subfolders for each class

## Features

- Upload MRI images in JPG, PNG, or BMP format
- Returns predicted tumor type and confidence score
- Displays the uploaded image alongside the prediction

## Requirements

- Python 3.8+
- Flask
- TensorFlow / Keras
- NumPy
- PIP
- sklearn
- Matplotlib

## Setup

1. Create and activate a virtual environment (recommended):

   ```bash
   python -m venv venv
   .\venv\Scripts\activate
   ```

2. Install required packages:

   ```bash
   pip install flask tensorflow numpy werkzeug
   ```

3. Ensure `model.h5` is present in the project root.

4. Run the app:

   ```bash
   python app.py
   ```

5. Open your browser at `http://127.0.0.1:5000/`.

## Usage

- Click **Choose MRI image** to upload an image file
- Submit the form to predict the tumor category
- The result page shows:
  - predicted class
  - confidence percentage
  - uploaded MRI image

## Notes

- The app resizes images to `128x128` before prediction.
- Uploaded images are stored in the `uploads/` directory.
- Maximum upload size is limited to 16 MB.
- If the model fails to load, the app prints an error and prediction will return `Model not loaded`.

## Dataset Layout

The `MRI images/` directory is organized as:

- `MRI images/Training/glioma/`
- `MRI images/Training/meningioma/`
- `MRI images/Training/notumor/`
- `MRI images/Training/pituitary/`
- `MRI images/Testing/glioma/`
- `MRI images/Testing/meningioma/`
- `MRI images/Testing/notumor/`
- `MRI images/Testing/pituitary/`

## Model File

The trained model (`model.h5`) is not included in this repository because it exceeds GitHub's file size limit.

To run the Flask application:

1. Train the model using the provided notebook/script.
2. Save the trained model as `model.h5`.
3. Place `model.h5` in the project root directory.
4. Run `app.py`.

## License

This project is provided as-is for educational and development use.
