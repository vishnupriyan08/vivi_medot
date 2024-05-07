from flask import Flask, render_template, request, jsonify
import numpy as np
from flask import send_file
from flask import send_file, session
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from werkzeug.utils import secure_filename
import os
import logging
import base64

app = Flask(__name__)

# Define the upload folder
app.config["UPLOAD_FOLDER"] = "temp"

# Load the trained model
model_path = "model/skin_disease_model.h5"
model_weight_path = "model/skin_model_.weights.h5"

try:
    model = load_model(model_path)
    model.load_weights(model_weight_path)
    logging.info("Model loaded successfully.")
except Exception as e:
    logging.error(f"Error loading model: {e}")

# Define the categories
categories = [
    "BA- cellulitis",
    "BA-impetigo",
    "FU-athlete-foot",
    "FU-nail-fungus",
    "FU-ringworm",
    "PA-cutaneous-larva-migrans",
    "VI-chickenpox",
    "VI-shingles",
]

# Define symptoms and precautions dictionaries
disease_info = {
    "PA-cutaneous-larva-migrans": {
        "Symptoms": "Itching, rash, raised winding tracks under the skin.",
        "Precautions": "Avoid walking barefoot in areas known to be contaminated with hookworm larvae. Wear protective footwear."
    },
    "FU-ringworm": {
        "Symptoms": "Red, itchy, ring-shaped patches on the skin.",
        "Precautions": "Keep the affected area clean and dry. Avoid sharing personal items like towels and clothing."
    },
    "VI-chickenpox": {
        "Symptoms": "Itchy rash, red spots, blisters, fever.",
        "Precautions": "Keep infected individuals away from others, use antiviral medications, and maintain good hygiene."
    },
    "BA-impetigo": {
        "Symptoms": "Red sores, blisters, lesions on the face, neck, hands.",
        "Precautions": "Keep the affected area clean, use antibiotics, avoid scratching, and wash hands frequently."
    },
    "BA-cellulitis": {
        "Symptoms": "Redness, swelling, warmth, pain or tenderness around the affected skin area.",
        "Precautions": "Use prescribed antibiotics, elevate the affected area, and keep it clean and dry."
    },
    "FU-athlete-foot": {
        "Symptoms": "Itching, burning, redness, cracking, peeling of the skin on the feet.",
        "Precautions": "Keep feet clean and dry, wear clean socks, use antifungal medications."
    },
    "VI-shingles": {
        "Symptoms": "Painful rash, blisters, itching, tingling, burning sensations.",
        "Precautions": "Use antiviral medications, keep the rash clean and dry, avoid scratching."
    },
    "FU-nail-fungus": {
        "Symptoms": "Thickened, brittle, yellow, discolored nails.",
        "Precautions": "Keep nails clean and dry, wear proper footwear, use antifungal treatments."
    }
}



@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get the image file from the request
        img_file = request.files["file"]

        # Check if the file is selected
        if img_file.filename == "":
            return jsonify({"error": "No file selected"})

        # Get the filename and sanitize it
        filename = secure_filename(img_file.filename)

        # Define the path to save the image
        temp_img_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)

        # Save the image
        img_file.save(temp_img_path)

        # Load and preprocess the image
        img = image.load_img(temp_img_path, target_size=(150, 150))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0

        # Make prediction
        prediction = model.predict(img_array)
        predicted_class = categories[np.argmax(prediction)]
        confidence = round(np.max(prediction) * 100, 2)

        # Get symptoms and precautions based on predicted class
        symptoms = disease_info[predicted_class]["Symptoms"]
        precautions = disease_info[predicted_class]["Precautions"]

        # Encode image to base64 for displaying in HTML
        with open(temp_img_path, "rb") as img_file:
            encoded_image = base64.b64encode(img_file.read()).decode("utf-8")

        # Delete the temporary image file
        os.remove(temp_img_path)

        # Pass prediction data to the HTML template
        return render_template("predicted.html", predicted=predicted_class, image=encoded_image, symptoms=symptoms, precautions=precautions)
         
        with open("predicted_report.html", "w") as html_file:
            html_file.write(predicted_html)

        # Redirect to the download route with the predicted class as a query parameter
        return redirect(url_for('download_report', predicted=predicted_class))
         
    except Exception as e:
        logging.error(f"Error processing image: {e}")
        return jsonify({"error": "Error processing image"}), 500
from flask import send_file


@app.route("/download_report")
def download_report():
    # Get predicted_class from session
    predicted_class = session.get('predicted_class')

    if predicted_class:
        # Get symptoms and precautions based on predicted class
        symptoms = disease_info[predicted_class]["Symptoms"]
        precautions = disease_info[predicted_class]["Precautions"]

        # Generate the content of the report
        report_content = f"Predicted Disease: {predicted_class}\n\nSymptoms:\n{symptoms}\n\nPrecautions:\n{precautions}"
        
        # Define the filename for the report
        report_filename = "skin_disease_report.txt"

        # Save the report content to a temporary file
        with open(report_filename, "w") as report_file:
            report_file.write(report_content)
        
        # Send the file for download
        return send_file(report_filename, as_attachment=True)
    else:
        return "Predicted class not found in session."


if __name__ == "__main__":
    app.run(debug=True)
