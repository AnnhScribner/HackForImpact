# Anna Scribner, Michael Gilbert, Muskan Gupta, Roger Tang

from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
import tempfile
from inference import run_inference  # Import the prediction function

app = Flask(__name__)

# ✅ Enable CORS for all routes and allow credentials (useful for frontend access)
CORS(app, supports_credentials=True, resources={r"/*": {"origins": "*"}})

@app.route('/serverTest', methods=['POST'])
def analyze_image():
    """
    POST endpoint that receives an image URL, downloads it,
    runs AI detection, and returns the result as JSON.
    """
    data = request.json
    image_url = data.get('image_url')
    print(f"image url: {image_url}")
    file_path = download(image_url)

    label, accuracy = run_inference(file_path)

    print(f"Received image URL: {image_url}")
    print(f"Prediction: {label}, Accuracy: {accuracy:.5f}")

    return jsonify({
        "result": label,
        "accuracy": format(accuracy, ".5f")
    })


def download(image_url):
    """
    Downloads an image from a URL and saves it as a temporary file.
    Returns the path to the saved file.
    """
    response = requests.get(image_url)
    print(response)
    fd, path = tempfile.mkstemp()

    if response.status_code == 200:
        with open(path, "wb") as f:
            f.write(response.content)
        print("✅ Image downloaded successfully!", path)
        return path
    else:
        print("❌ Failed to download image. Status code:", response.status_code)
        return None




if __name__ == '__main__':
    app.run(debug=True)
