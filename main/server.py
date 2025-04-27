# Anna Scribner
# Michael Gilbert
# Muskan Gupta
# Roger Tang

from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
import tempfile
from main import main

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})  # <-- Important

@app.route('/serverTest', methods=['POST'])
def analyze_image():
    data = request.json
    image_url = data.get('image_url')             # receive picture URL
    file_path = download(image_url)               # download  & save to a file
    fakeOrNotString, accuracy = main(file_path)   # send to main (call main with the file)

    print(f"Received image URL: {image_url}")

    accuracy = str(accuracy)
    accuracyNum = float(accuracy[7:-1])

    result = {"result": fakeOrNotString, "accuracy": format(accuracyNum, ".5f")}
    return jsonify(result)



def download(image_url):
    url = image_url # image URL here

    response = requests.get(url)

    #create a temporary file
    fd, path = tempfile.mkstemp()

    if response.status_code == 200:
        with open(path, "wb") as f:
            f.write(response.content)
        print("Image downloaded successfully!", path)
    else:
        print("Failed to download image. Status code:", response.status_code)

    return path

if __name__ == '__main__':
    app.run(debug=True)
