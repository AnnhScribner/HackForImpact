# Hack4Impact Hackathon
## RealSeal- Real/AI Image Detector 


## Setup

### Load the model
1. Download the model file for our AI vs real image detection here: https://drive.usercontent.google.com/download?id=1Wb1z9d_Nr4nKKYaevBym684xYGpvJRhY&export=download&authuser=0
2. Save it in `root/main` in the project files

### To add the Chrome extension to your browser
4. In your Chrome browser, open 'Manage Extensions' or go to `chrome://extensions/`
5. Enable dev mode
6. Click on 'Load Unpacked'
7. Click the `extension` folder inside this project to load it

### To run the Demo
8. Start the backend server by running the `server.py` in any IDE
9. Host the demo website by running `python3 -m http.server` in the project's root directory
10. Open/Refresh the `http://localhost:8000/website/realimages.html` sample link in Chrome
11. Right click on the web page, click `inspect` and switch to Console for the detection results
12. For each image, it will diplay if it's real or AI with how confidence we are about the results
