import tensorflow as tf
import numpy as np
from PIL import Image, ImageFont, ImageDraw
import aidetector

import matplotlib.pyplot as plt


def main(downloaded_image):
    weights_folder = "../main"
    realFake, probabilities = aidetector.main(downloaded_image, weights_folder)

    accuracy = probabilities[0][0] if probabilities[0][0] > probabilities[0][1] else probabilities[0][1]

    print("Real or Fake:", realFake)
    print("Accuracy:", accuracy)

    return realFake, accuracy


def generateWaterMark(imageFile, fakeOrNotString, accuracy):
    #Open and display original image
    img = Image.open(imageFile)
    # plt.imshow(img)
    # img.show()

# create a copy for watermarking
    watermark_image = img.copy()
    draw = ImageDraw.Draw(watermark_image)

    w, h = img.size
    cornerRatio = 8
    x = int(w * (1 - 1/cornerRatio))
    y = int(h * (1 - 1/cornerRatio))

    font_size = 50

    # Use default font instead of arial.ttf
    try:
        font = ImageFont.load_default(font_size)
        # font = ImageFont.truetype("arial.ttf",font_size)
    except Exception as e:
        print(f"Error loading font: {e}")
        return

    # Text to display
    watermark_text = f"{fakeOrNotString} ({accuracy*100:.4f}%)"

    ## todo: create the effect white and black font "outline"
    draw.text((x, y), watermark_text, fill=(0, 0, 0, 128), font=font, anchor='ra')

    plt.figure(figsize=(6, 6))
    plt.imshow(watermark_image)
    plt.axis('off')
    plt.tight_layout(pad=0)
    plt.show()

    index = len(imageFile) - 1
    cursor = imageFile[index]
    while cursor != "/":
        index -= 1;
        cursor = imageFile[index];
    imageFileName = imageFile[index:]

    watermark_image.save("../images/Evaluated/" + imageFileName)

    # # Add black watermark
    # draw.text((x, y), fakeOrNotString, fill=(0, 0, 0,128), font=font, anchor='ms')
    #
    # # Plot the image cleanly
    # plt.figure(figsize=(6, 6))
    # plt.imshow(watermark_image)
    # plt.axis('off')  # Turn off axes
    # plt.tight_layout(pad=0)  # Remove padding
    # plt.show()




   #
   #  # Create figure for a subplot
   #  plt.figure(figsize=(12, 6))
   #
   #  # Add black watermark
   #  draw.text((x, y), fakeOrNorString, fill=(0, 0, 0), font=font, anchor='ms')
   #  plt.subplot(1, 2, 1)
   # # plt.title("black text")
   #  plt.imshow(watermark_image)

    # Reset watermark image for white text
    # watermark_image = img.copy()
    # draw = ImageDraw.Draw(watermark_image)

    # Add white watermark
    # draw.text((x, y), "puppy", fill=(255, 255, 255), font=font, anchor='ms')
    # plt.subplot(1, 2, 2)
    # plt.title("white text")
    # plt.imshow(watermark_image)

    plt.show()



















# NEXT STEPS TEST ACCURACY
# Step 1: Load and preprocess the image
def preprocess_image(image_path, target_size=(224, 224)):
    img = Image.open(image_path).convert("RGB").resize(target_size)
    img_array = np.array(img) / 255.0  # Normalize pixel values
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

# Step 2: Predict the AI probability
def predict_ai_probability(model, image_path):
    img_processed = preprocess_image(image_path, target_size=model.input_shape[1:3])
    prediction = model.predict(img_processed)
    probability = float(prediction[0][0]) * 100  # Make sure it's a clean float
    return probability

# # Step 3: Main function with file protection
# def main(model_path, image_path):
#     # Check if model exists
#     if not os.path.exists(model_path):
#         raise FileNotFoundError(f"❌ Model not found: {model_path}")
#
#     # Check if image exists
#     if not os.path.exists(image_path):
#         raise FileNotFoundError(f"❌ Image not found: {image_path}")
#
#     # Load model
#     model = tf.keras.models.load_model(model_path)
#
#     # Predict probability
#     probability = predict_ai_probability(model, image_path)
#
#     # Print result
#     print(f"This picture is {probability:.2f}% AI-generated.")
#
# Example usage
# main("path_to_model/your_model.h5", "path_to_image/doggy.jpeg")










if __name__ == "__main__":
    main()