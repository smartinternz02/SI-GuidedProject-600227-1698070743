from flask import Flask, render_template, request
from keras.models import load_model
from keras.preprocessing import image
from keras.applications.vgg19 import preprocess_input
import numpy as np

app = Flask(__name__)

# Load the pre-trained model
VGG19_model = load_model('fakelogo.h5')

# Define class names
class_names = [
    'Adidas', 'Amazon', 'Android', 'Apple', 'Ariel', 'BMW', 'Bic', 'Burger King',
    'Cadbury', 'Chevrolet', 'Chrome', 'Coca Cola', 'Cowbell', 'Dominos', 'Fila',
    'Gillette', 'Google', 'Goya oil', 'Guinness', 'Heinz', 'Honda', 'Hp', 'Huawei',
    'Instagram', 'Kfc', 'Krisspy Kreme', 'Lays', 'Levis', 'Lg', 'Lipton', 'Mars', 'Marvel', 'McDonald',
    'Mercedes Benz', 'Microsoft', 'MnM', 'Mtn', 'Mtn dew', 'NASA', 'Nescafe', 'Nestle', 'Nestle milo',
    'Netflix', 'Nike', 'Nutella', 'Oral b', 'Oreo', 'Pay pal', 'Peak milk', 'Pepsi', 'PlayStation',
    'Pringles', 'Puma', 'Reebok', 'Rolex', 'Samsung', 'Sprite', 'Starbucks', 'Tesla', 'Tiktok',
    'Twitter', 'YouTube', 'Zara'
]

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Get the uploaded image file
        file = request.files['file']
        if file:
            # Save the uploaded image
            img_path = 'uploads/uploaded_image.png'
            file.save(img_path)

            # Preprocess the image
            img = image.load_img(img_path, target_size=(244, 244))
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            x = preprocess_input(x)

            # Make a prediction
            preds = VGG19_model.predict(x)

            # Get the predicted class
            predicted_class = class_names[np.argmax(preds)]

            return render_template('index.html', prediction=predicted_class, image_path=img_path)

    return render_template('index.html', prediction=None, image_path=None)

if __name__ == '__main__':
    app.run(debug=True)
