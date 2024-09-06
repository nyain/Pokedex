from flask import Flask, request, render_template, url_for
from keras.layers import LeakyReLU
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import numpy as np
import os

app = Flask(__name__)

# Load the saved model
model = load_model("pokemon_cnn_model_improved.h5", compile=False, custom_objects={'LeakyReLU': LeakyReLU})

# Compile the model manually
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Define image dimensions
img_width, img_height = 150, 150

# Load the train generator to get the class indices
from tensorflow.keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    rotation_range=20,       
    width_shift_range=0.2,   
    height_shift_range=0.2,  
    brightness_range=[0.8, 1.2],
    validation_split=0.2
)
train_generator = train_datagen.flow_from_directory(
    "PokemonData",
    target_size=(img_width, img_height),
    batch_size=32,
    class_mode='categorical',
    subset='training'
)

# Create 'static' directory if not exists
if not os.path.exists('static'):
    os.makedirs('static')

# Function to predict Pokémon from an image file
def predict_pokemon(img_array):
    img_array = img_array / 255.0
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction)
    return predicted_class, np.max(prediction) * 100

# Function to get the label (Pokémon name) of the predicted class
def get_prediction_details(predicted_class):
    predicted_pokemon = train_generator.class_indices
    predicted_pokemon = dict((v, k) for k, v in predicted_pokemon.items())
    predicted_pokemon = predicted_pokemon[predicted_class]
    return predicted_pokemon

# Function to get a representative image file from the predicted Pokémon folder
def get_representative_image(pokemon_folder):
    folder_path = os.path.join('PokemonData', pokemon_folder)
    try:
        files = os.listdir(folder_path)
        # Filter out non-image files and get the first image
        image_files = [f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if image_files:
            return os.path.join(folder_path, image_files[0])
    except FileNotFoundError:
        return None
    return None

# Route for image upload and prediction
@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['file']
    img = Image.open(file.stream)
    img = img.resize((img_width, img_height))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    
    # Save the uploaded image
    uploaded_image_path = os.path.join('static', 'uploaded_image.jpg')
    img.save(uploaded_image_path)
    
    predicted_class, similarity_percentage = predict_pokemon(img_array)
    predicted_pokemon = get_prediction_details(predicted_class)
    
    # Get the representative image from the dataset
    representative_image_path = get_representative_image(predicted_pokemon)
    if representative_image_path:
        # Save the predicted image
        predicted_image_path = os.path.join('static', 'predicted_image.jpg')
        predicted_image = Image.open(representative_image_path)
        predicted_image.save(predicted_image_path)
    else:
        predicted_image_path = url_for('static', filename='default_image.jpg')  # Use a default image if none found
    
    return render_template('index.html', predicted_pokemon=predicted_pokemon, similarity_percentage=similarity_percentage, image_url=url_for('static', filename='uploaded_image.jpg'), predicted_image_url=url_for('static', filename='predicted_image.jpg'))

@app.route("/")
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
