from flask import Flask, request, render_template, url_for
from keras.layers import LeakyReLU
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import numpy as np
import os
import base64
from io import BytesIO

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

app = Flask(__name__)

from flask import Flask

app = Flask(__name__)

# Define the custom filter
@app.template_filter('replace_spaces')
def replace_spaces(value):
    return value.replace(' ', '-').lower()

# Load the saved model
model = load_model("pokemon_cnn_model.keras", compile=False, custom_objects={'LeakyReLU': LeakyReLU})

# Compile the model manually
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Define image dimensions
img_width, img_height = 150, 150

import requests

def get_pokemon_data(pokemon_name):
    url = f'https://pokeapi.co/api/v2/pokemon/{pokemon_name.lower()}'
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        dex_number = data['id']
        types = [t['type']['name'] for t in data['types']]
        base_stats = {stat['stat']['name']: stat['base_stat'] for stat in data['stats']}
        return {
            'dex_number': dex_number,
            'types': types,
            'base_stats': base_stats
        }
    else:
        return None

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

# Function to convert image to base64
# Function to convert image to base64
def image_to_base64(image):
    if image.mode == 'RGBA':
        image = image.convert('RGB')  # Convert RGBA to RGB if necessary
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
    return img_base64


@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['file']
    img = Image.open(file.stream)
    img = img.resize((img_width, img_height))

    # Convert RGBA to RGB if necessary
    if img.mode == 'RGBA':
        img = img.convert('RGB')
    
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    
    # Save the uploaded image
    uploaded_image_path = os.path.join('static', 'uploaded_image.jpg')
    img.save(uploaded_image_path)
    
    predicted_class, similarity_percentage = predict_pokemon(img_array)
    predicted_pokemon = get_prediction_details(predicted_class)

    # Fetch data from PokeAPI
    pokemon_data = get_pokemon_data(predicted_pokemon)
    
    # Get the representative image from the dataset
    representative_image_path = get_representative_image(predicted_pokemon)
    if representative_image_path:
        predicted_image = Image.open(representative_image_path)
        max_width = 150  
        max_height = int(predicted_image.height * (max_width / predicted_image.width))
        predicted_image = predicted_image.resize((max_width, max_height))
        
        # Convert the predicted image to base64
        predicted_image_base64 = image_to_base64(predicted_image)
    else:
        predicted_image_base64 = None  # Use None if no image is found

    # Render the template with additional Pokémon data
    return render_template(
        'index.html',
        predicted_pokemon=predicted_pokemon,
        similarity_percentage=similarity_percentage,
        image_url=url_for('static', filename='uploaded_image.jpg'),
        predicted_image_base64=predicted_image_base64,
        dex_number=pokemon_data['dex_number'] if pokemon_data else 'N/A',
        types=pokemon_data['types'] if pokemon_data else [],
        base_stats=pokemon_data['base_stats'] if pokemon_data else {}
    )



@app.route("/")
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
