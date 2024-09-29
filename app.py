from flask import Flask, request, render_template, url_for
from keras.layers import LeakyReLU
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import numpy as np
import os
import base64
from io import BytesIO
import requests

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

app = Flask(__name__)

from flask import Flask

app = Flask(__name__)

# Load the saved model
model = load_model("pokemon.keras", compile=False, custom_objects={'LeakyReLU': LeakyReLU})

# Compile the model manually
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Define image dimensions
img_width, img_height = 150, 150

import requests
import logging

# Global cache to store different types of Pokémon data
global_cache = {
    'pokemon_data': {},
    'species_data': {},
    'base_stats': {},
    'api_data': {}  # For caching individual API responses if needed
}

special_cases = {
    "deoxys": "deoxys-normal",
    "wormadam": "wormadam-plant",
    "giratina": "giratina-altered",
    "shaymin": "shaymin-land",
    "basculin": "basculin-red-striped",
    "basculegion": "basculegion-male",
    "darmanitan": "darmanitan-standard",
    "tornadus": "tornadus-incarnate",
    "thundurus": "thundurus-incarnate",
    "landorus": "landorus-incarnate",
    "enamorus": "enamorus-incarnate",
    "keldeo": "keldeo-ordinary",
    "meloetta": "meloetta-aria",
    "meowstic": "meowstic-male",
    "aegislash": "aegislash-shield",
    "pumpkaboo": "pumpkaboo-average",
    "gourgeist": "gourgeist-average",
    "zygarde": "zygarde-50",
    "oricorio": "oricorio-baile",
    "lycanroc": "lycanroc-midday",
    "wishiwashi": "wishiwashi-solo",
    "minior": "minior-red-meteor",
    "mimikyu": "mimikyu-disguised",
    "toxtricity": "toxtricity-amped",
    "eiscue": "eiscue-ice",
    "indeedee": "indeedee-male",
    "urshifu": "urshifu-single-strike",
    "oinkologne": "oinkologne-male",
    "maushold": "maushold-family-of-four",
    "squawkabilly": "squawkabilly-green-plumage",
    "palafin": "palafin-zero",
    "tatsugiri": "tatsugiri-curly",
    "dudunsparce": "dudunsparce-two-segment",
    "farfetch'd": "farfetchd",
    "sirfetch'd": "sirfetchd",
    "mime jr": "mime-jr",
    "mr. mime": "mr-mime",
    "mr. rime": "mr-rime",
    "nidoran♀": "nidoran-f",
    "nidoran♂": "nidoran-m",
    "flabébé": "flabebe"
}

special_pokemon = [
    "type-null",
    "jangmo-o",
    "hakamo-o",
    "kommo-o",
    "tapu lele",
    "tapu bulu",
    "tapu fini",
    "tapu koko",
    "chi-yu",
    "chien-pao",
    "ting-lu",
    "wo-chien",
    "great tusk",
    "scream tail",
    "brute bonnet",
    "flutter mane",
    "slither wing",
    "sandy shocks",
    "roaring moon",
    "walking wake",
    "gouging fire",
    "raging bolt",
    "iron treads",
    "iron bundle",
    "iron hands",
    "iron jugulis",
    "iron moth",
    "iron thorns",
    "iron valiant",
    "iron leaves",
    "iron boulder",
    "iron crown"
]

def format_pokemon_name(name):
    formatted_name = special_cases.get(name.lower(), name.replace(' ', '-').lower())
    return formatted_name

# Function to fetch data with caching
def fetch_data_with_cache(url):
    if url in global_cache['api_data']:
        return global_cache['api_data'][url]  # Return from cache if available
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        global_cache['api_data'][url] = data  # Cache the result
        return data
    return None

def get_pokemon_data(pokemon_name):
    base_name = pokemon_name.lower()
    pokemon_data = {}

    if base_name in global_cache['pokemon_data']:
        return global_cache['pokemon_data'][base_name]

    # Fetch base form data
    base_url = f'https://pokeapi.co/api/v2/pokemon/{base_name}'
    base_data = fetch_data_with_cache(base_url)
    
    if base_data:
        pokemon_data['dex_number'] = base_data['id']
        pokemon_data['types'] = [t['type']['name'] for t in base_data['types']]  # Base types
        pokemon_data['base_stats'] = {stat['stat']['name']: stat['base_stat'] for stat in base_data['stats']}
        pokemon_data['sprite_url'] = base_data['sprites']['other']['official-artwork']['front_default']
        pokemon_data['forms'] = []

        # Add base form as the first form
        pokemon_data['forms'].append({
            'form_name': base_name,
            'sprite_url': base_data['sprites']['other']['official-artwork']['front_default'],
            'types': pokemon_data['types']  # Storing types for the base form
        })
        
        # Fetch species data to get form information
        species_url = f'https://pokeapi.co/api/v2/pokemon-species/{base_name}'
        species_data = fetch_data_with_cache(species_url)

        if species_data:
            forms = species_data['varieties']
            for form in forms:
                form_name = form['pokemon']['name']
                if form_name != base_name:
                    form_url = form['pokemon']['url']
                    form_data = fetch_data_with_cache(form_url)
                    if form_data:
                        form_sprite_url = form_data['sprites']['other']['official-artwork']['front_default']
                        form_types = [t['type']['name'] for t in form_data['types']]  # Form-specific types
                        # Append form if not already added
                        pokemon_data['forms'].append({
                            'form_name': form_name,
                            'sprite_url': form_sprite_url,
                            'types': form_types  # Store types for the specific form
                        })

        # Store the result in the global cache
        global_cache['pokemon_data'][base_name] = pokemon_data

    return pokemon_data

def get_pokemon_species_data(pokemon_name):
    # Convert the name to lowercase and replace spaces with hyphens
    formatted_name = pokemon_name.lower().replace(' ', '-')

    if '-' in formatted_name and formatted_name.split('-')[0] in special_cases:
        base_name = formatted_name.split('-')[0]  
    elif formatted_name in special_pokemon:
        base_name = formatted_name  
    else:
        base_name = formatted_name 

    # Debug: Log the base name
    logging.debug(f"Base name used for species data: {base_name}")

    # Check cache for species data using the base name
    if base_name in global_cache['species_data']:
        return global_cache['species_data'][base_name]

    url = f'https://pokeapi.co/api/v2/pokemon-species/{base_name}'
    data = fetch_data_with_cache(url)

    if data:
        # Extract necessary fields from the fetched data
        evolution_chain_url = data['evolution_chain']['url']
        forms = [variety['pokemon']['name'] for variety in data['varieties']]
        gender_rate = data['gender_rate']
        
        # Structure species data
        species_data = {
            'forms': forms,
            'evolution_chain_url': evolution_chain_url,
            'gender_rate': gender_rate
        }

        # Cache the fetched species data for future use
        global_cache['species_data'][base_name] = species_data
        return species_data
    else:
        # Log error if data fetching fails
        logging.error(f"Error fetching species data for {base_name}")
        return None

def get_pokemon_base_stats(pokemon_name):
    base_name = pokemon_name.lower()
    base_stats_by_form = []

    # Initialize cache for the Pokémon if not present
    if base_name not in global_cache:
        global_cache[base_name] = {}

    # Check if base stats are already cached
    if 'base_stats' in global_cache[base_name]:
        return global_cache[base_name]['base_stats']

    fetched_forms = set()  # Track forms that have been fetched to avoid duplication

    # Fetch default Pokémon data
    url = f'https://pokeapi.co/api/v2/pokemon/{base_name}'
    response = requests.get(url)

    if response.status_code == 200:
        base_data = response.json()
        default_base_stats = {stat['stat']['name']: stat['base_stat'] for stat in base_data['stats']}
        default_types = [t['type']['name'] for t in base_data['types']]
        default_sprite_url = base_data['sprites']['other']['official-artwork']['front_default']
        
        # Append only if the form has not been fetched already
        if base_name not in fetched_forms:
            base_stats_by_form.append({
                'form_name': base_name,
                'base_stats': default_base_stats,
                'types': default_types,  # Fetch the types for the base form
                'sprite_url': default_sprite_url
            })
            fetched_forms.add(base_name)

        # Fetch species data for additional forms
        species_url = f'https://pokeapi.co/api/v2/pokemon-species/{base_name}'
        species_response = requests.get(species_url)

        if species_response.status_code == 200:
            species_data = species_response.json()

            for variety in species_data['varieties']:
                form_name = variety['pokemon']['name']
                if form_name not in fetched_forms:  # Fetch form data only if not already fetched
                    form_response = requests.get(variety['pokemon']['url'])
                    if form_response.status_code == 200:
                        form_data = form_response.json()
                        form_base_stats = {stat['stat']['name']: stat['base_stat'] for stat in form_data['stats']}
                        form_types = [t['type']['name'] for t in form_data['types']]  # Fetch form-specific types
                        form_sprite_url = form_data['sprites']['other']['official-artwork']['front_default']
                        
                        # Append the form to the list
                        base_stats_by_form.append({
                            'form_name': form_name,
                            'base_stats': form_base_stats,
                            'types': form_types,  # Include form-specific types
                            'sprite_url': form_sprite_url
                        })
                        fetched_forms.add(form_name)

    # Store the fetched base stats in cache
    global_cache[base_name]['base_stats'] = base_stats_by_form

    return base_stats_by_form

def get_evolution_chain(evolution_chain_url):
    evolution_data = fetch_data_with_cache(evolution_chain_url)
    
    if evolution_data:
        evolution_chain = evolution_data['chain']
        evolutions = []
        
        def parse_evolution(evolution_chain):
            pokemon_name = evolution_chain['species']['name']
            url = f'https://pokeapi.co/api/v2/pokemon/{pokemon_name}'
            pokemon_data = fetch_data_with_cache(url)
            if pokemon_data:
                sprite_url = pokemon_data['sprites']['other']['official-artwork']['front_default']
                evolutions.append({
                    'name': pokemon_name,
                    'sprite_url': sprite_url
                })
            for evo in evolution_chain['evolves_to']:
                parse_evolution(evo)
        
        parse_evolution(evolution_chain)
        return evolutions
    return []

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
    "Training",
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
    folder_path = os.path.join('Training', pokemon_folder)
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
def image_to_base64(image):
    if image.mode == 'RGBA':
        image = image.convert('RGB')  # Convert RGBA to RGB if necessary
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
    return img_base64

import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG)

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['file']
    img = Image.open(file.stream)
    img = img.resize((img_width, img_height))

    if img.mode == 'RGBA':
        img = img.convert('RGB')

    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)

    uploaded_image_path = os.path.join('static', 'uploaded_image.jpg')
    img.save(uploaded_image_path)

    predicted_class, similarity_percentage = predict_pokemon(img_array)
    predicted_pokemon = get_prediction_details(predicted_class)
    
    predicted_pokemon_formatted = format_pokemon_name(predicted_pokemon)

    # Log the predicted Pokémon and formatted name
    logging.debug(f"Predicted Pokémon: {predicted_pokemon}, Formatted: {predicted_pokemon_formatted}")

    # Try to fetch Pokémon species data
    try:
        species_data = get_pokemon_species_data(predicted_pokemon_formatted)
        forms = species_data.get('forms', []) if species_data else []
        evolution_chain = get_evolution_chain(species_data.get('evolution_chain_url', '')) if species_data else []
        logging.debug(f"Species Data: {species_data}, Forms: {forms}, Evolution Chain: {evolution_chain}")
    except Exception as e:
        logging.error(f"Error fetching species data: {e}")
        species_data, forms, evolution_chain = None, [], []
        
    # Try to fetch Pokémon data from /pokemon/
    try:
        pokemon_data = get_pokemon_data(predicted_pokemon_formatted)
        logging.debug(f"Pokémon Data: {pokemon_data}")
    except KeyError as e:
        logging.error(f"Error fetching Pokémon data: {e}")
        pokemon_data = None
    
    # Handle if pokemon_data is missing (i.e., special cases)
    if not pokemon_data:
        base_stats_by_form = get_pokemon_base_stats(predicted_pokemon_formatted)
        if species_data and base_stats_by_form:
            pokemon_data = {
                'dex_number': species_data.get('id', 'N/A'),
                'types': [],
                'base_stats': base_stats_by_form,
                'sprite_url': base_stats_by_form[0].get('sprite_url', None) if base_stats_by_form else None
            }
            logging.debug(f"Handling special case Pokémon data: {pokemon_data}")
        else:
            logging.warning(f"Failed to handle special case for {predicted_pokemon_formatted}")
    else:
        base_stats_by_form = get_pokemon_base_stats(predicted_pokemon_formatted)
    
    if base_stats_by_form is None:
        base_stats_by_form = []

    # Ensure there are no duplicate forms
    forms = list(dict.fromkeys(forms))  # Removes duplicates

    # Fetch base stats for each unique form
    base_stats_by_form = []
    fetched_stats = set()  # Set to track fetched stats and avoid duplicates
    for form in forms:
        form_stats = get_pokemon_base_stats(form)
        for stats in form_stats:
            if stats['form_name'] not in fetched_stats:  # Check if form's stats were already added
                base_stats_by_form.append(stats)
                fetched_stats.add(stats['form_name'])

    # Fetch representative image (if available)
    representative_image_path = get_representative_image(predicted_pokemon)
    if representative_image_path:
        predicted_image = Image.open(representative_image_path)
        max_width = 150
        max_height = int(predicted_image.height * (max_width / predicted_image.width))
        predicted_image = predicted_image.resize((max_width, max_height))
        predicted_image_base64 = image_to_base64(predicted_image)
    else:
        predicted_image_base64 = None
            
    return render_template(
        'index.html',
        predicted_pokemon=predicted_pokemon,
        similarity_percentage=similarity_percentage,
        image_url=url_for('static', filename='uploaded_image.jpg'),
        predicted_image_base64=predicted_image_base64,
        dex_number=pokemon_data.get('dex_number', 'N/A') if pokemon_data else 'N/A',
        types=pokemon_data.get('types', []) if pokemon_data else [],
        base_stats_by_form=base_stats_by_form,  # Ensure all forms' stats are passed
        forms=forms,
        evolution_chain=evolution_chain
    )


@app.route("/")
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)