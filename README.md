# Pokémon Image Classification with Convolutional Neural Networks (CNN)

This Flask web application allows users to upload an image of a Pokémon and predicts its name using a pre-trained neural network model (CNN). Additionally, it fetches detailed information about the predicted Pokémon, such as its forms, base stats, and evolution chain, by integrating with **PokeAPI**.

# Jupyter Notebook

For those interested in exploring the model and its training process, a Jupyter Notebook file is included in the repository. You can use it to see how the image classification model was developed and tested.

## Features
- Upload an image of a Pokémon and receive a prediction of its species.
- Fetches detailed Pokémon information from **PokeAPI**, including types, forms, base stats, and evolution chains.
- Displays predicted Pokémon along with a representative image, forms, and evolution data.
- Handles special Pokémon forms (e.g., "Deoxys" or "Lycanroc") and caches API responses for improved performance.

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/nyain/Pokedex.git
    ```

2. Install required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Ensure the pre-trained model file `pokemon.keras` is in the root directory.

4. Create a `static/` directory if it doesn't exist:
    ```bash
    mkdir static
    ```
5. Make sure you already have the dataset. You can scrape the dataset from this [repository](https://github.com/nyain/Pokemon-Image-Scraper)

## Usage

1. Start the Flask app:
    ```bash
    python app.py
    ```

2. Open a browser and navigate to `http://127.0.0.1:5000/`.

3. Upload a Pokémon image (PNG, JPG, JPEG) and view the prediction along with the Pokémon’s base stats, types, forms, and evolution chain.

## Model Information
- The app uses a pre-trained **Keras** model (`pokemon.keras`) with **LeakyReLU** activation.
- The model is compiled using the **Adam** optimizer and categorical cross-entropy loss.

## API Integration
- The app utilizes **PokeAPI** to fetch real-time data about Pokémon species, forms, types, and evolution chains.
- Special Pokémon forms and alternate names (like "Mr. Mime" or "Giratina-Altered") are handled via custom logic.

## Notes
- The app resizes and preprocesses the uploaded image to `150x150` before making predictions.
- Cached API responses improve performance by reducing redundant requests.
