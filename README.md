### Pokémon Image Classification with Convolutional Neural Networks (CNN)

This project implements an image recognition and classification model using a Convolutional Neural Network (CNN) to identify and categorize images of Pokémon from Generation I. The application provides a user-friendly web interface, allowing users to upload Pokémon images and receive real-time predictions on the Pokémon's identity, alongside a similarity score.

**Dataset:** [Pokémon Classification Dataset](https://www.kaggle.com/datasets/lantian773030/pokemonclassification)

### Features:
- Predicts Pokémon species from Generation I using a trained CNN model.
- Displays both the uploaded image and the predicted Pokémon.
- Provides a similarity score to indicate the confidence of the prediction.

### Technologies Used:
- **Flask**: For building the web application and serving the model.
- **Keras/TensorFlow**: For the CNN model and image classification.
- **Pillow**: For image handling and processing.
- **Base64 Encoding**: For serving images directly from memory without storing them on disk.

### How to Run the Application:

#### Prerequisites:
Ensure the following are installed on your system:
- Python 3.8+
- Flask
- TensorFlow/Keras
- Pillow

#### Installation Steps:
1. **Clone the repository:**
   ```bash
   git clone <repository_url>
   cd <repository_directory>
   ```

2. **Install required packages:**
   Install the necessary Python dependencies by running:
   ```bash
   pip install -r requirements.txt
   ```

3. **Download and Prepare the Dataset:**
   - Download the [Pokémon Classification Dataset](https://www.kaggle.com/datasets/lantian773030/pokemonclassification).
   - Extract the dataset and place it in a folder named `PokemonData` in the project directory.

4. **Run the Flask Application:**
   Start the Flask server by running:
   ```bash
   python app.py
   ```

5. **Access the Application:**
   Open your web browser and navigate to:
   ```
   http://127.0.0.1:5000/
   ```
   From here, you can upload images of Pokémon and get predictions.

#### Notes:
- The application resizes uploaded images to fit the model's input dimensions.
- Images are processed in real-time, and no unnecessary files are saved to the server.

### Future Enhancements:
- Expand the dataset to include more Pokémon generations.
- Implement a REST API for broader integration.
