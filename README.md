### Pokémon Image Classification with Convolutional Neural Networks (CNN)

This project implements an image recognition and classification model using a Convolutional Neural Network (CNN) to identify and categorize images of Pokémon from Generations I through IX. The application provides a user-friendly web interface, allowing users to upload Pokémon images and receive real-time predictions, including detailed Pokémon information such as base stats, typing, and Pokédex number.

**Dataset:** To create the dataset, use the script provided in the [Pokémon Image Scraper repository]([https://github.com/<your-username>/<your-repo-name>](https://github.com/nyain/Pokemon-Image-Scraper)). The dataset is not included in this repository.

### Features:
- Predicts Pokémon species from Generations I through IX using a trained CNN model.
- Displays the uploaded image and the predicted Pokémon.
- Provides detailed Pokémon information, including:
  - Base stats
  - Typing
  - Pokédex number
- Includes a similarity score to indicate the confidence of the prediction.

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
   - Use the script from the [Pokémon Image Scraper repository]([https://github.com/<your-username>/<your-repo-name>](https://github.com/nyain/Pokemon-Image-Scraper)) to scrape and create the dataset.
   - Follow the instructions in that repository to generate and prepare the dataset.

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
   From here, you can upload images of Pokémon and get predictions along with detailed Pokémon information.

#### Notes:
- The application resizes uploaded images to fit the model's input dimensions.
- Images are processed in real-time, and no unnecessary files are saved to the server.
- 
### Future Enhancements:
- **Expand Dataset:** Include Pokémon from additional generations and variations to improve model accuracy and robustness.
- **Model Improvements:** Experiment with advanced CNN architectures or transfer learning to enhance classification performance.
- **REST API:** Develop a REST API for easy integration with other applications and services.
- **User Interface:** Enhance the web interface with additional features such as batch image uploads and detailed prediction analytics.
- **Performance Optimization:** Implement techniques for faster image processing and reduced server response times.
