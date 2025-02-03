# Animal Recognition Project

This project is an image classification system designed to recognize different animals using deep learning models. It utilizes pre-trained models, a custom database, and supporting Python scripts to preprocess, train, and test the recognition pipeline.

---

## Table of Contents
1. [Project Overview](#project-overview)
2. [Features](#features)
3. [Files Description](#files-description)
4. [How to Run](#how-to-run)
5. [Requirements](#requirements)
6. [Future Improvements](#future-improvements)

---

## Project Overview

This project provides a pipeline for animal recognition, starting with organized datasets and leading to a fully trained deep learning model. It can:
- Identify animals from given images.
- Handle organized datasets stored in a database.
- Allow easy extensions for new classes of animals or plants.

---

## Features

- Pre-trained model integration (`full_model.pth`).
- Organized data pipeline for training and testing.
- Database support (`species.db`) for storing species information.
- Fully documented code for further customization.

---

## Files Description

Here’s a breakdown of the files in this repository:

### Python Scripts
- **`back-end.py`**: Contains the backend API logic for serving predictions from the trained model.
- **`check_data.py`**: Utility script for verifying and cleaning dataset files.
- **`data_base.py`**: Defines the database schema and manages database interactions using SQLAlchemy.
- **`db_utils.py`**: Utility functions to manage database interactions (e.g., migrations or schema setup).
- **`organise_data.py`**: Organizes and preprocesses raw datasets into structured formats ready for training.
- **`populate_database.py`**: Populates the database (`species.db`) with species and class information.
- **`train_py.py`**: The main script for training the deep learning model.
- **`test.py`**: Used for evaluating the trained model on a test dataset.

### Data Files
- **`species.db`**: A SQLite database containing information about animal and plant species.
   - **Plants Table**: Stores details about plants, including:
     - `id`: Unique identifier.
     - `name`: Common name of the plant.
     - `scientific_name`: Scientific name of the plant.
     - `description`: A textual description.
     - `habitat`: The plant's natural habitat.
     - `flowering_season`: Typical flowering season.
   - **Animals Table**: Stores details about animals, including:
     - `id`: Unique identifier.
     - `name`: Common name of the animal.
     - `scientific_name`: Scientific name of the animal.
     - `description`: A textual description.
     - `habitat`: The animal's natural habitat.
     - `diet`: The animal's diet.

### Model Files
- **`full_model.pth`**: Pre-trained PyTorch model used for animal recognition.

### Other Files
- **`classes_map.json`**: JSON file mapping class IDs to animal or plant names.
- **`requirements.txt`**: A list of Python dependencies required to run this project.
- **`index.html`**: A simple front-end interface for interacting with the backend API (if applicable).

---

# How to Run

Follow these steps to run the project:

1. **Clone the repository**:

   ```bash
   git clone https://github.com/yourusername/animal-recognition.git
   cd animal-recognition
   ```

2. **Install the required dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

3. **Prepare the database**:  
   Run the script to populate the database:

   ```bash
   python populate_database.py
   ```

4. **Train the model** (optional):

   ```bash
   python train_py.py
   ```

5. **Run the backend**:  
   Start the API backend to serve predictions:

   ```bash
   python back-end.py
   ```

6. **Test the system**:  
   Use the sample image or your custom images with the backend.

---

## Requirements

To run this project, ensure you have:
- Python 3.7 or higher.
- Dependencies listed in `requirements.txt` installed.
- A CUDA-capable GPU (optional for faster training and inference).

---

## Future Improvements

- Add support for additional plant species.
- Implement a web-based interface for easier interaction.
- Optimize the training process with advanced augmentation techniques.
- Migrate large files to a cloud-based storage system.

---

## Credits

This project was created by Ahmed Ben aissa. Feel free to contribute or raise issues for improvement!

   
