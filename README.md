CLIP Image Retrieval App
========================

This is a Streamlit application that allows users to search for visually similar images using the CLIP (Contrastive Language-Image Pre-training) model.

Project Structure
-----------------

The project files are organized as follows:

``image-retrieval-app/
├── images/
├── venv/
├── app.py
├── image_similarity.ipynb
├── vector.index
└── vector.index.paths``

-   `images/`: This directory contains the images that will be used for the image retrieval task.
-   `venv/`: This directory contains the Python virtual environment for the project.
-   `app.py`: The main Streamlit application file.
-   `image_similarity.ipynb`: A Jupyter Notebook that demonstrates the image retrieval process.
-   `vector.index` and `vector.index.paths`: These files contain the pre-computed CLIP embeddings and image paths, used for efficient retrieval.

Prerequisites
-------------

-   Python 3.7 or later
-   pip
-   Streamlit
-   Sentence Transformers
-   Pillow
-   Faiss
-   Matplotlib

Installation
------------

1.  Create a Python virtual environment:

    `python -m venv venv`

2.  Activate the virtual environment:
    -   On Windows:

        `venv\Scripts\activate`

    -   On macOS/Linux:

        `source venv/bin/activate`

3.  Install the required dependencies:

    `pip install streamlit sentence-transformers pillow faiss-cpu matplotlib`

Running the App
---------------

1.  Ensure you have the `vector.index` and `vector.index.paths` files in the project directory.
2.  Run the Streamlit application:

    `streamlit run app.py`

    This will start the Streamlit application and open it in your default web browser.

Usage
-----

The application allows you to search for similar images using either text-based or image-based queries.

1.  **Text-based search**: Enter a text query in the input field and click the "Find Similar Images" button.
2.  **Image-based search**: Upload an image using the file uploader and click the "Find Similar Images" button.

The application will display the top `N` (configurable) most similar images to the query, where `N` is the number of results specified in the sidebar.

You can also download the retrieved images by clicking the "Download" button under each image.

Customization
-------------

If you want to customize the application further, you can modify the `app.py` file. Some potential customizations include:

-   Changing the styling and layout of the application
-   Adding more advanced search options (e.g., filtering by tags, categories, or metadata)
-   Integrating the application with a database or content management system to handle larger image collections
-   Deploying the application to a web server or cloud platform

Acknowledgements
----------------

This project uses the CLIP model, which was developed by OpenAI and Anthropic. The image retrieval functionality is based on the Faiss library for efficient nearest neighbor search.
