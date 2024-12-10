import streamlit as st
import numpy as np
from sentence_transformers import SentenceTransformer
from PIL import Image
import faiss
import os
import io

class ImageRetrievalApp:
    def __init__(self):
        # Initialize the model and index
        self.model = None
        self.index = None
        self.image_paths = None
        
    def load_model(self):
        """Load the CLIP model"""
        try:
            self.model = SentenceTransformer('clip-ViT-B-32')
            return True
        except Exception as e:
            st.error(f"Error loading model: {e}")
            return False
    
    def load_index(self, index_path='vector.index'):
        """Load the Faiss index and image paths"""
        try:
            self.index = faiss.read_index(index_path)
            with open(index_path + '.paths', 'r') as f:
                self.image_paths = [line.strip() for line in f]
            return True
        except Exception as e:
            st.error(f"Error loading index: {e}")
            return False
    
    def convert_uploaded_file_to_pil(self, uploaded_file):
        """Convert Streamlit uploaded file to PIL Image"""
        if uploaded_file is None:
            return None

        try:
            image_bytes = uploaded_file.getvalue()
            pil_image = Image.open(io.BytesIO(image_bytes))
            pil_image = pil_image.convert('RGB')
            return pil_image
        except Exception as e:
            st.error(f"Error converting uploaded file: {e}")
            return None
    
    def retrieve_similar_images(self, query, top_k=3):
        """Retrieve similar images based on query"""
        if not self.model or not self.index:
            st.error("Model or index not loaded!")
            return None, None
        
        try:
            # Preprocess query
            if isinstance(query, st.runtime.uploaded_file_manager.UploadedFile):
                query = self.convert_uploaded_file_to_pil(query)
            
            if query is None:
                st.error("Invalid query")
                return None, None
            
            # Encode query
            query_features = self.model.encode(query)
            query_features = query_features.astype(np.float32).reshape(1, -1)
            
            # Search in index
            distances, indices = self.index.search(query_features, top_k)
            
            # Retrieve image paths
            retrieved_images = [self.image_paths[int(idx)] for idx in indices[0]]
            
            return query, retrieved_images
        
        except Exception as e:
            st.error(f"Error retrieving similar images: {e}")
            return None, None
    
    def run(self):
        """Main Streamlit app"""
        # Set page configuration
        st.set_page_config(
            page_title="CLIP Image Retrieval",
            page_icon=":mag_right:",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        # Custom CSS for styling
        st.markdown("""
        <style>
        .main-container {
            background-color: #f0f2f6;
            padding: 2rem;
            border-radius: 10px;
        }
        .stButton > button {
            background-color: #4CAF50;
            color: white;
            border: none;
            border-radius: 5px;
            padding: 10px 20px;
            transition: all 0.3s ease;
        }
        .stButton > button:hover {
            background-color: #45a049;
            transform: scale(1.05);
        }
        .result-image {
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            transition: transform 0.3s ease;
        }
        .result-image:hover {
            transform: scale(1.05);
        }
        </style>
        """, unsafe_allow_html=True)
        
        # Title and description
        st.title(":mag_right: CLIP Image Retrieval")
        st.markdown("Find similar images using advanced machine learning techniques!")
        
        # Sidebar for configuration
        st.sidebar.header("Image Retrieval Settings")
        
        # Load model and index
        if self.load_model() and self.load_index():
            st.sidebar.success("Model and Index Loaded Successfully!")
        
        # Query input
        query_type = st.sidebar.radio("Query Type", ["Text", "Image"])
        
        # Number of results
        top_k = st.sidebar.slider("Number of Similar Images", 1, 10, 3)
        
        # Query input based on type
        if query_type == "Text":
            query = st.text_input("Enter search text")
            query_image = None
        else:
            query_image = st.file_uploader("Upload an image", type=['png', 'jpg', 'jpeg', 'tiff', 'bmp', 'gif'])
            query = query_image
        
        # Search button
        if st.button("Find Similar Images"):
            if query:
                with st.spinner('Searching for similar images...'):
                    # Retrieve similar images
                    original_query, retrieved_images = self.retrieve_similar_images(query, top_k)
                
                if retrieved_images:
                    # Display results
                    st.subheader("Results")
                    
                    # Create columns for images
                    cols = st.columns(len(retrieved_images))
                    
                    for i, (col, img_path) in enumerate(zip(cols, retrieved_images)):
                        with col:
                            # Open and display image
                            img = Image.open(img_path)
                            st.image(img, caption=f"Match {i+1}", use_container_width=True, 
                                     output_format='PNG', clamp=True)
                            
                            # Option to download
                            with open(img_path, "rb") as file:
                                st.download_button(
                                    label="Download",
                                    data=file,
                                    file_name=os.path.basename(img_path),
                                    mime="image/png"
                                )
                else:
                    st.warning("No similar images found.")
            else:
                st.warning("Please enter a text query or upload an image.")
        
        # Footer
        st.markdown("---")
        st.markdown(":copyright: 2024 CLIP Image Retrieval App")

# Run the app
def main():
    app = ImageRetrievalApp()
    app.run()

if __name__ == "__main__":
    main()