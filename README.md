# Document-KeywordGen-Clustering
This python program uses OCR text extraction, GPT-powered keyword extraction, and hierarchical clustering. It processes PDFs into JSON, generates keywords, and applies unsupervised clustering. Supporting bilingual text (Malay &amp; English), it uses OpenAI embeddings for semantic analysis and visualizes clusters via dendrograms, PCA, and t-SNE.
# Document Management Program  

## Overview  
This repository provides an **document management solution** for processing **PDF** using **OCR, NLP, and clustering techniques**. The system extracts text from **PDF documents**, converts it into structured JSON, **generates keywords** using GPT-4o, and applies **hierarchical clustering** for document categorization. It is designed to streamline document organization and facilitate **semantic search and retrieval**.

## Features  
- **OCR-based Text Extraction**: Converts scanned PDFs into structured text using Tesseract OCR.  
- **GPT-4o-powered Keyword Extraction**: Extracts **relevant keywords** for each document.  
- **Unsupervised Document Clustering**: Organizes documents using **Hierarchical and K-Means clustering**.  
- **Data Representation with OpenAI Embeddings**: Converts raw and processed text into dense numerical vectors.  
- **Cluster Visualization**: Generates **dendrograms, PCA, and t-SNE projections** for better cluster interpretation.  
- **User-Friendly GUI**: Includes a PySimpleGUI-based interface for easy document processing.  

## Installation  
### **Prerequisites**  
- Python 3.8+  
- Tesseract OCR (Install from [here](https://github.com/UB-Mannheim/tesseract/wiki))  
- Poppler for PDF conversion (Install from [here](https://poppler.freedesktop.org/))  

### **Python Dependencies**  
Install required packages using:  
```sh
pip install -r requirements.txt
```

## Usage  
### **1. Extract Keywords from a PDF**  
1. Run the script:  
   ```sh
   python main.py
   ```  
2. Select **"1) Select PDF for Keyword Extraction"** in the GUI.  
3. Processed results (JSON and CSV) will be stored in the `outputJSON/` and `outputCSV/` folders.  

### **2. Perform Document Clustering**  
1. Run the script and select **"2) Select Folder for Hierarchical Clustering"**.  
2. The system will process all PDFs in the folder, extract text, generate embeddings, and cluster the documents.  
3. Clustering results, along with visualizations (dendrograms, PCA, t-SNE), will be saved in the project folder.  

## Output Files  
- **Structured JSON files** (`outputJSON/`): Contains extracted fields from each application form.  
- **CSV files** (`outputCSV/`): Stores extracted keywords and structured text.  
- **Cluster Assignment CSV** (`clusterAssignment.csv`): Contains assigned cluster IDs for each document.  
- **Visualization Files** (`Dendrogram.png`, `PCA.png`, `tSNE.png`): Help in understanding document relationships.  

## Technologies Used  
- **OCR Processing**: Tesseract  
- **Text Structuring**: GPT-4o  
- **Keyword Extraction**: TF-IDF, KeyBERT, OpenAI Embeddings, GPT-4o  
- **Clustering Algorithms**: Hierarchical Clustering (Agglomerative), K-Means  
- **Embedding Models**: OpenAI `text-embedding-3-large`  
- **Visualization**: Matplotlib, Seaborn, PCA, t-SNE 
