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
