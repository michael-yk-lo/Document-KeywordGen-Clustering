#!/usr/bin/env python
# coding: utf-8

"""
Local Python Program for:
1) Keyword Extraction from PDF (OCR -> GPT-4o -> JSON & CSV)
2) Folder-Based Hierarchical Clustering

Requirements:
- PySimpleGUI (pip install PySimpleGUI)
- pytesseract (pip install pytesseract)
- Tesseract-OCR installed on Windows (https://github.com/UB-Mannheim/tesseract/wiki)
- openai (pip install openai)
- numpy, pandas, scikit-learn, matplotlib, seaborn
"""

import os
import hashlib
import csv
import json
import PySimpleGUI as sg
import openai
from openai import OpenAI
import pytesseract
from PIL import Image
from pdf2image import convert_from_path
import pandas as pd
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# ---------------------------------------------------------------------
# --------------- Global / Developer Settings -------------------------
# ---------------------------------------------------------------------
DEVELOPER_MODE = False  # toggle in the GUI to show/hide developer options
DEFAULT_OPENAI_API_KEY = "OPENAI API KEY"
DEFAULT_NUM_CLUSTERS = 7  # default for hierarchical clustering

openai.api_key = DEFAULT_OPENAI_API_KEY

# GPT-4o model name or any custom model endpoint
GPT_MODEL = "gpt-4o"

TESSERACT_CMD = r"C:\Program Files\Tesseract-OCR\tesseract.exe"  # Update path if needed
pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD

# JSON Structure for GPT-4o
JSON_STRUCTURE = """
{
  "A": {
    "Details of Project": {
      "Title of proposed project": "original text here"
    }
  },
  "C": {
    "Details of Key Result Area": {
      "Key Result Area of application": "original English text for checked option"
    }
  },
  "F": {
    "Details of Community": {
      "Summary of community problem statement": "original text here"
    }
  },
  "G": {
    "Proposed Project": {
      "Executive Summary of Project Proposal": {
        "Background of project": "original text here",
        "Literature reviews": "original text here",
        "Objectives": "original text here",
        "Project methodology": "original text here",
        "Type of knowledge/technology to be transferred": "original text here",
        "Expected Outcomes": "original text here"
      }
    }
  },
  "I": {
    "Programme Outputs and Impacts": [
      {
        "Measurement": "Output to target community/organization",
        "Outputs": "original text here",
        "Impacts": "original text here"
      },
      {
        "Measurement": "Return to the USM",
        "Outputs": "original text here",
        "Impacts": "original text here"
      },
      {
        "Measurement": "Human capital development",
        "Outputs": "original text here",
        "Impacts": "original text here"
      },
      {
        "Measurement": "Intangible output",
        "Outputs": "original text here",
        "Impacts": "original text here"
      },
      {
        "Measurement": "Others",
        "Outputs": "original text here",
        "Impacts": "original text here"
      }
    ]
  }
}
"""

# ---------------------------------------------------------------------
# --------------- Utility Functions -----------------------------------
# ---------------------------------------------------------------------

def extract_text_from_pdf(pdf_path):
    poppler_path = r'C:/Program Files/poppler/Library/bin'
    images = convert_from_path(pdf_path, poppler_path=poppler_path)
    ocr_text = ""
    for i, image in enumerate(images):
        text = pytesseract.image_to_string(image, lang='eng+msa')  # Assuming mixed English & Malay
        ocr_text += f"\n--- Page {i+1} ---\n" + text
    return ocr_text

def call_gpt4o_for_json(ocr_text):
    """
    Calls GPT-4o model to format OCR text into the structured JSON.
    Returns the JSON string from the model.
    """
    prompt = f"""
1) You are given OCR-extracted text from a scanned grant application form in Malay and English.
The form contains structured data in specific sections with fields for project details, faculty information, key result areas, budget, etc.
Extract specified data field from the text of the provided OCR text in a structured JSON format, preserving all original wording and formatting. 
Do not summarize, paraphrase, or modify the text in any way. 
Use the following JSON structure, with each section's content as-is:

{JSON_STRUCTURE}

Here is the extracted text:

{ocr_text}
"""
    try:
        client = OpenAI()
        response = client.chat.completions.create(
            model=GPT_MODEL,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that formats OCR-extracted text into structured JSON."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=15000,
            temperature=0
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        sg.popup_error(f"OpenAI API error: {e}")
        return None

def call_gpt4o_for_keywords(json_extracted_text):
    """
    Calls GPT-4o model to extract top 5 relevant keywords from the JSON text (1-2 words each).
    Returns the comma-separated keyword string.
    """
    prompt = """Extract the top 5 most relevant keywords for the JSON extracted text, 
the keywords must be 1 to 2 words only. Keywords provided should be a balance between 
general domain terms and specific details about the project or content. 
Include both broad categories (e.g., education, health) and project-specific terms 
(e.g., community health, STEM initiatives). Provide them as a comma-separated list without explanation.
"""
    combined_prompt = prompt + "\n\n" + json_extracted_text
    try:
        client = OpenAI()
        response = client.chat.completions.create(
            model=GPT_MODEL,
            messages=[
                {"role": "system", "content": "You extract concise 1-2 word keywords."},
                {"role": "user", "content": combined_prompt}
            ],
            max_tokens=300,
            temperature=0
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        sg.popup_error(f"OpenAI API error: {e}")
        return None

def parse_json_and_build_csv(json_str, pdf_filename):
    """
    Creates the output folders (outputJSON, outputCSV) if not exist.
    Saves the JSON file, then builds a CSV row with the required columns.
    Returns a DataFrame row for further usage if needed.
    """
    # Output folder creation
    base_dir = os.path.dirname(pdf_filename)
    json_out_dir = os.path.join(base_dir, "outputJSON")
    csv_out_dir = os.path.join(base_dir, "outputCSV")
    os.makedirs(json_out_dir, exist_ok=True)
    os.makedirs(csv_out_dir, exist_ok=True)

    # Save JSON
    base_pdf_name = os.path.splitext(os.path.basename(pdf_filename))[0]
    first_5_words = "_".join(base_pdf_name.split()[:5])
    json_output_path = os.path.join(json_out_dir, f"{first_5_words}.json")

    with open(json_output_path, "w", encoding="utf-8") as jf:
        jf.write(json_str)

    # parse JSON to get fields
    try:
        data_dict = json.loads(json_str)
    except json.JSONDecodeError as e:
        sg.popup_error(f"JSON decoding error: {e}")
        return None

    # Extract fields from JSON if they exist
    # Example paths based on your structure
    project_title = ""
    project_KeyResultArea = ""
    community_details = ""
    project_details = ""
    outputs_impacts = ""

    try:
        project_title = data_dict["A"]["Details of Project"]["Title of proposed project"]
    except:
        pass

    try:
        project_KeyResultArea = data_dict["C"]["Details of Key Result Area"]["Key Result Area of application"]
    except:
        pass

    try:
        community_details = data_dict["F"]["Details of Community"]["Summary of community problem statement"]
    except:
        pass

    try:
        # Example of concatenating the entire G section
        g_content = data_dict["G"]["Proposed Project"]["Executive Summary of Project Proposal"]
        # Concatenate them
        project_details_list = []
        for key, val in g_content.items():
            project_details_list.append(f"{key.upper()}: {val}")
        project_details = "\n".join(project_details_list)
    except:
        pass

    try:
        # Combine all items in "I" into a single text
        i_content = data_dict["I"]["Programme Outputs and Impacts"]
        combined_i = []
        for meas in i_content:
            # e.g. measurement, outputs, impacts
            line = f"{meas['Measurement']} | Outputs: {meas['Outputs']} | Impacts: {meas['Impacts']}"
            combined_i.append(line)
        outputs_impacts = "\n".join(combined_i)
    except:
        pass

    # Build raw_text
    raw_text = community_details + "\n" + project_details + "\n" + outputs_impacts

    # Generate project_id from combined text
    combined_text = f"{project_title} {project_KeyResultArea} {community_details} {project_details} {outputs_impacts}"
    project_id = hashlib.md5(combined_text.encode('utf-8')).hexdigest()

    return {
        "project_id": project_id,
        "project_title": project_title,
        "project_KeyResultArea": project_KeyResultArea,
        "community_details": community_details,
        "project_details": project_details,
        "output_impacts": outputs_impacts,
        "raw_text": raw_text
    }

def save_csv_row(row_data, pdf_filename, gpt4o_keywords):
    base_dir = os.path.dirname(pdf_filename)
    csv_out_dir = os.path.join(base_dir, "outputCSV")
    os.makedirs(csv_out_dir, exist_ok=True)

    base_pdf_name = os.path.splitext(os.path.basename(pdf_filename))[0]
    first_5_words = "_".join(base_pdf_name.split()[:5])
    csv_output_path = os.path.join(csv_out_dir, f"{first_5_words}.csv")

    row_data["GPT-4o_keywords"] = gpt4o_keywords

    # Save to CSV (append if exists, else write header)
    write_header = not os.path.exists(csv_output_path)
    with open(csv_output_path, mode="a", encoding="utf-8", newline="") as cf:
        writer = csv.DictWriter(
            cf,
            fieldnames=[
                "project_id", "project_title", "project_KeyResultArea",
                "community_details", "project_details", "output_impacts",
                "raw_text", "GPT-4o_keywords"
            ]
        )
        if write_header:
            writer.writeheader()
        writer.writerow(row_data)

# ---------------------------------------------------------------------
# ------------------------ Clustering Functions ------------------------
# ---------------------------------------------------------------------

def get_embedding(text, model="text-embedding-3-large"):
    import time
    text = text.replace("\n", " ")
    try:
        client = OpenAI()
        resp = client.embeddings.create(input=[text], model=model)
        return resp.data[0].embedding
    except Exception as e:
        sg.popup_error(f"OpenAI Embedding Error: {e}")
        time.sleep(2)
        return None

def do_folder_clustering(folder_path, n_clusters=7):
    print("""
    1) For each PDF in folder, check if there's an existing CSV in 'outputCSV' subfolder.
       If not, process the PDF for keyword extraction, etc.
    2) Merge all CSV rows into a single DataFrame clusterData.csv
    3) Embed raw_text, run hierarchical clustering, display metrics, produce visuals.
    """)

    # Step 1: gather data
    print("DEBUG: Step 1: gather data")
    pdf_files = [f for f in os.listdir(folder_path) if f.lower().endswith(".pdf")]
    all_rows = []
    for pdf_file in pdf_files:
        pdf_full = os.path.join(folder_path, pdf_file)

        # CSV name
        base_pdf_name = os.path.splitext(os.path.basename(pdf_file))[0]
        first_5_words = "_".join(base_pdf_name.split()[:5])
        out_csv_dir = os.path.join(folder_path, "outputCSV")
        csv_path = os.path.join(out_csv_dir, f"{first_5_words}.csv")

        if os.path.exists(csv_path):
            # read existing row(s)
            temp_df = pd.read_csv(csv_path)
            all_rows.append(temp_df)
        else:
            # process PDF fresh
            row_df = process_single_pdf_for_keywords(pdf_full)  # custom function below
            if row_df is not None:
                all_rows.append(row_df)

    if not all_rows:
        sg.popup("No data found for clustering.")
        return

    # Combine all into one DataFrame
    print("DEBUG: Combine all into one DataFrame")
    combined_df = pd.concat(all_rows, ignore_index=True)
    # Save clusterData.csv
    cluster_data_path = os.path.join(folder_path, f"{os.path.basename(folder_path)}_clusterData.csv")
    combined_df.to_csv(cluster_data_path, index=False, encoding="utf-8")

    # 4) Embedding + scaling
    print("DEBUG: Embedding + scaling")
    embeddings = []
    for idx, row in combined_df.iterrows():
        emb = get_embedding(row["raw_text"])
        embeddings.append(emb)

    # if any are None, filter them out or handle
    valid_indices = [i for i, e in enumerate(embeddings) if e is not None]
    valid_embeddings = [embeddings[i] for i in valid_indices]
    cluster_df = combined_df.iloc[valid_indices].copy()

    if len(valid_embeddings) < 2:
        sg.popup("Not enough valid embeddings for clustering.")
        return

    X = np.vstack(valid_embeddings)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 5) Hierarchical Clustering
    print("DEBUG: Hierarchical Clustering")
    hc_model = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
    labels = hc_model.fit_predict(X_scaled)
    cluster_df["cluster_id"] = labels

    # 6) save cluster assignment
    print("DEBUG: save cluster assignment")
    cluster_assign_path = os.path.join(folder_path, "clusterAssignment.csv")
    cluster_df[["cluster_id","project_id","project_title","project_KeyResultArea","GPT-4o_keywords"]].to_csv(
        cluster_assign_path, index=False, encoding="utf-8"
    )

    # 7) evaluate metrics
    print("DEBUG: evaluate metrics")
    silhouette = silhouette_score(X_scaled, labels)
    db = davies_bouldin_score(X_scaled, labels)

    # show results
    sg.popup(f"Hierarchical Clustering Completed!\nSilhouette Score: {silhouette:.4f}\nDavies-Bouldin: {db:.4f}")

    # produce dendrogram
    produce_dendrogram(X_scaled, folder_path)
    produce_pca_plot(X_scaled, labels, folder_path)
    produce_tsne_plot(X_scaled, labels, folder_path)


def process_single_pdf_for_keywords(pdf_path):
    print("""
    OCR -> GPT-4o JSON -> parse JSON & build CSV -> GPT-4o keywords -> finalize CSV.
    Returns a DataFrame of 1 row with the needed columns, or None if error.
    """)
    print("DEBUG: Starting extract_text_from_pdf")
    ocr_text = extract_text_from_pdf(pdf_path)
    if not ocr_text:
        print("DEBUG: OCR text is empty or None.")
        return None

    # GPT-4o => JSON
    print("DEBUG: Starting call_gpt4o_for_json")
    json_resp = call_gpt4o_for_json(ocr_text)
    json_resp = json_resp.strip().strip("```json").strip('```')
    if not json_resp:
        print("DEBUG: No JSON response from GPT.")
        return None

    # parse JSON => row dict
    print("DEBUG: Starting parse_json_and_build_csv")
    row_data = parse_json_and_build_csv(json_resp, pdf_path)
    if not row_data:
        return None

    # GPT-4o => keywords
    print("DEBUG: Starting call_gpt4o_for_keywords")
    gpt4o_keywords = call_gpt4o_for_keywords(json_resp)
    if not gpt4o_keywords:
        gpt4o_keywords = ""

    # save CSV
    print("DEBUG: Starting save_csv_row")
    save_csv_row(row_data, pdf_path, gpt4o_keywords)

    # convert row_data to DataFrame
    print("DEBUG: convert row_data to DataFrame")
    df = pd.DataFrame([row_data])
    df["GPT-4o_keywords"] = gpt4o_keywords
    return df


def produce_dendrogram(X_scaled, folder_path):
    """
    Create and save dendrogram plot for hierarchical clustering.
    """
    print("Create and save dendrogram plot for hierarchical clustering")
    from scipy.cluster.hierarchy import linkage, dendrogram
    Z = linkage(X_scaled, method='ward')
    plt.figure(figsize=(10, 6))
    dendrogram(Z)
    plt.title("Hierarchical Clustering Dendrogram")
    plt.xlabel("Documents")
    plt.ylabel("Distance")
    dendro_path = os.path.join(folder_path, f"{os.path.basename(folder_path)}_Dendrogram.png")
    plt.savefig(dendro_path)
    plt.close()

def produce_pca_plot(X_scaled, labels, folder_path):
    """
    PCA plot for cluster visualization
    """
    print("Create and save PCA plot for hierarchical clustering")
    pca = PCA(n_components=2, random_state=42)
    pcs = pca.fit_transform(X_scaled)
    plt.figure(figsize=(8,6))
    sns.scatterplot(x=pcs[:,0], y=pcs[:,1], hue=labels, palette="Set2")
    plt.title("PCA Projection")
    pca_path = os.path.join(folder_path, f"{os.path.basename(folder_path)}_PCA.png")
    plt.savefig(pca_path)
    plt.close()

def produce_tsne_plot(X_scaled, labels, folder_path):
    """
    t-SNE plot for cluster visualization
    """
    print("Create and save t-SNE plot for hierarchical clustering")
    tsne = TSNE(n_components=2, random_state=42)
    tsne_res = tsne.fit_transform(X_scaled)
    plt.figure(figsize=(8,6))
    sns.scatterplot(x=tsne_res[:,0], y=tsne_res[:,1], hue=labels, palette="Set1")
    plt.title("t-SNE Projection")
    tsne_path = os.path.join(folder_path, f"{os.path.basename(folder_path)}_tSNE.png")
    plt.savefig(tsne_path)
    plt.close()

# ---------------------------------------------------------------------
# -------------------- Main GUI / Program -----------------------------
# ---------------------------------------------------------------------

def main():
    global DEVELOPER_MODE
    global DEFAULT_OPENAI_API_KEY
    global DEFAULT_NUM_CLUSTERS

    layout = [
        [sg.Text("Document Management Program", font=("Any", 14, "bold"))],
        [sg.Frame("Options", [
            [sg.Button("1) Select PDF for Keyword Extraction")],
            [sg.Button("2) Select Folder for Hierarchical Clustering")],
        ])],
        [sg.Checkbox("Developer Mode", default=DEVELOPER_MODE, key="devmode", enable_events=True)],
        [sg.Frame("Developer Settings", [
            [sg.Text("OpenAI API Key:"), sg.InputText(DEFAULT_OPENAI_API_KEY, key="openai_key")],
            [sg.Text("Number of Clusters:"), sg.InputText(str(DEFAULT_NUM_CLUSTERS), key="num_clusters")]
        ], visible=DEVELOPER_MODE, key="dev_frame")],
        [sg.Exit()]
    ]

    window = sg.Window("Grant Document Management", layout, finalize=True)

    while True:
        event, values = window.read()

        if event in (sg.WIN_CLOSED, "Exit"):
            break

        if event == "devmode":
            DEVELOPER_MODE = values["devmode"]
            window["dev_frame"].update(visible=DEVELOPER_MODE)

        if event == "1) Select PDF for Keyword Extraction":
            pdf_path = sg.popup_get_file("Select a PDF file", file_types=(("PDF Files","*.pdf"),))
            if pdf_path:
                sg.popup("Processing PDF for keyword extraction, please wait...")
                process_single_pdf_for_keywords(pdf_path)
                sg.popup("Keyword Extraction Completed!")

        if event == "2) Select Folder for Hierarchical Clustering":
            folder_path = sg.popup_get_folder("Select a folder containing PDFs")
            if folder_path:
                # read dev settings
                if DEVELOPER_MODE:
                    openai.api_key = values["openai_key"]
                    try:
                        DEFAULT_NUM_CLUSTERS = int(values["num_clusters"])
                    except:
                        pass

                sg.popup("Clustering all PDFs in folder, please wait...")
                do_folder_clustering(folder_path, n_clusters=DEFAULT_NUM_CLUSTERS)
                sg.popup("Clustering Completed!")

    window.close()

if __name__ == "__main__":
    main()
