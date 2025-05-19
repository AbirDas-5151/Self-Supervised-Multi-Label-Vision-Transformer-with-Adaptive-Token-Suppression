import pandas as pd
import os
import shutil
import numpy as np
import argparse
from skmultilearn.model_selection import iterative_train_test_split

# --- Full list of 14 diseases for NIH and CheXpert ---
NIH_DISEASES = [
    'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule',
    'Pneumonia', 'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema',
    'Fibrosis', 'Pleural_Thickening', 'Hernia'
]

CHEXPERT_DISEASES = [
    'Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Opacity',
    'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia',
    'Atelectasis', 'Pneumothorax', 'Pleural Effusion', 'Pleural Other',
    'Fracture', 'Support Devices'
]

def prepare_nih_labels_for_split(df, disease_list):
    """
    Build binary matrix for NIH stratification:
      - 1 if present in Finding Labels, else 0
    """
    labels = np.zeros((len(df), len(disease_list)), dtype=int)
    for i, d in enumerate(disease_list):
        labels[:, i] = df['Finding Labels'].apply(lambda x: 1 if d in x else 0).values
    return labels

def prepare_chexpert_labels_for_split(df, disease_list):
    """
    Build binary matrix for CheXpert stratification:
      - 1.0 -> 1
      - 0.0, NaN, -1.0 -> 0
    Does NOT modify df, so original -1 labels remain for training.
    """
    labels_df = df[disease_list].copy().fillna(0.0)
    for col in disease_list:
        labels_df[col] = labels_df[col].replace(-1.0, 0.0)\
                                       .apply(lambda x: 1 if x == 1.0 else 0)
    return labels_df.values.astype(int)

def copy_image_files(df_sample, full_image_dir, sample_image_dir, img_col, dataset_type):
    """
    Copy sampled images (flat for NIH, nested for CheXpert).
    """
    os.makedirs(sample_image_dir, exist_ok=True)
    copied, missing = 0, 0
    for _, row in df_sample.iterrows():
        rel_path = row[img_col]
        if dataset_type=='nih':
            src = os.path.join(full_image_dir, rel_path)
            dst = os.path.join(sample_image_dir, os.path.basename(rel_path))
        else:
            src = os.path.join(full_image_dir, rel_path)
            dst = os.path.join(sample_image_dir, rel_path)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
        if os.path.exists(src):
            shutil.copy(src, dst)
            copied += 1
        else:
            missing += 1
    print(f"Copied {copied} images; {missing} missing.")

def main_create_sample(args):
    # Load full CSV
    df = pd.read_csv(args.full_csv_path)
    # Choose diseases & build split-matrix
    if args.dataset_type=='nih':
        diseases = NIH_DISEASES
        y_split = prepare_nih_labels_for_split(df, diseases)
        img_col = 'Image Index'
    else:
        # Use custom disease list if provided
        if args.disease_list_str:
            diseases = [d.strip() for d in args.disease_list_str.split(',')]
        else:
            diseases = CHEXPERT_DISEASES
        y_split = prepare_chexpert_labels_for_split(df, diseases)
        img_col = 'Path'
    # Determine fraction
    # Determine test_size from total_sample_size or sample_fraction
    if args.total_sample_size:
        if not 0 < args.total_sample_size < len(df):
            raise ValueError("total_sample_size must be less than total dataset size and greater than 0.")
        test_size = 1.0 - (args.total_sample_size / len(df))
    else:
        if not 0 < args.sample_fraction < 1:
            raise ValueError("sample_fraction must be between 0 and 1")
        test_size = 1.0 - args.sample_fraction

    # Stratified split
    X = df.index.to_numpy().reshape(-1, 1)
    X_tr, _, _, _ = iterative_train_test_split(X, y_split, test_size=test_size)
    df_sample = df.loc[X_tr.flatten()].copy()
    # Save subset CSV (with original -1s intact)
    os.makedirs(os.path.dirname(args.sample_csv_path), exist_ok=True)
    df_sample.to_csv(args.sample_csv_path, index=False)
    print(f"Saved subset CSV ({len(df_sample)} rows) to {args.sample_csv_path}")
    # Copy images
    copy_image_files(df_sample, args.full_image_dir, args.sample_image_dir, img_col, args.dataset_type)

if __name__=="__main__":
    parser = argparse.ArgumentParser(
        description="Create stratified subsets for NIH or CheXpert (all 14 classes)."
    )
    parser.add_argument("--dataset_type", choices=['nih','chexpert'], required=True)
    parser.add_argument("--full_csv_path",   required=True)
    parser.add_argument("--full_image_dir",  required=True)
    parser.add_argument("--sample_csv_path", required=True)
    parser.add_argument("--sample_image_dir",required=True)
    parser.add_argument("--sample_fraction", type=float, default=None)
    parser.add_argument("--total_sample_size", type=int, default=None)
    parser.add_argument("--disease_list_str", type=str, default=None, help="Comma-separated list of diseases to use (only for chexpert). Overrides default 13 diseases.")
    args = parser.parse_args()


    if not (args.sample_fraction or args.total_sample_size):
        parser.error("Specify --sample_fraction or --total_sample_size")
    if args.sample_fraction and args.total_sample_size:
        parser.error("Use only one of --sample_fraction or --total_sample_size")

    main_create_sample(args)
    print("\n--- STRATIFIED SAMPLE CREATION COMPLETE ---")
    print("1. Check the sample CSV and image directory.")
    print("2. If using Google Colab, ZIP the sample folder and upload to Google Drive.")

    print("\n--- SCRIPT USAGE REMINDERS ---")
    print("1. This script runs on your LOCAL computer, on the FULL downloaded datasets.")
    print("2. Example - NIH (sample 5%):")
    print("   python your_script_name.py --dataset_type nih \\")
    print("     --full_csv_path \"/path/to/full/NIH/Data_Entry_2017.csv\" \\")
    print("     --full_image_dir \"/path/to/full/NIH/images/\" \\")
    print("     --sample_csv_path \"./NIH_Sample_Stratified/sample_nih.csv\" \\")
    print("     --sample_image_dir \"./NIH_Sample_Stratified/images/\" \\")
    print("     --sample_fraction 0.05")
    print("\n3. Example - CheXpert (sample 2000 images from train.csv):")
    print("   python your_script_name.py --dataset_type chexpert \\")
    print("     --full_csv_path \"/path/to/full/CheXpert-v1.0/train.csv\" \\")
    print("     --full_image_dir \"/path/to/full/CheXpert_data_root/\" \\") # e.g., /path/to/full/ if CSV paths are CheXpert-v1.0/...
    print("     --sample_csv_path \"./CheXpert_Sample_Train_Stratified/sample_chexpert_train.csv\" \\")
    print("     --sample_image_dir \"./CheXpert_Sample_Train_Stratified/images_data_root/\" \\")
    print("     --total_sample_size 2000 \\")
    print("     --disease_list_str \"Atelectasis,Cardiomegaly,Consolidation,Edema,Pleural Effusion\"")
    print("\n4. After running, ZIP the output sample folder (e.g., 'NIH_Sample_Stratified/') and upload to Google Drive.")
    print("5. In Colab: Mount Drive, copy ZIP, unzip, then update your YAML configs to use these unzipped paths.")