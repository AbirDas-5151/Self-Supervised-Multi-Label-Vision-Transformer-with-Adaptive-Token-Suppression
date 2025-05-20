import pandas as pd
import os
import shutil
import numpy as np
import argparse
from skmultilearn.model_selection import iterative_train_test_split

# Disease definitions
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
    labels = np.zeros((len(df), len(disease_list)), dtype=int)
    for i, d in enumerate(disease_list):
        labels[:, i] = df['Finding Labels'].apply(lambda x: 1 if d in x else 0).values
    return labels

def prepare_chexpert_labels_for_split(df, disease_list):
    labels_df = df[disease_list].copy().fillna(0.0)
    for col in disease_list:
        labels_df[col] = labels_df[col].replace(-1.0, 0.0).apply(lambda x: 1 if x == 1.0 else 0)
    return labels_df.values.astype(int)

def copy_image_files(df_sample, full_image_dir, sample_image_dir, img_col, dataset_type):
    os.makedirs(sample_image_dir, exist_ok=True)
    copied, missing = 0, 0
    for _, row in df_sample.iterrows():
        rel_path = row[img_col]
        if dataset_type == 'nih':
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

def main_create_split(args):
    df = pd.read_csv(args.full_csv_path)

    # Determine labels and image column
    if args.dataset_type == 'nih':
        diseases = NIH_DISEASES
        y_all = prepare_nih_labels_for_split(df, diseases)
        img_col = 'Image Index'
    else:
        diseases = [d.strip() for d in args.disease_list_str.split(',')] if args.disease_list_str else CHEXPERT_DISEASES
        y_all = prepare_chexpert_labels_for_split(df, diseases)
        img_col = 'Path'

    # Determine number of samples
    if args.total_sample_size:
        if not 0 < args.total_sample_size < len(df):
            raise ValueError("total_sample_size must be less than full dataset size.")
        test_size = 1.0 - (args.total_sample_size / len(df))
        X_sampled, _, y_sampled, _ = iterative_train_test_split(df.index.to_numpy().reshape(-1,1), y_all, test_size=test_size)
        df_sampled = df.loc[X_sampled.flatten()].copy()
        y_all = y_sampled  # update labels for next split
    else:
        df_sampled = df.copy()

    # Final train-valid split (80-20)
    X = df_sampled.index.to_numpy().reshape(-1, 1)
    y = y_all
    X_train, X_valid, _, _ = iterative_train_test_split(X, y, test_size=args.valid_fraction)
    df_train = df_sampled.loc[X_train.flatten()].copy()
    df_valid = df_sampled.loc[X_valid.flatten()].copy()

    # Save CSVs
    os.makedirs(args.sample_dir, exist_ok=True)
    train_csv_path = os.path.join(args.sample_dir, "sample_nih_train.csv")
    valid_csv_path = os.path.join(args.sample_dir, "sample_nih_valid.csv")
    df_train.to_csv(train_csv_path, index=False)
    df_valid.to_csv(valid_csv_path, index=False)
    print(f"Saved {len(df_train)} training rows to {train_csv_path}")
    print(f"Saved {len(df_valid)} validation rows to {valid_csv_path}")

    # Copy images
    copy_image_files(df_train, args.full_image_dir, os.path.join(args.sample_dir, "images_train"), img_col, args.dataset_type)
    copy_image_files(df_valid, args.full_image_dir, os.path.join(args.sample_dir, "images_valid"), img_col, args.dataset_type)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create train/valid splits for NIH or CheXpert datasets.")
    parser.add_argument("--dataset_type", choices=['nih','chexpert'], required=True)
    parser.add_argument("--full_csv_path", required=True)
    parser.add_argument("--full_image_dir", required=True)
    parser.add_argument("--sample_dir", required=True)  # Directory to save CSVs and images
    parser.add_argument("--valid_fraction", type=float, default=0.2)  # 20% validation by default
    parser.add_argument("--total_sample_size", type=int, default=None)
    parser.add_argument("--disease_list_str", type=str, default=None)
    args = parser.parse_args()

    main_create_split(args)
    print("\n--- STRATIFIED TRAIN/VALID SPLIT COMPLETE ---")
    print("CSV and images are stored under:", args.sample_dir)
