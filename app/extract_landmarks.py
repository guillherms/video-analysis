import os
import cv2
import io
import textwrap
import pandas as pd
import mediapipe as mp
from typing import Optional
from sklearn.model_selection import train_test_split
from huggingface_hub import HfApi, create_repo


def create_landmark_dataset() -> None:
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose()
    csv_path = "data/output/files/dataset_pose.csv"
    videos_path = "data/output/videos"

    descriptions = {
        "reading": "A person reading or reviewing a document attentively.",
        "waving": "A person waving their hand as a greeting or farewell gesture.",
        "agreeing": "A person nodding or showing agreement through gestures.",
        "dancing": "A person performing rhythmic body movements to music.",
        "lying_down": "A person lying down in a resting or relaxed position.",
        "medical_procedure": "A healthcare professional performing a medical procedure on a patient.",
        "medical_observation": "A healthcare worker observing or monitoring a patients condition.",
        "office_work": "People working in an office using laptops papers or mobile devices.",
        "handshake": "Two individuals greeting or closing a deal with a handshake."
    }

    header = [f"{a}{i}" for i in range(33) for a in ("x", "y", "z", "v")] + ["label", "description"]

    if not os.path.exists(csv_path):
        with open(csv_path, "w") as f:
            f.write(",".join(header) + "\n")

    for file in os.listdir(videos_path):
        label = os.path.splitext(file)[0]
        desc = descriptions.get(label, "No description available.")
        cap = cv2.VideoCapture(os.path.join(videos_path, file))
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if results.pose_landmarks:
                row = [str(v) for lm in results.pose_landmarks.landmark 
                            for v in (lm.x, lm.y, lm.z, lm.visibility)]
                # Escreve linha com label e descrição
                with open(csv_path, "a") as f:
                    f.write(",".join(row + [label, desc]) + "\n")
        
        cap.release()

    print("Dataset saved in", csv_path)

def publish_landmark_dataset() -> None:
    repo_id = "guillherms/human-activity-pose_v4"
    parquet_path = "data/output/files/dataset_pose.parquet"
    output_dir = "data/output/files/splits"
    readme_path = "README.md"

    if not os.path.exists(parquet_path):
        raise FileNotFoundError(f"parquet file not found: {parquet_path}")
    
    df = pd.read_parquet(parquet_path) if parquet_path.endswith(".parquet") else pd.read_csv(parquet_path)
    print(f"Loaded dataset with {len(df)} rows and {len(df.columns)} columns")
    
    # --- DIVISÃO TRAIN / VALIDATION ---
    X = df.drop(["label", "description"], axis=1)
    y = df["label"]

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    train_df = X_train.assign(label=y_train)
    val_df = X_val.assign(label=y_val)

    train_path = os.path.join(output_dir, "train.parquet")
    val_path = os.path.join(output_dir, "validation.parquet")

    train_df.to_parquet(train_path, index=False)
    val_df.to_parquet(val_path, index=False)

    print(f"Saved train: {len(train_df)} rows")
    print(f"Saved validation: {len(val_df)} rows")

    # --- PUBLICAÇÃO NO HUGGING FACE ---
    print("Authenticating with Hugging Face Hub...")
    api = HfApi()

    print(f"Creating dataset repo: {repo_id}")
    create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True)

    print("Uploading train split...")
    api.upload_file(
        path_or_fileobj=train_path,
        path_in_repo="train/train.parquet",
        repo_id=repo_id,
        repo_type="dataset"
    )

    print("Uploading validation split...")
    api.upload_file(
        path_or_fileobj=val_path,
        path_in_repo="validation/validation.parquet",
        repo_id=repo_id,
        repo_type="dataset"
    )
    
    readme = """
    ---
    annotations_creators:
    - human-annotated
    language:
    - en
    license: cc-by-4.0
    pretty_name: Human Activity Pose Dataset (Split Version)
    task_categories:
    - video-classification
    - keypoint-detection
    task_ids:
    - multi-class-image-classification
    - pose-estimation
    size_categories:
    - 1K<n<10K
    ---

    # 🧍 Human Activity Pose Dataset (Split Version)

    This dataset contains human pose landmarks extracted using **MediaPipe Pose**,  
    annotated with **activity labels** and **descriptions** in English.

    ## 📊 Dataset Structure
    - `train/` → 80% of the samples for training  
    - `validation/` → 20% of the samples for validation  

    Each record contains:
    - 33 pose keypoints (`x, y, z, visibility`)
    - `label`: activity name (e.g., `reading`, `dancing`, `office_work`)
    - `description`: textual description of the action context.

    ## ⚙️ Recommended Usage
    ```python
    from datasets import load_dataset
    dataset = load_dataset("guillherms/human-activity-pose_v2")
    train = dataset["train"]
    val = dataset["validation"]
    ```

    ## 🧠 Example

    | label | description |
    |--------|--------------|
    | reading | A person reading or reviewing a document attentively. |
    | waving | A person waving their hand as a greeting or farewell gesture. |
    | office_work | People working in an office, using laptops, papers, or mobile devices. |

    ## 🧩 Source
    Created by **Guilherme Santos** using MediaPipe and OpenCV.
    """

    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(readme)

    print("Uploading README.md...")
    api.upload_file(
        path_or_fileobj=readme_path,
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="dataset"
    )

    print("Dataset published successfully at:")
    print(f"https://huggingface.co/datasets/{repo_id}")


def convert_csv_to_parquet(csv_path: str, parquet_path: str) -> None:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    print("Loading CSV...")
    df = pd.read_csv(csv_path)

    print(f"CSV loaded with {len(df)} rows and {len(df.columns)} columns")

    print("Converting to Parquet format...")
    df.to_parquet(parquet_path, index=False)

    print(f"Parquet file saved at: {parquet_path}")
    
def get_readme(repo_id: str = "guillherms/human-activity-pose_v4") -> str:
    """
    Return the README content in English.
    The YAML front-matter begins on the first line to avoid the HF 'empty or missing yaml metadata' warning.
    Updated task fields use official Hugging Face task names to avoid YAML Metadata Warnings.
    """
    readme = textwrap.dedent(f"""\
    ---
    annotations_creators:
    - human-annotated
    language:
    - en
    license: cc-by-4.0
    pretty_name: "Human Activity Pose Dataset (Split Version)"
    task_categories:
    - keypoint-detection
    - image-classification
    task_ids:
    - pose-estimation
    - multi-class-classification
    size_categories:
    - 1K<n<10K
    ---

    # 🧍 Human Activity Pose Dataset (Split Version)

    This dataset contains human pose landmarks extracted with MediaPipe Pose,
    annotated with activity labels and textual descriptions in English.

    ## Dataset structure
    - `train/` — 80% of samples for training
    - `validation/` — 20% of samples for validation

    Each record includes:
    - 33 pose keypoints (fields: `x`, `y`, `z`, `visibility`)
    - `label`: activity name (e.g., `reading`, `dancing`, `office_work`)
    - `description`: a short textual description of the action context

    ## Recommended usage
    ```python
    from datasets import load_dataset
    dataset = load_dataset("{repo_id}")
    train = dataset["train"]
    val = dataset["validation"]
    ```

    ## Example labels

    | label | description |
    |-------|-------------|
    | reading | A person reading or reviewing a document attentively. |
    | waving  | A person waving their hand as a greeting or farewell gesture. |
    | office_work | People working in an office, using laptops, papers, or mobile devices. |

    ## Source
    Created by **Guilherme Santos** using MediaPipe and OpenCV.

    ## Notes
    - The YAML front-matter at the top of this file is used by Hugging Face Cards to populate the repo card. It must start on the first line (`---`) with no preceding blank lines or spaces.
    - License: CC-BY-4.0
    """)
    return readme

def upload_readme_to_hf(repo_id: str = "guillherms/human-activity-pose_v4", hf_token: Optional[str] = None, save_local: bool = False) -> None:
    """
    Upload the README (English) directly to the Hugging Face dataset repo.
    - repo_id: "username/repo_name"
    - hf_token: optional Hugging Face token (if None, local auth / env var is used)
    - save_local: if True, also saves README.md locally
    """
    readme = get_readme(repo_id=repo_id)
    if save_local:
        with open("README.md", "w", encoding="utf-8") as f:
            f.write(readme)
        print("README.md written locally.")

    # Prepare an in-memory bytes buffer for upload
    buffer = io.BytesIO(readme.encode("utf-8"))
    buffer.seek(0)

    api = HfApi(token=hf_token) if hf_token else HfApi()
    print(f"Uploading README.md to dataset repo: {repo_id} ...")
    api.upload_file(
        path_or_fileobj=buffer,
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="dataset",
    )
    print("Upload finished — check the dataset page to confirm the YAML warnings are gone.")



if __name__ == "__main__":
    '''
    create_landmark_dataset()
    convert_csv_to_parquet(
        csv_path="data/output/files/dataset_pose.csv",
        parquet_path="data/output/files/dataset_pose.parquet")
    '''
    #publish_landmark_dataset()
    upload_readme_to_hf()