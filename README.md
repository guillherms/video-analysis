
    ---
    annotations_creators:
    - human-annotated
    language:
    - en
    license: cc-by-4.0
    pretty_name: Human Activity Pose Dataset (Split Version)
    task_categories:
    - pose-estimation
    - action-recognition
    task_ids:
    - human-activity-recognition
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
    