# 🎥 Video Analysis Tech Challenge (English Version)

Welcome to the **Video Analysis** project! This repository contains the solution for the proposed challenge, aiming to implement an application capable of performing video analysis using computer vision and machine learning techniques.

---

## 📝 Challenge Description

The project addresses the following problem:

> **THE PROBLEM:**  
> The Tech Challenge in this phase consists of creating an application that uses video analysis.  
> The project must incorporate techniques for facial recognition, emotional expression analysis in videos, and activity detection.

### 📋 Required Features

1. **Facial recognition**: Identify and mark faces present in the video.
2. **Emotional expression analysis**: Analyze the emotional expressions of the detected faces.
3. **Activity detection**: Detect and categorize the activities performed in the video.
4. **Summary generation**: Automatically create a summary of the main activities and emotions detected in the video.

---

## 🚀 Technologies Used

- Python
- OpenCV
- Deep Learning (e.g., TensorFlow, Keras, or PyTorch)
- Pre-trained models for face detection and emotion analysis
- Other helper libraries for video processing and visualization

---

## ⚙️ How to Run

1. Clone this repository:
   ```bash
   git clone https://github.com/guillherms/video-analysis.git
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Place the provided video in the `input/` folder (create it if necessary).
4. Run the main script:
   ```bash
   python main.py
   ```
5. Results will be saved in the `output/` folder.

---

## 📑 Project Structure

```
video-analysis/
├── input/                 # Input videos
├── output/                # Generated results and summaries
├── src/                   # Main source code
│   ├── facial_recognition.py
│   ├── emotion_analysis.py
│   └── activity_detection.py
├── requirements.txt
└── main.py
```

---

## 🛠️ Development Log

| Date       | Change                                                                                      |
|------------|--------------------------------------------------------------------------------------------|
| 2024-06-01 | 🎉 Project started and proposal defined                                                     |
| 2024-06-02 | 👤 Initial implementation of facial recognition                                             |
| 2024-06-03 | 😊 Added emotional expression analysis to detected faces                                    |
| 2024-06-04 | 🏃‍♂️ Implemented activity detection and categorization in the video                        |
| 2024-06-05 | 📝 Automatic generation of summary for detected activities and emotions                     |
| 2024-06-06 | 🐞 Bug fixes and improvements in result visualization interface                             |
| 2024-06-07 | 📦 Code refactoring and documentation update                                                |
| ...        | ...                                                                                        |

---

## 💡 Possible Future Improvements

- Support for multiple videos simultaneously
- Graphical interface for easier use
- Export results in different formats (PDF, Excel)
- Integration with third-party APIs for advanced analysis

---

## 📄 License

This project is for academic purposes only.

---

## 🙋‍♂️ Contact

Questions or suggestions?  
Contact via [GitHub](https://github.com/guillherms) or [LinkedIn](https://www.linkedin.com/in/guilherme-santos-de-oliveira-ba9986161/)

---