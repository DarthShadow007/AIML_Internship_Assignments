# 🚀 AIML Internship Assignments Repository

An organized collection of Artificial Intelligence and Machine Learning internship projects. This repository spans a wide range of AI domains, including classical machine learning, computer vision, reinforcement learning, and advanced natural language processing.

---

## 📂 Project Directory Structure

| Module ID | Assignment Name | Core Technology | Execution File |
| :--- | :--- | :--- | :--- |
| `01` | Adult Census Income Classification | Classical ML | `01_adult_census_income.py` |
| `02` | CIFAR-10 Image Classification | CNNs | `02_cifar10_cnn.py` |
| `02` | Movie Recommendation System | Collaborative Filtering | `02_movie_recommendation.py` |
| `03` | Cart-Pole RL Agent Training | Reinforcement Learning | `03_cartpole_rl.py` |
| `05` | Cancer Detection using MRI Images | Deep Learning (Vision) | `05_cancer_detection.py` |
| `06` | Face Recognition using CNN in Wild | CNNs | `06_face_recognition.py` |
| `07` | Lunar Lander RL Agent Training | Reinforcement Learning | `07_lunar_lander.py` |
| `08` | End to End Render Deployment | Cloud / MLOps | `08_render_deployment.py` |
| `09` | RAG Chatbot | LLM / Vector Search | `09_rag_chatbot.py` |

---

## 🔬 Assignment Details

### 1. Adult Census Income Classification
*   **Objective:** Build a predictive machine learning pipeline to classify if an individual's income exceeds a specific threshold using tabular census data.
*   **Key Files:** `01_adult_census_income.py`, `Screenshot 2026-07-31 201322.png`

### 2. CIFAR-10 Image Classification using CNN
*   **Objective:** Design and train a Convolutional Neural Network (CNN) to categorize images into 10 distinct classes.
*   **Key Files:** `02_cifar10_cnn.py`
*   **Mathematical Core:** Feature extraction utilizes the standard 2D discrete convolution operation:
    $$S(i, j) = (I * K)(i, j) = \sum_m \sum_n I(i-m, j-n) K(m, n)$$

### 3. Movie Recommendation System
*   **Objective:** Develop a collaborative filtering engine that recommends movies based on user viewing history leveraging the MovieLens 100k dataset.
*   **Key Files:** `02_movie_recommendation.py`, `ml-100k` directory, `ml-100k.zip`

### 4. Cart-Pole RL Agent Training
*   **Objective:** Implement a Deep Q-Network (DQN) that trains an autonomous agent to balance a pole on a moving cart within a physics simulation environment.
*   **Key Files:** `03_cartpole_rl.py`, `cartpole_dqn.pth` (Saved Model Weights)
*   **Mathematical Core:** The agent optimizes action-value estimations iteratively via the Bellman Equation:
    $$Q(s, a) \leftarrow Q(s, a) + \alpha \left[ r + \gamma \max_{a'} Q(s', a') - Q(s, a) \right]$$

### 5. Cancer Detection using MRI Images
*   **Objective:** Utilize deep learning techniques on medical imaging data (MRI scans) to detect and classify malignant tissues accurately.
*   **Key Files:** `05_cancer_detection.py`

### 6. Face Recognition using CNN in Wild Life
*   **Objective:** Construct an advanced facial recognition and verification model capable of identifying faces in unconstrained, real-world environments.
*   **Key Files:** `06_face_recognition.py`

### 7. Lunar Lander RL Agent Training
*   **Objective:** Train a reinforcement learning agent to safely pilot a spacecraft to a designated landing pad while managing fuel consumption and gravity constraints.
*   **Key Files:** `07_lunar_lander.py`, `lunar_lander_dqn.pth` (Saved Model Weights)

### 8. End to End Render Deployment Project
*   **Objective:** Bridge the gap between local modeling and production by deploying a trained machine learning model to the web using the Render cloud platform.
*   **Key Files:** `08_render_deployment.py`

### 9. RAG Chatbot
*   **Objective:** Engineer an autonomous AI assistant that uses Retrieval-Augmented Generation to search vector databases and synthesize grounded conversational responses.
*   **Key Files:** `09_rag_chatbot.py`

---

## 🚀 Getting Started

1.  Clone this repository to your local development environment.
2.  Ensure you have a modern version of Python (3.9+) installed on your machine.
3.  Install necessary AI/ML dependencies including PyTorch, TensorFlow, Scikit-Learn, OpenCV, and Gymnasium.
4.  Navigate into any of the specific project directories to review its dedicated local `README.md`.
5.  Execute the main Python script inside the respective directory to run the training pipelines or interface with the models. 