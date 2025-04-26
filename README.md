
# 🏥 Medicine Recommendation System

A Machine Learning-powered web application that predicts diseases based on symptoms and recommends appropriate medicines.  
Built with ❤️ using Python, Flask, and Scikit-learn.

---

## ✨ Features
✅ Predicts possible diseases from user-selected symptoms  
✅ Recommends appropriate medicines for predicted diseases  
✅ Clean and interactive web interface (Flask + HTML/CSS)  
✅ Machine Learning models trained and compared for best performance  
✅ Deployed online and accessible to all  

---

## 🧰 Tech Stack

| Category         | Technologies Used           |
|------------------|------------------------------|
| Backend          | Python, Flask                |
| Machine Learning | Scikit-learn, Pandas, NumPy   |
| Frontend         | HTML, CSS                    |
| Deployment       | Render                       |

---

## 📂 Project Structure

```
Medicine-Recommendation-System/
├── kaggle_dataset/                  # Dataset files (Training, Symptoms, Medicines)
├── model/                           # Trained machine learning model
├── static/                          # Static files (images, CSS)
├── templates/                       # HTML templates (for Flask)
├── EDA_ModelComparison_RF.ipynb     # Exploratory Data Analysis & model comparison
├── disease_prediction_system.ipynb  # Notebook for building the prediction system
├── main.py                          # Flask application file
├── requirements.txt                 # Python dependencies
└── README.md                        # Project documentation
```

---

## 🚀 Getting Started Locally

### Prerequisites
- Python 3.8+
- pip

### Steps

#### 1. Clone the repository
```bash
git clone https://github.com/ShamScripts/Medicine-Recommendation-System.git
```
#### 2. Navigate to project folder
```bash
cd Medicine-Recommendation-System
```
#### 3. Install required dependencies
```bash
pip install -r requirements.txt
```
#### 4. Run the application
```bash
python main.py
```

### Access
Open your browser and visit:  
`http://127.0.0.1:5000`

---

## 🌐 Live Demo

Access the live project here:  
👉 [HealthCareAI](https://medicine-recommendation-system-jump.onrender.com/)

---

## 📊 Machine Learning Workflow

- Dataset cleaning and preprocessing
- Feature engineering and encoding
- Model training and evaluation (Random Forest, SVM, Naive Bayes)
- Model selection based on accuracy and performance metrics
- Saving the best model and integrating into Flask app

---

## 🧪 Test Cases

| Input Symptoms                  | Predicted Disease | Recommended Medicines         |
|----------------------------------|-------------------|--------------------------------|
| Fever, Cough, Headache           | Common Cold       | Paracetamol, Antihistamines    |
| Chest Pain, Shortness of Breath  | Heart Attack      | Aspirin, Nitroglycerin         |
| Fatigue, Frequent Urination      | Diabetes          | Metformin, Insulin             |

✅ Handles multiple symptoms  
✅ Handles no symptom input (shows warning)  
✅ Handles edge cases  

---

## 🛠 Future Enhancements

- Expand database to include more diseases and symptoms
- Add severity-based recommendations
- Improve UI/UX (multi-language support, better form controls)
- Mobile-friendly design
- Integrate chatbot assistant for health queries

---

## 📜 License

This project is intended for educational purposes only.  
Please consult a certified doctor for actual medical advice.

---

## 🤝 Connect With Me

- GitHub: [ShamScripts](https://github.com/ShamScripts)
- LinkedIn: [Shambhavi Jha](https://www.linkedin.com/in/shamscript009/)
---

# 🔥 Thank you for visiting!
