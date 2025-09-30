# 🎬 IMDB Sentiment Analysis

A Machine Learning project that performs **Sentiment Analysis** on IMDB movie reviews.  
It classifies user reviews as **Positive 😊** or **Negative 😞** using a Logistic Regression model trained on TF-IDF features.

---

## 📁 Project Structure

```
project_root/
│
├── data/
│   └── IMDB Dataset.csv           # Dataset file (IMDB reviews)
│
├── result/                        # Stores trained model and vectorizer
│   ├── sentiment_model.pkl
│   └── tfidf_vectorizer.pkl
│
└── src/
    ├── data_preprocessing.py      # Data cleaning and TF-IDF transformation
    ├── model.py                   # Model creation, saving, and loading
    └── train.py                   # Main script for training and testing
```

---

## ⚙️ Features

✅ Cleans and preprocesses IMDB reviews  
✅ Converts text to numerical form using **TF-IDF**  
✅ Trains a **Logistic Regression** model  
✅ Evaluates performance on test data  
✅ Saves trained model and vectorizer in `/result`  
✅ Allows **real-time user input** for sentiment prediction  

---

## 🧠 Tech Stack

- **Python 3.8+**
- **scikit-learn**
- **pandas**
- **BeautifulSoup4**
- **joblib**

---

## 🚀 How to Run

### 1️⃣ Clone this Repository
```bash
git clone git@github.com:<your-username>/imdb-sentiment-analysis.git
cd imdb-sentiment-analysis
```

### 2️⃣ Install Dependencies
Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate    # for Linux / macOS
venv\Scripts\activate       # for Windows
```

Install required packages:
```bash
pip install -r requirements.txt
```

> 💡 If you don’t have `requirements.txt`, create one with:
> ```bash
> pip install pandas scikit-learn beautifulsoup4 joblib
> pip freeze > requirements.txt
> ```

---

### 3️⃣ Add Dataset
Place your dataset file in the `data/` folder:
```
data/IMDB Dataset.csv
```

---

### 4️⃣ Train the Model
Move into the `src` folder and run:
```bash
cd src
python train.py
```

This will:
- Load and preprocess data  
- Train and evaluate the model  
- Save results to `/result`  
- Start **interactive prediction mode**

---

### 5️⃣ Try Your Own Reviews
After training, type any review to test sentiment:

```
💬 Sentiment Prediction Mode (type 'exit' to quit):

Enter a movie review: This movie was absolutely amazing!
👉 Sentiment: Positive 😄
```

---

## 📊 Example Output

```
✅ Accuracy: 0.8894
📊 Classification Report:
              precision    recall  f1-score   support
           0       0.88      0.89      0.88     5000
           1       0.89      0.89      0.89     5000
    accuracy                           0.89    10000
```

---

## 📦 Result Files

After training, your `/result` folder will contain:
```
sentiment_model.pkl         # Trained model
tfidf_vectorizer.pkl        # TF-IDF vectorizer
```

You can reuse them later for predictions without retraining.

---

## 💡 Future Improvements

- Upgrade to **Deep Learning (LSTM / BERT)**
- Build a **Flask / Streamlit web interface**
- Add **visualizations** for sentiment distribution

---

## 👨‍💻 Author

**Junaid Salman**  
📍 COMSATS University Lahore  
💼 React Developer | Machine Learning Enthusiast  
📧 junaidsalman@example.com  
📞 +92 3486892824  

---

## 🏷️ License

This project is open-source and available under the [MIT License](LICENSE).
