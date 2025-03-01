

# 🎬 Movie Recommendation System  

A **Movie Recommendation System** built using **machine learning techniques**, including **Content-Based Filtering, Collaborative Filtering, and a Hybrid Model**. The system suggests movies based on user preferences and ratings, deployed as a **Flask web application** with an interactive UI.  

---

## 📸 Screenshots  

### 🔹 **Home Page**  

![Screenshot (29)](https://github.com/user-attachments/assets/4ebd6455-b67d-4290-a0dd-e87bd4668e6f)


### 🔹 **Movie Recommendation Results**  
![Screenshot (40)](https://github.com/user-attachments/assets/a06aa223-a225-40c3-bd4b-cf60c079ae51)
![Screenshot (41)](https://github.com/user-attachments/assets/1ef88cb0-14c5-48b5-8183-7ee014013414)



### 🔹 **Watchlist & Search History**  
![Screenshot (43)](https://github.com/user-attachments/assets/39750786-1da2-4376-8a43-29083ebd2fc2)

### Data Analysis 
![image](https://github.com/user-attachments/assets/a0aa96b1-a2f3-4942-86ce-4167b55c96e6)
![image](https://github.com/user-attachments/assets/1a7c7cca-1c43-45f7-8afe-f96231dd306b)
![image](https://github.com/user-attachments/assets/da83b55e-f7de-4e36-9ae3-a6ae6fcbc05b)



## 📂 Project Structure  

```
MRfinalP/
│── templates/                  # HTML templates for the web application
│── .gitattributes               # Git configuration file
│── app.py                        # Flask application for deployment
│── jpnb1, jpnb2, jpnb3         # Jupyter Notebooks for model development
│── movies_metadata.xlsx        # Raw movie dataset
│── movies_metadata_cleaned.xlsx # Processed dataset
│── ratings_small.xlsx          # User ratings dataset
│── screenshots/                # Folder for project screenshots
```

---

## 🚀 Features  

✅ **Personalized Movie Recommendations**  
✅ **Hybrid Model (CBF + CF) for better accuracy**  
✅ **Web-based Interface (Flask, HTML, Bootstrap)**  
✅ **Auto-Suggestions & Search History**  
✅ **Watchlist Feature for Users**  
✅ **Suggests Platforms where Movies are Available**  

---

## 🛠️ Technologies Used  

- **Programming Language:** Python  
- **Libraries:** Pandas, NumPy, Scikit-Learn, Surprise, Flask  
- **Web Technologies:** HTML, CSS (Bootstrap), JavaScript  
- **Tools:** Jupyter Notebook, Visual Studio Code  

---

## ⚡ Getting Started  

Follow these steps to set up and run the project locally.  

### 🔹 **Prerequisites**  

- Install **Python 3.x**  
- Install **pip** (Python package manager)  
- Install **Flask & required dependencies**  

### 🔹 **Installation & Setup**  

1️⃣ **Clone the repository**  
```bash
git clone https://github.com/your-username/movie-recommendation-system.git
cd movie-recommendation-system
```
  
2️⃣ **Install dependencies**  
```bash
pip install -r requirements.txt
```
  
3️⃣ **Run the Flask application**  
```bash
python app.py
```
  
4️⃣ **Open your browser and go to**  
```bash
http://127.0.0.1:5000
```

---

## 🎯 Recommendation Models  

### 🔹 **1. Content-Based Filtering (CBF)**  
- Uses movie overviews and genres to find similar movies.  
- **TF-IDF** vectorization + **Cosine Similarity** for similarity detection.  

### 🔹 **2. Collaborative Filtering (CF)**  
- Predicts ratings for unseen movies based on user preferences.  
- Uses **Singular Value Decomposition (SVD)** for matrix factorization.  

### 🔹 **3. Hybrid Model (CBF + CF)**  
- Uses **CBF for new users** and **CF for existing users**.  
- Provides **highly personalized recommendations**.  

---

## 📊 Results  

✅ **Content-Based Filtering** → Works well but lacks user preference data.  
✅ **Collaborative Filtering** → More personalized but needs sufficient user data.  
✅ **Hybrid Model** → Best performance, combining both approaches.  

---

## 📅 Future Enhancements  

📌 **Deep Learning Models** (Neural Networks for recommendations)  
📌 **User Authentication & Watch History**  
📌 **Real-time Feedback & Adaptive Suggestions**  

---

## 📜 License  

This project is **open-source** and available under the **MIT License**.  

---

### ⭐ **Like this project? Give it a star! 🌟**  

---

