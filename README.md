🚕 TripFare: Urban Taxi Fare Prediction

👩‍💻 Project by: Rathi Priya

This project is a machine learning-based web application that predicts taxi fares for urban trips using user input such as pickup/drop-off coordinates, time, and ride details. Built with Streamlit, it offers a user-friendly interface and displays essential Exploratory Data Analysis (EDA) insights.


---

🧠 Features

Interactive Streamlit web app

EDA visualizations like:

Fare vs Distance

Fare Distribution

Fare by Hour and Day


Predict taxi fare based on:

Pickup/Drop-off location

Passenger count

Pickup date and time

Rate code and Payment type


Dynamic feature engineering

Pre-trained ML model integration using Pickle



---

🔧 Tech Stack

Python

Streamlit

Pandas

Scikit-learn

Pickle

Matplotlib / Seaborn (for visualizations)



---

📦 Installation & Setup

1. Clone the repository

git clone https://github.com/your-username/tripfare-prediction.git
cd tripfare-prediction

2. Create and activate virtual environment

python -m venv venv
# For Windows
venv\Scripts\activate
# For macOS/Linux
source venv/bin/activate

3. Install dependencies

python -m pip install --upgrade pip
pip install -r requirements.txt

4. Run the application

streamlit run app.py


---

🗂 File Structure

🔼 tripfare-prediction
├── app.py                # Main Streamlit app
├── best_model.pkl        # Trained machine learning model
├── utils.py              # Feature engineering script
├── requirements.txt      # List of required Python packages
├── fare_vs_distance.png  # EDA plots
├── fare_distribution.png
├── fare_by_hour.png
├── fare_by_day.png


---

📝 How It Works

1. User provides trip details via input fields.


2. Inputs are transformed using engineer_features() from utils.py.


3. A trained model (best_model.pkl) is used to predict the fare.


4. The predicted fare is displayed instantly on the UI.




---

⚠ Note

If visualizations are missing, run the data exploration script to generate the EDA plots.

If model file is missing, ensure to train and save the model as best_model.pkl.



---

📬 Contact

For queries or improvements, contact Rathi Priya or open an issue in the repository.


---

⭐ If you find this project helpful, don't forget to give it a star!
