# 🚀 Projectile Motion Predictor - Streamlit App (with Air Resistance)

An interactive Streamlit app that predicts **Range**, **Time of Flight**, and **Maximum Height** for a projectile using machine learning models trained on synthetic physics-based data.

Includes an option to simulate with or without **air resistance** and offers visual trajectory plots and model comparison metrics.

---

## 🧪 Features

- ✅ Predict **Range**, **Time of Flight**, and **Max Height**
- 🎛️ Choose between models: Gradient Boosting, Random Forest, Linear Regression
- 🔄 Toggle **Air Resistance**
- 📊 Visualize real-time projectile **trajectory**
- 📈 Compare model performance with **RMSE** and **R²**
- ⚙️ Interactive sliders for launch angle and velocity
- 💻 Built with **Streamlit**, **scikit-learn**, **matplotlib**, and **NumPy**

---

## 📦 Requirements

Install required libraries using pip:

```bash
pip install streamlit scikit-learn matplotlib numpy pandas
```

---

## ▶️ How to Run

In your terminal:

```bash
streamlit run projectile_motion_predictor.py
```

---

## 📌 User Inputs

- **Initial Velocity (v₀)**: 5 – 1000 m/s
- **Launch Angle (θ)**: 0 – 90 degrees
- **Air Resistance Toggle**: Include or exclude real-world drag

---

## 📉 Output Metrics

| Metric               | Description                          |
|----------------------|--------------------------------------|
| **Range (m)**        | Total horizontal distance            |
| **Time of Flight (s)** | Time projectile remains in air     |
| **Maximum Height (m)**| Peak vertical position              |

---

## 📊 Visual Output

- **2D Trajectory Plot** (with or without air resistance)
- **Model Comparison Table** (RMSE, R²)

---

## 🧠 Models Used

| Model               | Description                                 |
|---------------------|---------------------------------------------|
| Gradient Boosting   | High-performance, nonlinear regression      |
| Random Forest       | Ensemble-based, interpretable               |
| Linear Regression   | Fast, linear baseline model                 |

---

## 📁 File Structure

```
projectile_motion_predictor.py    # Streamlit app source code
README.md                         # Project documentation
```

---

## 📚 References

- Physics-based equations for projectile motion
- Scikit-learn regressors with `MultiOutputRegressor`
- Streamlit for interactive UI and visualizations

---

## 👨‍💻 Author

Made with ❤️ by [@Mrusmangoraya](https://github.com/Mrusmangoraya)  
Powered by Physics + Machine Learning 🤝

---
