# 🚀 Projectile Motion Predictor - Streamlit App (Air Resistance Included)

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# ----------------------------
# App Configuration
# ----------------------------
st.set_page_config(page_title="Projectile Motion Predictor", page_icon="🚀", layout="centered")
st.title("🚀 Projectile Motion Predictor")
st.markdown("""
Use this app to predict **Range**, **Time of Flight**, and **Maximum Height** of a projectile using Machine Learning.
Input the **initial velocity** and **launch angle**, and get accurate predictions based on physics-trained data with optional air resistance!
""")

# ----------------------------
# Generate Synthetic Data with Optional Air Resistance
# ----------------------------
def simulate_projectile(v0, theta_deg, k=0.05, dt=0.01):
    g = 9.81
    theta = np.radians(theta_deg)
    vx = v0 * np.cos(theta)
    vy = v0 * np.sin(theta)
    x, y = 0, 0
    t = 0

    x_vals, y_vals, t_vals = [0], [0], [0]

    while y >= 0:
        v = np.sqrt(vx**2 + vy**2)
        ax = -k * v * vx
        ay = -g - k * v * vy

        vx += ax * dt
        vy += ay * dt

        x += vx * dt
        y += vy * dt
        t += dt

        x_vals.append(x)
        y_vals.append(y)
        t_vals.append(t)

    return x_vals, y_vals, t_vals[-1], max(y_vals)

def generate_data(n_samples=1000, air_resistance=True):
    data = []
    for _ in range(n_samples):
        v0 = np.random.uniform(5, 100)
        theta = np.random.uniform(10, 80)

        if air_resistance:
            x_vals, y_vals, t_flight, h_max = simulate_projectile(v0, theta)
            rng = x_vals[-1]
        else:
            g = 9.81
            theta_rad = np.radians(theta)
            rng = (v0**2) * np.sin(2 * theta_rad) / g
            t_flight = (2 * v0 * np.sin(theta_rad)) / g
            h_max = (v0**2) * (np.sin(theta_rad)**2) / (2 * g)

        data.append([v0, theta, rng, t_flight, h_max])

    df = pd.DataFrame(data, columns=['v0', 'theta', 'range', 'time', 'height'])
    return df

# ----------------------------
# Train and Compare Models
# ----------------------------
def train_models(data):
    X = data[['v0', 'theta']]
    y = data[['range', 'time', 'height']]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)

    models = {
        'Gradient Boosting': MultiOutputRegressor(GradientBoostingRegressor(n_estimators=500, learning_rate=0.05, max_depth=5, random_state=42)),
        'Random Forest': MultiOutputRegressor(RandomForestRegressor(n_estimators=300, max_depth=10, random_state=42)),
        'Linear Regression': MultiOutputRegressor(LinearRegression())
    }

    trained_models = {}
    scores = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        trained_models[name] = model
        scores[name] = (rmse, r2)

    return trained_models, scores

# ----------------------------
# Load Data and Train
# ----------------------------
st.subheader("🔧 Training the models...")
air_resistance_toggle = st.toggle("Include Air Resistance", value=True)
data = generate_data(air_resistance=air_resistance_toggle)
models, scores = train_models(data)
selected_model = models['Gradient Boosting']

best_model_name = min(scores, key=lambda x: scores[x][0])
st.success(f"✅ Best Model: {best_model_name} | RMSE: {scores[best_model_name][0]:.5f}, R²: {scores[best_model_name][1]:.4f}")

# ----------------------------
# User Input Section
# ----------------------------
st.subheader("📌 Enter Projectile Parameters")
user_v0 = st.slider("Initial Velocity (m/s)", 5, 1000, 500)
user_theta = st.slider("Launch Angle (degrees)", 0, 90, 45)

# Prediction and Trajectory
if st.button("🎯 Predict Motion"):
    input_df = pd.DataFrame([[user_v0, user_theta]], columns=['v0', 'theta'])
    result = selected_model.predict(input_df)[0]
    rng, t_flight, h_max = result

    st.markdown("### 📊 Predicted Results:")
    st.metric("Range (m)", f"{rng:.5f}")
    st.metric("Time of Flight (s)", f"{t_flight:.5f}")
    st.metric("Maximum Height (m)", f"{h_max:.5f}")

    if air_resistance_toggle:
        x_vals, y_vals, _, _ = simulate_projectile(user_v0, user_theta)
    else:
        g = 9.81
        theta_rad = np.radians(user_theta)
        t_vals = np.linspace(0, t_flight, num=500)
        x_vals = user_v0 * np.cos(theta_rad) * t_vals
        y_vals = user_v0 * np.sin(theta_rad) * t_vals - 0.5 * g * t_vals**2

    fig, ax = plt.subplots()
    ax.plot(x_vals, y_vals)
    ax.set_title("🧭 Projectile Trajectory" + (" (with Air Resistance)" if air_resistance_toggle else ""))
    ax.set_xlabel("Range (m)")
    ax.set_ylabel("Height (m)")
    ax.grid(True)
    st.pyplot(fig)

# Model Comparison Table
st.subheader("📈 Model Comparison")
comparison_df = pd.DataFrame(scores, index=['RMSE', 'R² Score']).T
st.dataframe(comparison_df.style.format({"RMSE": "{:.5f}", "R² Score": "{:.4f}"}))

# Footer
st.markdown("---")
st.markdown("Made with ❤️ using Streamlit | Powered by Physics & ML")
