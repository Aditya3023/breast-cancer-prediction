from flask import Flask, request, render_template, jsonify
import numpy as np
import pickle
from sklearn.datasets import load_breast_cancer

app = Flask(__name__)

# Load model
model = pickle.load(open("model.pkl", "rb"))

# Load dataset to generate realistic random samples
data = load_breast_cancer()
X = data.data
y = data.target

@app.route('/')
def index():
    return render_template("index.html", random_sample="")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        input_str = request.form.get("feature")
        features = [float(x.strip()) for x in input_str.split(",")]
        np_features = np.asarray(features).reshape(1, -1)

        # Predict
        prob = model.predict_proba(np_features)[0][1]
        prediction = "Cancerous" if prob >= 0.5 else "Not Cancerous"

        return render_template("index.html",
                               message=[prediction, f"Probability: {prob*100:.2f}%"],
                               random_sample=input_str)
    except Exception as e:
        return render_template("index.html",
                               message=[f"Error: {str(e)}"],
                               random_sample="")

@app.route("/random")
def random_sample():
    # Pick random class (0=benign, 1=malignant)
    sample_class = np.random.choice([0, 1])
    samples_of_class = X[y == sample_class]
    sample = samples_of_class[np.random.randint(0, samples_of_class.shape[0])]
    sample_str = ",".join([f"{x:.6f}" for x in sample])
    return jsonify({"sample": sample_str})

if __name__ == "__main__":
    app.run(debug=True)
