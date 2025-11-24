from flask import Flask, request, jsonify
import tensorflow as tf

app = Flask(__name__)
model = tf.keras.models.load_model("/opt/ml/model/model.h5")

@app.route("/ping")
def ping():
    return "pong"

@app.route("/invocations", methods=["POST"])
def predict():
    data = request.get_json(force=True)
    # assume data['instances'] is list of flattened arrays
    preds = model.predict(data['instances'])
    return jsonify(preds.tolist())

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8080)
