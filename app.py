from flask import Flask

def create_app(config_filename):
    app = Flask(__name__)
    app.config.from_object(config_filename)

    return app

app = create_app("config")

@app.route('/soil-model/predict/',methods=['GET','POST'])
def predict():
    response = "For ML Prediction"
    return {"message": response}	


if __name__ == "__main__":
    app.run(debug=True, port=3007)
