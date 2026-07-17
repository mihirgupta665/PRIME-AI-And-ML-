from flask import Flask

app = Flask(__name__)

@app.route("/")
def hello_world():
    return "<p>Hello, Flask World. This side Mihir Gupta reporting for the duty!</p>"

@app.route("/prime")
def prime():
    return "<p>Hello from Prime this time</p>"


app.run(debug=True)