# each file has __name__ private variable with value __main__ inside the file and when imported or outsie the file the __name__ variabel value changes

from flask import Flask, render_template, request

app = Flask(__name__)

@app.route("/")
def hello_world():
    return render_template("index.html")

@app.route("/login")
def login():
    return render_template("login.html")

@app.route("/handle-login", methods=["GET", "POST"])
def handle_login():
    if request.method == "POST":
        return "<p>POST request</p>"
    if request.method == "GET":
        return "<p>GET request</p>"
    return "<p>This route is to handle login</p>"



if __name__ == "__main__":
    app.run(debug=True) # degub=True updates the page automatically whenever there is a change in code