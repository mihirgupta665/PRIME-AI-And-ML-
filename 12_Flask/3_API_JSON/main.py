# each file has __name__ private variable with value __main__ inside the file and when imported or outsie the file the __name__ variabel value changes

from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

@app.route("/")
def hello_world():
    
    data= {
        "message": "Welcome to the platform"
    }

# jsonify(dict) : converts a python dictionary to json data. We could alos padd status code along with the data
    return jsonify(data), 200

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        print(request.form)  # POST request data can be accessed through request.form
        name = request.form["username"]
        password = request.form["password"]
        return f"<p>Welcome {name}!</p>"
    
    else:
        return render_template("login.html")
    


if __name__ == "__main__":
    app.run(debug=True) # degub=True updates the page automatically whenever there is a change in code
