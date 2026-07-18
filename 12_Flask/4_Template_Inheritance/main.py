# each file has __name__ private variable with value __main__ inside the file and when imported or outsie the file the __name__ variabel value changes

from flask import Flask, render_template

app = Flask(__name__)

@app.route("/")
def hello_world():
    return render_template("index.html")

@app.route("/contact")
def contact():    
    return render_template("contact.html")
    


if __name__ == "__main__":
    app.run(debug=True) # debug=True updates the page automatically whenever there is a change in code
