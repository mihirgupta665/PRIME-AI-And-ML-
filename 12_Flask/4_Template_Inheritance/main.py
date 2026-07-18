# each file has __name__ private variable with value __main__ inside the file and when imported or outsie the file the __name__ variabel value changes

from flask import Flask, render_template, flash, redirect, url_for

app = Flask(__name__)

app.secret_key = "some secret message or key"

app.secret_key = "some secret message or key"

@app.route("/")
def hello_world():
    # redirect to another route
    return redirect(url_for("login"))  # dynamic url generation for login route

@app.route("/login")
def login():
    return "<p>login page</p>"

@app.route("/contact")
def contact():
    flash("support timings are from 9 to 5")    
    return render_template("contact.html")
    


if __name__ == "__main__":
    app.run(debug=True) # debug=True updates the page automatically whenever there is a change in code
