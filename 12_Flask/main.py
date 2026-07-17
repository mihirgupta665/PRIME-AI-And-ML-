# flask is a framework for server backend
# flask render html from templates folder and render styling and scripts from static folder
# to chnange static folder name pass static_foler="name_desired" at the time of application creation
# In flask static files could be accessed with roothost+hierarchy and if you want to change the static file access then pass static_url_path="assets_new" at the time of applciation generation

from flask import Flask, render_template, url_for

app = Flask(__name__, static_folder="assets", static_url_path="/assets_new")

@app.route("/")
def hello_world():
    #  static file => dynamically generate the url
    print(url_for("static", filename="style2.css"))
    return render_template("index.html")

@app.route("/login")
def login():
    return render_template("login.html")



if __name__ == "__main__":
    app.run(debug=True)