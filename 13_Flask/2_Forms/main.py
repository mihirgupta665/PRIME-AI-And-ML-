# each file has __name__ private variable with value __main__ inside the file and when imported or outsie the file the __name__ variabel value changes

from flask import Flask, render_template, request

app = Flask(__name__)

@app.route("/")
def hello_world():
    # query is sent in url at the end in ?key=value   [key is used to access the value]
    # query is exracted in flask server as arguments like request.args.get("key_string")
    query = request.args.get("q")
    print("Value of q "+query)
    # defult value to a query could be given which acts as the placeholder to be displayed when that specific query parameter is not specified
    firstname = request.args.get("name", default="anonymous")
    # jinja templating is used to send the query and receive them. which are accessible in html file with double curly and specified variable name.
    # so beside file to be render with commma jinja template variable name with its value is sent as { ,variable=value+value'}
    # multiple queries could be send with and in the url {   url?key1=value1&key2=value2 } and could be extracted individually
    return render_template("index.html", query=query, name=firstname)

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
