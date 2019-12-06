from flask import Flask
from flask import request
from testSample import testSample
import json
import os   

app = Flask(__name__)

@app.route("/")
def hello():
    return "Hello World!"


@app.route("/latency")
def latency():
    return ""

@app.route('/ans', methods=['POST'])
def postReq():
    if request.method == 'POST':
        f = request.files['file']
        f.save(f.filename)
        path = os.getcwd() + os.sep + f.filename
        print(path)
        user = request.form['UserName'] 
        classifier = request.form['ClassifierName']
        print(user)
        print(classifier)
        response= testSample(path,user,"Naive-Bayes")
        print(response)
        answer= {"status": response}
        return answer

@app.route('/register', methods=['POST'])
def postReqRegister():
    if request.method == 'POST':
        f = request.files['file']
        print("hello1")
        f.save(f.filename)
        print("hello2")
        path = os.getcwd() + os.sep + f.filename
        print(path)
        user = request.form['UserName'] 
        print(user)
        answer= {"status": "Success"}
        return answer


if __name__ == "__main__":
	app.run(ssl_context='adhoc')
