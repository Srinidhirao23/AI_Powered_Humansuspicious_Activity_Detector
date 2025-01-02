from flask import Flask, render_template, url_for, request
import sqlite3
from A_Recognition import analyse
import os

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/userlog', methods=['GET', 'POST'])
def userlog():
    if request.method == 'POST':

        connection = sqlite3.connect('user_data.db')
        cursor = connection.cursor()

        name = request.form['name']
        password = request.form['password']

        query = "SELECT name, password FROM user WHERE name = '"+name+"' AND password= '"+password+"'"
        cursor.execute(query)

        result = cursor.fetchall()

        if len(result) == 0:
            return render_template('index.html', msg='Sorry, Incorrect Credentials Provided,  Try Again')
        else:
            return render_template('userlog.html')

    return render_template('index.html')


@app.route('/userreg', methods=['GET', 'POST'])
def userreg():
    if request.method == 'POST':

        connection = sqlite3.connect('user_data.db')
        cursor = connection.cursor()

        name = request.form['name']
        password = request.form['password']
        mobile = request.form['phone']
        email = request.form['email']
        
        print(name, mobile, email, password)

        command = """CREATE TABLE IF NOT EXISTS user(name TEXT, password TEXT, mobile TEXT, email TEXT)"""
        cursor.execute(command)

        cursor.execute("INSERT INTO user VALUES ('"+name+"', '"+password+"', '"+mobile+"', '"+email+"')")
        connection.commit()

        return render_template('index.html', msg='Successfully Registered')
    
    return render_template('index.html')

@app.route('/video', methods=['GET', 'POST'])
def video():
    if request.method == 'POST':
        
        fileName=request.form['filename']
        print(fileName)

        analyse('test/'+fileName)
        return render_template('userlog.html')
    return render_template('userlog.html')


@app.route('/Live')
def Live():
    f = open('temp.txt', 'w')
    f.write('0')
    f.close()
    os.system('python detect.py')
    return render_template('userlog.html')

if __name__ == "__main__":

    app.run(debug=True, use_reloader=False)
