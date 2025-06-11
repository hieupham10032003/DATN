from werkzeug.security import generate_password_hash, check_password_hash
import mysql.connector
from flask import Flask, render_template, request, redirect, url_for, session
from utils import (
    etl_process,
    load_to_mysql,
    process_and_load_data,
    analyze_data,
    plot_data,
    train_lstm,
    evaluate_model
)
import os

app = Flask(__name__)
app.secret_key = 'hieu10032003'  

# Kết nối tới MySQL
def get_db_connection():
    return mysql.connector.connect(
        host='localhost',
        user='root',
        password='hieu10032003',
        database='giao_dien'
    )

# Hàm đăng ký
def register_user(username, password):
    hashed_password = generate_password_hash(password)
    connection = get_db_connection()
    cursor = connection.cursor()
    cursor.execute("INSERT INTO users (username, password) VALUES (%s, %s)", (username, hashed_password))
    connection.commit()
    cursor.close()
    connection.close()

# Hàm đăng nhập
def login_user(username, password):
    connection = get_db_connection()
    cursor = connection.cursor()
    cursor.execute("SELECT password FROM users WHERE username = %s", (username,))
    user = cursor.fetchone()
    cursor.close()
    connection.close()
    if user and check_password_hash(user[0], password):
        return True
    return False

@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")
        register_user(username, password)
        return redirect(url_for('login'))
    return render_template("register.html")

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")
        if login_user(username, password):
            session['user'] = username
            return redirect(url_for('index'))  
        else:
            return "Đăng nhập không thành công."
    return render_template("login.html")

@app.route("/", methods=["GET", "POST"])
def index():
    if 'user' not in session:  # Kiểm tra xem người dùng đã đăng nhập chưa
        return redirect(url_for('login'))  # Chuyển hướng đến trang đăng nhập nếu chưa đăng nhập

    message = ""
    if request.method == "POST":
        tickers_input = request.form.get("tickers", "")
        tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]
        action = request.form.get("action")

        if not tickers:
            message = "Vui lòng nhập ít nhất một ticker."
        else:
            try:
                if action == "etl":
                    message = etl_process(tickers)
                elif action == "mysql":
                    results = []
                    for ticker in tickers:
                        file_path = f"{ticker}_data.csv"
                        if os.path.exists(file_path):
                            results.append(load_to_mysql(file_path, f"{ticker}_table"))
                    message = "<br>".join(results)
                elif action == "spark":
                    process_and_load_data(tickers)
                    message = "Dữ liệu đã được tải vào Spark."
                elif action == "analyze":
                    message = analyze_data(tickers).replace("\n", "<br>")
                elif action == "plot":
                    plot_data(tickers)
                    message = "Biểu đồ đã hiển thị."
                elif action == "train":
                    train_lstm(tickers)
                    message = "Huấn luyện và dự đoán hoàn tất."
                elif action == "evaluate":
                    for ticker in tickers:
                        evaluate_model(ticker, f"{ticker}_data.csv")
                    message = "Đánh giá mô hình hoàn tất."
            except Exception as e:
                message = f"Lỗi: {e}"
    
    return render_template("index.html", message=message)

if __name__ == "__main__":
    app.run(debug=True)