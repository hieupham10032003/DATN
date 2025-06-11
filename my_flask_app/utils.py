import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dropout, Dense
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
from sqlalchemy import create_engine, text  
from pyspark.sql import SparkSession
from sklearn.metrics import r2_score, mean_absolute_error, mean_absolute_percentage_error
import os

def etl_process(tickers):
    end_date = datetime.now()
    start_date = end_date - timedelta(days=3650)

    for ticker in tickers:
        stock_data = yf.download(ticker, start=start_date, end=end_date, interval='1d')
        stock_data.reset_index(inplace=True)
        stock_data['Ticker'] = ticker

        # Đổi tên cột sang tiếng Việt
        stock_data.rename(columns={
            'Date': 'Ngày',
            'Close': 'Đóng cửa',
            'High': 'Cao nhất',
            'Low': 'Thấp nhất',
            'Open': 'Mở cửa',
            'Volume': 'Số lượng',
            'Ticker': 'Nhãn dán'
        }, inplace=True)

        stock_data.to_csv(f'{ticker}_data.csv', index=False)

    return "Dữ liệu đã được tải về và lưu CSV."

# ---------- 2. Lưu vào MySQL ----------
def load_to_mysql(file_path, table_name):
    DB_USER = 'root'
    DB_PASSWORD = 'hieu10032003'
    DB_HOST = 'localhost'
    DB_PORT = '3306'
    DB_NAME = 'giao_dien'

    df = pd.read_csv(file_path)

    # Thêm cột Id tự tăng bắt đầu từ 1
    df.insert(0, 'Id', range(1, len(df) + 1))

    engine = create_engine(f'mysql+mysqlconnector://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}')
    
    # Lưu dữ liệu vào MySQL
    df.to_sql(name=table_name, con=engine, if_exists='append', index=False)

    # Thiết lập Id là khóa chính
    with engine.connect() as conn:
        conn.execute(text(f"ALTER TABLE {table_name} ADD PRIMARY KEY (Id);"))

    return f"Dữ liệu từ {file_path} đã được lưu vào MySQL bảng {table_name} .."

# ---------- 3. Đẩy lên Spark ----------
# Load: Chèn dữ liệu vào Spark
def load_to_spark(dataframe):
    # Khởi tạo Spark
    spark = SparkSession.builder \
        .appName("Stock Data ETL") \
        .config("spark.driver.extraClassPath", "/path/to/mysql-connector-java-x.x.x.jar") \
        .config("spark.ui.port", "4040") \
        .getOrCreate()
    
    # Chuyển đổi Pandas DataFrame thành Spark DataFrame
    spark_df = spark.createDataFrame(dataframe)
    spark_df.createOrReplaceTempView("stocks_temp_view")

# Đọc dữ liệu từ file CSV và hiển thị nội dung
def read_csv_to_spark(csv_path):
    # Khởi tạo SparkSession
    spark = SparkSession.builder \
        .appName("CSV to Spark DataFrame") \
        .getOrCreate()
    
    # Đọc dữ liệu từ file CSV
    df = spark.read.csv(csv_path, header=True, inferSchema=True)
    # Hiển thị nội dung của DataFrame
    df.show()
    return df

# Hàm xử lý và tải dữ liệu từ nhiều tệp CSV
def process_and_load_data(tickers):
    for ticker in tickers:
        try:
            # Đọc tệp CSV cho mỗi ticker
            df_pandas = pd.read_csv(f'{ticker}_data.csv')  # Giả sử tên tệp CSV theo định dạng: TICKER_data.csv
            
            # Chèn dữ liệu vào Spark
            load_to_spark(df_pandas)
            
            # Đọc lại dữ liệu CSV vào Spark DataFrame và hiển thị
            df_spark = read_csv_to_spark(f'{ticker}_data.csv')
        except Exception as e:
            print(f"Lỗi khi xử lý {ticker}: {e}")
    
    print("Dữ liệu đã được tải vào Spark thành công!")
# ---------- 4. Đọc và phân tích dữ liệu ----------
def analyze_data(tickers):
    output = ""
    for ticker in tickers:
        try:
            df = pd.read_csv(f"{ticker}_data.csv")
            df = df.drop(columns=["Số lượng", "Nhãn dán"], errors='ignore')
            df["Ngày"] = pd.to_datetime(df["Ngày"], errors='coerce')
            output += f"\n Dữ liệu cho {ticker}:\n"
            output += f"  - Hình dạng: {df.shape}\n"
            output += f"  - Mô tả thống kê:\n{df.describe().to_string()}\n"
        except Exception as e:
            output += f"\n Lỗi với {ticker}: {e}\n"
    return output

# ---------- 5. Vẽ biểu đồ ----------
def plot_data(tickers):
    for ticker in tickers:
        try:
            df = pd.read_csv(f"{ticker}_data.csv")
            df = df.drop(columns=["Số lượng", "Nhãn dán"], errors='ignore')
            df['Ngày'] = pd.to_datetime(df['Ngày'], errors='coerce')
            df = df.dropna(subset=['Ngày']).sort_values(by='Ngày')
            for col in ['Đóng cửa', 'Mở cửa', 'Cao nhất', 'Thấp nhất']:
                df[col] = df[col].astype(str).str.replace(',', '').astype(float)
            plt.figure(figsize=(10, 5))
            plt.plot(df['Ngày'], df['Đóng cửa'], label='Giá đóng cửa')
            plt.xlabel('Năm')
            plt.ylabel('Giá')
            plt.title(f'Biểu đồ giá đóng cửa {ticker}')
            plt.legend()
            plt.tight_layout()
            plt.show()
        except Exception as e:
            print(f"Lỗi khi vẽ biểu đồ {ticker}: {e}")

# ---------- 6. Huấn luyện mô hình ----------
def train_lstm(tickers):
    for ticker in tickers:
        try:
            df = pd.read_csv(f"{ticker}_data.csv")
            df = df.drop(columns=["Số lượng", "Nhãn dán"], errors='ignore')
            df['Ngày'] = pd.to_datetime(df['Ngày'], errors='coerce')
            df = df.dropna(subset=["Ngày"]).sort_values(by="Ngày")
            for col in ['Đóng cửa', 'Mở cửa', 'Cao nhất', 'Thấp nhất']:
                df[col] = df[col].astype(str).str.replace(',', '').astype(float)
            df1 = pd.DataFrame(df, columns=["Ngày", "Đóng cửa"])
            df1.index = df1["Ngày"]
            df1.drop("Ngày", axis=1, inplace=True)

            data = df1.values
            train_data = data[:2514]
            test_data = data[2514:]

            sc = MinMaxScaler()
            sc_data = sc.fit_transform(data)

            x_train, y_train = [], []
            for i in range(50, len(train_data)):
                x_train.append(sc_data[i-50:i, 0])
                y_train.append(sc_data[i, 0])
            x_train = np.reshape(np.array(x_train), (len(x_train), 50, 1))
            y_train = np.reshape(np.array(y_train), (-1, 1))

            model = Sequential([
                LSTM(128, return_sequences=True, input_shape=(50, 1)),
                LSTM(64),
                Dropout(0.5),
                Dense(1)
            ])
            model.compile(loss='mean_absolute_error', optimizer='adam')
            model_path = f"{ticker}_best_model.keras"
            checkpoint = ModelCheckpoint(model_path, monitor='loss', save_best_only=True)
            model.fit(x_train, y_train, epochs=100, batch_size=50, verbose=0, callbacks=[checkpoint])

            # Dự đoán và vẽ
            model = load_model(model_path)
            y_train_pred = model.predict(x_train)
            y_train_pred = sc.inverse_transform(y_train_pred)
            df1_pred = df1.copy()
            df1_pred["Dự đoán"] = np.nan
            df1_pred.iloc[50:2514, 1] = y_train_pred.flatten()
            plt.figure(figsize=(15, 5))
            plt.plot(df1, label="Thực tế", color='red')
            plt.plot(df1_pred["Dự đoán"], label="Dự đoán", color='green')
            plt.title(f"Dự đoán LSTM cho {ticker}")
            plt.legend()
            plt.tight_layout()
            plt.show()
        except Exception as e:
            print(f"Lỗi LSTM {ticker}: {e}")
# ---------- 7. Đánh giá mô hình ----------
def evaluate_model(company, file_name):
    print(f"\n Kiểm tra độ chính xác và dự đoán tiếp theo cho: {company}")
    try:
        df = pd.read_csv(file_name)
        df = df.drop(columns=["Số lượng", "Nhãn dán"], errors='ignore')
        df['Ngày'] = pd.to_datetime(df['Ngày'], format='%Y-%m-%d', errors='coerce')
        df = df.dropna(subset=["Ngày"])
        df = df.sort_values(by="Ngày")
        for col in ['Đóng cửa', 'Mở cửa', 'Cao nhất', 'Thấp nhất']:
            df[col] = df[col].astype(str).str.replace(',', '').astype(float)
        df1 = pd.DataFrame(df, columns=["Ngày", "Đóng cửa"])
        df1.index = df1["Ngày"]
        df1.drop("Ngày", axis=1, inplace=True)
        
        # Dữ liệu
        data = df1.values
        train_data = data[:1500]
        test_data = data[1500:]
        
        sc = MinMaxScaler(feature_range=(0, 1))
        sc_data = sc.fit_transform(data)
        
        # Xử lý dữ liệu huấn luyện
        x_train, y_train = [], []
        for i in range(50, len(train_data)):
            x_train.append(sc_data[i-50:i, 0])
            y_train.append(sc_data[i, 0])
        x_train, y_train = np.array(x_train), np.array(y_train)
        x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
        y_train = np.reshape(y_train, (y_train.shape[0], 1))
        
        # Tải mô hình đã huấn luyện
        model_path = f"{company}_best_model.keras"
        final_model = load_model(model_path)
        
        # Dự đoán và đánh giá trên tập train
        y_train_predict = final_model.predict(x_train)
        y_train_predict = sc.inverse_transform(y_train_predict)
        y_train_true = sc.inverse_transform(y_train)
        
        # Dự đoán trên tập test
        test_input = df1[len(train_data)-50:].values.reshape(-1, 1)
        test_input_scaled = sc.transform(test_input)
        x_test = []
        for i in range(50, test_input_scaled.shape[0]):
            x_test.append(test_input_scaled[i-50:i, 0])
        x_test = np.array(x_test)
        x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
        y_test_true = data[1500:]
        y_test_predict = final_model.predict(x_test)
        y_test_predict = sc.inverse_transform(y_test_predict)
        
        # Đánh giá độ chính xác
        print('Tập TRAIN:')
        print('  - R2 score:', r2_score(y_train_true, y_train_predict))
        print('  - MAE:', mean_absolute_error(y_train_true, y_train_predict))
        print('  - MAPE:', mean_absolute_percentage_error(y_train_true, y_train_predict))
        print('Tập TEST:')
        print('  - R2 score:', r2_score(y_test_true, y_test_predict))
        print('  - MAE:', mean_absolute_error(y_test_true, y_test_predict))
        print('  - MAPE:', mean_absolute_percentage_error(y_test_true, y_test_predict))
        
        # Dự đoán giá ngày kế tiếp
        next_date = df['Ngày'].iloc[-1] + pd.Timedelta(days=1)
        last_50 = sc_data[-50:, 0].reshape(1, -1, 1)
        y_next_predict = final_model.predict(last_50)
        y_next_predict = sc.inverse_transform(y_next_predict)
        
        # So sánh với giá ngày cuối
        actual_closing_price = df['Đóng cửa'].iloc[-1]
        comparison_df = pd.DataFrame({
            "Ngày": [next_date],
            "Giá ngày trước": [actual_closing_price],
            "Giá dự đoán ngày kế tiếp": [y_next_predict[0][0]]
        })
        print(comparison_df)
        
        # Vẽ biểu đồ
        plt.figure(figsize=(15, 5))
        plt.plot(df['Ngày'], df['Đóng cửa'], label='Giá thực tế', color='red')
        plt.plot(df['Ngày'][50:1500], y_train_predict, label='Dự đoán train', color='green')
        plt.plot(df['Ngày'][1500:], y_test_predict, label='Dự đoán test', color='blue')
        plt.scatter([next_date], [y_next_predict[0][0]], color='orange', label='Dự đoán ngày kế tiếp')
        plt.title(f' Dự đoán giá đóng cửa ngày kế tiếp - {company}')
        plt.xlabel('Thời gian')
        plt.ylabel('Giá đóng cửa')
        plt.legend()
        plt.tight_layout()
        plt.show()
        
    except Exception as e:
        print(f"Lỗi khi xử lý {company}: {e}")

