
# Đồ án tốt nghiệp Data Science
## Topic: Dự đoán giá, xác định bất thường giá cho xe máy

## Giới thiệu chung
Đây là bài đồ án tốt nghiệp với topic Dự đoán giá, xác định bất thường giá cho xe máy cũ của nhóm học viên **Nguyễn Ngọc Giao** - **Nguyễn Thị Tuyển** thực hiện.
Bao gồm các bài toán sau:
- Bài toán 1: Dự đoán giá xe theo theo các thông tin được cung cấp trên trang chợ tốt để gợi ý cho người bán giá hợp lý
- Bài toán 2: Phát hiện bất thường về giá để cảnh báo bất thường.
- Tạo GUI để nhập thông tin xe và nhận giá dự báo, cảnh báo khi giá bất thường

## Cài đặt
Để chạy được file cần cài đặt các thư viện:
1. Cài đặt python 3.10
2. Thư viện: pandas, numpy, scikit-learn, matplotlib, seaborn, openpyxl, streamlit
3. Thư viện Pyspark

## Hướng dẫn sử dụng
1. Các dữ liệu của project này được lưu tại folder final  
2. Chạy project_01_final.ipynb cho bài toán 1
3. Chạy project_02_final.ipynb cho bài toán 2
4. ### 3.1 Chạy GUI
```bash
cd final/GUI
streamlit run app.py
```
Nhập thông tin xe → nhấn Predict → nhận giá dự đoán.
5. Webapp: https://dubaobatthuongv1-bnigjbuhepgjzgndxc63nz.streamlit.app/

## Cấu trúc file
final/
│
├── source_code/
│ ├── du_bao_bat_thuong.py # bài toán dự báo bất thường
│ ├── du_bao_gia.py # bài toán dự báo giá
│ ├── project_01_final.ipynb # notebook bài toán 1
│ └── project_02_final.ipynb # notebook bài toán 2
│
├── GUI/
│ ├── firstGUI.py
│ └── files/ # file hình ảnh
│ ├── setup.sh
│ ├── Procfile
│ ├── requirement.txt
│
├── slides/
│ └── DoAn_Project1.pptx
│
└── eda_report.html # file báo cáo phân tích dữ liệu theo pandas profiling
└── README.md


## Các bước chính
### Bài toán 1:
1. Làm sạch dữ liệu và xử lý missing value, outlier
2. Chuyển giá trị Giá sang ln(Giá)
3. Loại bỏ outlier và biến đổi dữ liệu: string và category sang dạng số
4. Trực quan hóa bằng matplotlib và seaborn và phân tích mối quan hệ giữa Năm sản xuất, Số km đã đi, Thương hiệu, Dòng xe, Loại xe với Giá
5. Chuẩn hóa dữ liệu bằng StandardScaler
6. Sử dụng các feature 'Năm đăng ký', 'Số Km đã đi', 'Thương hiệu_ohe','Dòng xe_ohe','Loại xe_ohe', 'Dung tích xe_ohe','Xuất xứ_ohe' để dự đoán giá
7. Dự đoán giá trên môi trường Pyspark bằng các model: Linear Regression, Decision Tree, RandomForest, GBT Regressor
8. Dự đoán giá trên môi trường scikit-learn bằng các model: Linear Regression, Decision Tree, RandomForest, SVR, XGBoost
### Bài toán 2:
Xây dựng mô hình, bộ quy luật bất thường bằng các cách sau
1. Isolation Forest theo các feature 'Giá', 'Khoảng giá min', 'Khoảng giá max', 'Số Km đã đi'
2. Vượt ngưỡng threshold so với giá dự báo từ mô hình (XGBoost)
3. Vượt ngưỡng Giá_min/Giá_max: Sử dụng vòng lặp duyệt qua cột Giá. Nếu giá trị nào nhỏ hơn min_value (min của Khoảng giá min) hoặc lớn hơn max_value (max của Khoảng giá max) thì sẽ coi là vượt ngưỡng và là outliers.
4. Theo Q1 và Q3: Những điểm số liệu nhỏ Q1−1.5×IQR hoặc lớn hơn Q3+1.5×IQR được xem là outliers.
5. Theo Kmeans: phân bộ dữ liệu thành 4 cụm và phát hiện các giá trị có khoảng cách đến trung tâm các cụm >95%


