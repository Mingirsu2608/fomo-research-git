# Model card

## Mục tiêu

Phân cụm hành vi nhà đầu tư để phục vụ nghiên cứu FOMO. Output không phải tư
vấn đầu tư và không dùng để ra quyết định tự động đối với khách hàng.

## Thuật toán

- K-Means
- MiniBatchKMeans
- Gaussian Mixture Model với diagonal covariance

K được chọn trong khoảng cấu hình bằng Silhouette score. Evaluation bổ sung
Calinski–Harabasz, Davies–Bouldin, cluster size và cluster profile.

## Preprocessing

Từ 32 feature theo investor-window, lọc coverage >= 80%, std > 0.02 và
IQR > 0.02. Loại feature tương quan Spearman tuyệt đối >= 0.70 theo thứ tự
ưu tiên cố định; các feature trong `lower_priority_features` được xét sau.
Các ngưỡng và thứ tự ưu tiên nằm trong `params.yaml`. Không cố định số feature
còn lại. Tương quan được tính theo các cặp quan sát có giá trị.

Chỉ giữ các cửa sổ đầy đủ giá trị hữu hạn trên bộ feature được chọn. Fit
StandardScaler rồi PCA giữ ít nhất 90% phương sai. Pipeline đã fit được lưu
trong `preprocessor.joblib`; evaluation tái sử dụng pipeline và cùng quy tắc
chọn cohort. Profile cụm dùng feature được chọn ở đơn vị gốc; notebook bổ sung
heatmap chuẩn hóa trước PCA để diễn giải.

`artifacts/models/feature_selection.csv` ghi coverage, std, IQR, feature tương
quan bị trùng và quyết định giữ/loại. `pca_variance.csv` ghi phương sai từng PC.

## Giới hạn

Metric được tính trên cohort training, chưa phải đánh giá ngoài mẫu. Complete-case
có thể thiên về nhà đầu tư hoạt động thường xuyên. Cần kiểm tra stability theo
thời gian trước khi sử dụng cho phân tích thực tế.
