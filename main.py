# %% [markdown]
# # بخش اول : آموزش و تست مدل

# %%
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score 
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
import joblib

# %%
# 1. بارگذاری دیتاست
# فایل CSV دیتاست
data = pd.read_csv('weather_forecast_data.csv')
data.head()

# %%
# 2. تبدیل ستون Rain به مقدار عددی
label_encoder = LabelEncoder()
data['Rain'] = label_encoder.fit_transform(data['Rain'])
data.head()


# %%
# 2.1 ذخیره کردن label encoder
joblib.dump(label_encoder, 'label_encoder.pkl')

# %%
# 2.2 نمودار توزیع (Histogram) برای هر ویژگی
data.drop('Rain', axis=1).hist(bins=10, figsize=(10, 8))
plt.suptitle('Distribution of Features')
plt.show()

# 2.3 نمودار پراکندگی (Scatter Plot) بین ویژگی‌ها
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Temperature', y='Humidity', 
                data=data, hue='Rain', style='Rain')
plt.title('Scatter Plot: Temperature vs Humidity')
plt.show()

# 2.4 نمودار پراکندگی (Scatter Plot) بین ویژگی‌ها
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Cloud_Cover', y='Humidity', 
                data=data, hue='Rain', style='Rain')
plt.title('Scatter Plot: Cloud_Cover vs Humidity')
plt.show()


# %%
# 3.1 محاسبه ماتریس همبستگی
correlation_matrix = data.corr()

# 3.2 رسم Heatmap از ماتریس همبستگی
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix,
            annot=True,
            fmt='.2f',
            cmap='coolwarm',
            cbar=True,
            square=True)
plt.title('Feature Correlation Matrix')
plt.show()

# %%
# 4.1 جداسازی ویژگی‌ها و برچسب هدف
X = data[['Temperature', 'Humidity', 'Wind_Speed', 'Cloud_Cover', 'Pressure']]
y = data['Rain']

# %%
# 4.2 تقسیم داده‌ها به مجموعه‌های آموزش و تست
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.2,
                                                    random_state=42)

# %%
# 4.3 استانداردسازی داده‌ها
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 4.4 ذخیره کردن scaler
joblib.dump(scaler, 'scaler.pkl')

# %%
# 5. تعریف مدل‌ها
models = {
    'Logistic Regression': LogisticRegression(),
    'Decision Tree': DecisionTreeClassifier(),
    'Naive Bayes': GaussianNB(),
    'MLP': MLPClassifier(hidden_layer_sizes=(50, 50), max_iter=500, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
    'XGBoost': XGBClassifier(eval_metric='logloss', random_state=42),
    'KNN': KNeighborsClassifier(n_neighbors=5),
    'SVM': SVC(kernel='linear', random_state=42),
}

# %%
# 6. آموزش و ارزیابی مدل‌ها
results = {}
for model_name, model in models.items():
    print(f"Model: {model_name}")
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)

    # گزارش مدل
    print(classification_report(y_test, y_pred, zero_division=1))

    # ماتریس درهم‌ریختگی
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d',
                cmap='Blues',
                xticklabels=label_encoder.classes_,
                yticklabels=label_encoder.classes_)
    plt.title(f'Confusion Matrix for {model_name}')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

    # دقت مدل
    accuracy = accuracy_score(y_test, y_pred)
    results[model_name] = accuracy

    # ذخیره مدل
    joblib.dump(model, f'{model_name}_model.pkl')

# %%
# 7. نمایش نتایج
print("Accuracy Results:")
for model_name, accuracy in results.items():
    print(f"{model_name}: {accuracy:.2f}")

# %% [markdown]
# ### پایان آموزش و ذخیره مدل

# %% [markdown]
# # ------------------------

# %% [markdown]
# # بخش دوم: مراحل استفاده از مدل ذخیره شده

# %%
# 1. بارگذاری مدل‌ها (اگر نیاز دارید از مدل‌های ذخیره‌شده استفاده کنید)
models_loaded = {}
for model_name in models.keys():
    models_loaded[model_name] = joblib.load(f'{model_name}_model.pkl')

# %%
# 2. استفاده از مدل‌های بارگذاری‌شده
# مثلا برای پیش‌بینی با مدل 'Random Forest'
y_pred_loaded = models_loaded['Random Forest'].predict(X_test_scaled)
# گزارش مدل لود شده
print(classification_report(y_test, y_pred_loaded, zero_division=1))

# %%
# 3. پیش بینی با داده های جدید
loaded_model = joblib.load('Decision Tree_model.pkl')
scaler = joblib.load('scaler.pkl')
label_encoder = joblib.load('label_encoder.pkl')

# %%
# 3.1 ایجاد داده تست
test_df = pd.DataFrame({
    'Temperature': [22.5, 12],
    'Humidity': [85.0, 60],
    'Wind_Speed': [5.2, 0],
    'Cloud_Cover': [65.0, 95],
    'Pressure': [1015.0, 900]
})
test_df.head()

# %%
# 3.2 استفاده از آنها برای پیش‌بینی روی داده‌های جدید
# درصورتیکه داده ها نیاز به scale داشته باشند
X_new_scaled = scaler.transform(test_df)
predict = loaded_model.predict(X_new_scaled)
predicted_labels = label_encoder.inverse_transform(predict)
print(predicted_labels)

# %%



