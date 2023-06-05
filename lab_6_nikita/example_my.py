import streamlit as st
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.impute import SimpleImputer


@st.cache_data
def load_data():
    '''
    Загрузка данных
    '''
    data = pd.read_csv('weather.csv')
    data.drop(['Evaporation','Sunshine','Cloud9am','Cloud3pm','Unnamed: 0'], axis=1, inplace=True)
    data.dropna(subset=['RainToday'], axis=0, inplace=True)
    imp1 = SimpleImputer(missing_values=np.nan, strategy='mean')
    imp2 = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
    data[['MinTemp','MaxTemp','Rainfall','WindGustSpeed','WindSpeed9am','WindSpeed3pm','Humidity9am','Humidity3pm','Pressure9am','Pressure3pm','Temp9am','Temp3pm']] = imp1.fit_transform(data[['MinTemp','MaxTemp','Rainfall','WindGustSpeed','WindSpeed9am','WindSpeed3pm','Humidity9am','Humidity3pm','Pressure9am','Pressure3pm','Temp9am','Temp3pm']])
    le = LabelEncoder()
    data['Location']= le.fit_transform(data['Location'])
    data['WindGustDir']= le.fit_transform(data['WindGustDir'])
    data['WindDir9am']= le.fit_transform(data['WindDir9am'])
    data['WindDir3pm']= le.fit_transform(data['WindDir3pm'])
    data['RainToday']= le.fit_transform(data['RainToday'])
    data['RainTomorrow']= le.fit_transform(data['RainTomorrow'])
    data[['WindGustDir','WindDir9am','WindDir3pm']] = imp2.fit_transform(data[['WindGustDir','WindDir9am','WindDir3pm']])
    return data


@st.cache_data
def preprocess_data(data_in):
    '''
    Масштабирование признаков, функция возвращает X и y для кросс-валидации
    '''
    data_out = data_in.copy()
    # Числовые колонки для масштабирования
    scale_cols = ['WindGustSpeed', 'Humidity9am', 'Humidity3pm', 'Pressure9am','Pressure3pm']
    new_cols = []
    sc1 = MinMaxScaler()
    sc1_data = sc1.fit_transform(data_out[scale_cols])
    for i in range(len(scale_cols)):
        col = scale_cols[i]
        new_col_name = col + '_scaled'
        new_cols.append(new_col_name)
        data_out[new_col_name] = sc1_data[:,i]
    X_train, X_test, y_train, y_test = train_test_split(data_out[new_cols], data_out['RainTomorrow'], train_size=0.8, random_state=1)
    return X_train, X_test, y_train, y_test



st.sidebar.header('Support Vector Classification')
data = load_data()
kernel = st.sidebar.radio('Ядро:', ['rbf', 'linear', 'sigmoid', 'poly'])
if kernel == 'poly':
    degree_slider = st.sidebar.slider('Степень полинома:', min_value=1, max_value=10, value=1, step=1)

if st.checkbox('Показать корреляционную матрицу'):
    fig1, ax = plt.subplots(figsize=(10,5))
    sns.heatmap(data.corr(), annot=True, fmt='.2f')
    st.pyplot(fig1)

#Количество записей
data_len = data.shape[0]
#Вычислим количество возможных ближайших соседей
# rows_in_one_fold = int(data_len / cv_slider)
# allowed_knn = int(rows_in_one_fold * (cv_slider-1))
# st.write('Количество строк в наборе данных - {}'.format(data_len))
# st.write('Максимальное допустимое количество ближайших соседей с учетом выбранного количества фолдов - {}'.format(allowed_knn))

# Подбор гиперпараметра
# n_range_list = list(range(1,allowed_knn,step_slider))
# n_range = np.array(n_range_list)
# st.write('Возможные значения соседей - {}'.format(n_range))
# tuned_parameters = [{'n_neighbors': n_range}]

X_train, X_test, y_train, y_test = preprocess_data(data)
if kernel == 'poly':
    clf_svc = SVC(kernel=kernel, degree=degree_slider)
else:
    clf_svc = SVC(kernel=kernel)
clf_svc.fit(X_train, y_train)

st.subheader('Оценка качества модели')

y_pred_test_svc = clf_svc.predict(X_test)
y_pred_train_svc = clf_svc.predict(X_train)
st.write('На тренировочной выборке: ', accuracy_score(y_train, y_pred_train_svc))
st.write('На тестовой выборке: ',accuracy_score(y_test, y_pred_test_svc))

st.subheader('Матрица ошибок')
fig2, ax = plt.subplots(figsize=(10,5))
cm = confusion_matrix(y_test, y_pred_test_svc, normalize='all', labels = clf_svc.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['target','no target'])
disp.plot(ax=ax, cmap=plt.cm.PiYG)
fig2.suptitle('Confusion Matrix')
st.pyplot(fig2)

def draw_roc_curve(y_true, y_score, ax, pos_label=1, average='micro'):
    fpr, tpr, thresholds = roc_curve(y_true, y_score,
                                     pos_label=pos_label)
    roc_auc_value = roc_auc_score(y_true, y_score, average=average)
    # plt.figure()
    lw = 2
    ax.plot(fpr, tpr, color='darkorange',
            lw=lw, label='ROC curve (area = %0.2f)' % roc_auc_value)
    ax.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    ax.set_xlim([0.0, 1.0])
    ax.set_xlim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Receiver operating characteristic')
    ax.legend(loc="lower right")

st.subheader('ROC-кривая')
fig3, ax = plt.subplots(figsize=(10,5))
draw_roc_curve(y_test.values, y_pred_test_svc, ax)
st.pyplot(fig3)
