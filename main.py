# https://www.codeit.kr/learn/courses/machine-learning/3325
# 데이터 전처리
# 09. Normalization 직접 해보기

# 필요한 도구 임포트
from sklearn import preprocessing
import pandas as pd

PATIENT_FILE_PATH = './datasets/liver_patient_data.csv'
pd.set_option('display.float_format', lambda x: '%.5f' % x)

# 데이터 파일을 pandas dataframe으로 가지고 온다
liver_patients_df = pd.read_csv(PATIENT_FILE_PATH)

# Normalization할 열 이름들
features_to_normalize = ['Total_Bilirubin','Direct_Bilirubin', 'Alkaline_Phosphotase', 'Alamine_Aminotransferase']

new_df = liver_patients_df[features_to_normalize]

scaler = preprocessing.MinMaxScaler()
normalized = scaler.fit_transform(new_df)

normalized_df = pd.DataFrame(normalized, columns=features_to_normalize)

# 체점용 코드
print(normalized_df.describe())