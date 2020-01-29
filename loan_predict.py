import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler 

df = pd.read_csv("Envision_Predict Loans_Train.csv")
df_test = pd.read_csv("Envision_Predict Loans_Test.csv")

# Dropping rows with null values
df.dropna(inplace=True)

# Replacing categorical data to binary representation
df = df.replace(to_replace="Bad", value=0)
df = df.replace(to_replace="Good", value=1)

bank = pd.get_dummies(df['employment_status_clients'])
df.drop(['employment_status_clients'], axis=1, inplace=True)

bank_test = pd.get_dummies(df_test['employment_status_clients'])
df_test.drop(['employment_status_clients'], axis=1, inplace=True)

df = pd.concat([df, bank], axis=1)
df_test = pd.concat([df_test, bank_test], axis=1)

# Drop irrelevant columns for model creation
df = df.drop(['customer_id', 'system_loanid', 'loan_number', 'approved_time', 'creation_time', 'bank_account_type',
             'longitude_gps', 'latitude_gps', 'bank_name_clients'], axis=1)
df_test = df_test.drop(['customer_id', 'system_loanid', 'loan_number', 'approved_time', 'creation_time', 'bank_account_type',
             'longitude_gps', 'latitude_gps', 'bank_name_clients', 'good_bad_flag'], axis=1)

# Split the training data into test data and train data
X_train, X_test, y_train, y_test = train_test_split(df.drop('good_bad_flag', axis=1), 
                                                            df['good_bad_flag'])

# Standardize the train dataset
sc_x = StandardScaler() 
X_train = sc_x.fit_transform(X_train)  

# Model creation using train dataset
model = LogisticRegression()
model.fit(X_train, y_train)

print model.score(X_test, y_test)
print model.predict(df_test)
