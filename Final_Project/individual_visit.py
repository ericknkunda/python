
#import the required libraries

import pandas as pd
import pymysql
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import TruncatedSVD
from pandas import to_numeric
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

#connecting to the database

databaseConnection =pymysql.connect(
    user="root", 
    password="", 
    host="localhost",
    database="final_year_project"
)
cursor =databaseConnection.cursor()
visitHistoryQuerry ="select *from fp_individual_visit_history"

try:
    cursor.execute(visitHistoryQuerry)

    #fetch all records

    historyDeatails =cursor.fetchall()
    #for user in historyDeatails:
        #print("Item Visited by " ,user[1], "is ", user[3])
except:
    print("Error: Connection Failured")
#creating a dataframe
#by calling a pandas function

pandaFrame =pd.DataFrame(historyDeatails)
pandaFrame.columns = ['Id', 'user_phone', 'user_age', 'item_visited', 'rating', 'visit_time']
rating_mapping = {
    "Very_Satisfied": 5,
    "Satisfied": 4,
    "Neutral": 3,
    "Unsatisfied": 2,
    "Very_Unsatisfied": 1
}
pandaFrame["rating"] = pandaFrame["rating"].map(rating_mapping)
#print(pandaFrame.columns)

#understanding the dataset

#head 5 rows
#print(pandaFrame.head())

#tail 5 rows
#print(pandaFrame.tail())

#summary of the data
#print(pandaFrame.info())

#description of numerical columns
#print(pandaFrame.describe())
#print(pandaFrame[4].value_counts())

#visualization how many many items each user have visited
x =pandaFrame['Id']
y =pandaFrame['user_age']
plt.scatter(x,y)
#plt.show()

#sclaling 
#starts here
encoder_phone =OneHotEncoder()
encoderUseAges =OneHotEncoder()
encoderItems =OneHotEncoder()
encodeRating =OneHotEncoder()
encoderVisitTimes =MinMaxScaler()

#fit the encoder

encoder_phone.fit(pandaFrame[['user_phone']])
encoded_user_phone =encoder_phone.transform(pandaFrame[['user_phone']])

df_encoded_user_phone = pd.DataFrame(encoded_user_phone.toarray())
df_encoded_user_phone.columns = encoder_phone.get_feature_names_out(['user_phone'])

# Concatenate the encoded dataframe with the original dataframe
pandaFrame = pd.concat([pandaFrame, df_encoded_user_phone], axis=1)

#renaming original columns
# pandaFrame.rename(columns={pandaFrame[[0]]: 'Id',\
#     pandaFrame[[1]]: 'user_phone', pandaFrame[[2]]:'user_age',\
#         pandaFrame[[3]]:'item_visited', pandaFrame[[4]]:'rating', \
#             pandaFrame[[5]]:'visitit_times'}, inplace=True)

#encoding user_ages

encoderUseAges.fit(pandaFrame[['user_age']])
encoded_user_age =encoderUseAges.transform(pandaFrame[['user_age']])

df_encoded_user_age = pd.DataFrame(encoded_user_age.toarray())
df_encoded_user_age.columns = encoderUseAges.get_feature_names_out(['user_age'])

# Concatenate the encoded dataframe with the original dataframe
pandaFrame = pd.concat([pandaFrame, df_encoded_user_age], axis=1)

#scaling item_visited
encoderItems.fit(pandaFrame[['item_visited']])
encoded_user_item =encoderItems.transform(pandaFrame[['item_visited']])

df_encoded_user_item = pd.DataFrame(encoded_user_item.toarray())
df_encoded_user_item.columns = encoderItems.get_feature_names_out(['item_visited'])

# Concatenate the encoded dataframe with the original dataframe
pandaFrame = pd.concat([pandaFrame, df_encoded_user_item], axis=1)

#scalling retings
encodeRating.fit(pandaFrame[['rating']])
encoded_user_rating =encodeRating.transform(pandaFrame[['rating']])

df_encoded_user_rating = pd.DataFrame(encoded_user_rating.toarray())
df_encoded_user_rating.columns = encodeRating.get_feature_names_out(['rating'])
#print(df_encoded_user_rating.columns)

# Concatenate the encoded dataframe with the original dataframe
pandaFrame = pd.concat([pandaFrame, df_encoded_user_rating], axis=1)

#scalling visittimes
# Fit the scaler to the visit_times column
encoderVisitTimes.fit(pandaFrame[['visit_time']])

# Transform the visit_times column and store the result in a new dataframe
scaled_visit_times = encoderVisitTimes.transform(pandaFrame[['visit_time']])

# You can then convert the scaled data back into a dataframe if you want
df_scaled_visit_times = pd.DataFrame(scaled_visit_times, columns=["visit_times"])

# Finally, you can concatenate the scaled dataframe with the original dataframe
pandaFrame = pd.concat([pandaFrame, df_scaled_visit_times], axis=1)

#print(pandaFrame.describe())

# Dropping the original columns
pandaFrame = pandaFrame.drop(['user_phone','user_age','item_visited'], axis=1)

# Specify the features and target
X = pandaFrame.drop(['rating'],axis=1)
# y = pandaFrame[['rating']]
y = pandaFrame['rating']
# pandaFrame=pandaFrame.dropna()

# Split the data into a training set and a test set
X_train, X_test, y_train, y_test = train_test_split(pandaFrame, y, test_size=0.2, random_state=42)
# X_train['user_age'] = to_numeric(X_train['user_age'], errors='coerce')
X_train = X_train.dropna()
X_test =X_test.dropna()
y_train = y_train.dropna()
y_test =y_test.dropna()



# Instantiate the SVD model
svd = TruncatedSVD(n_components=1)

# Fit the model to the training data
# svd.fit(X_train)
reg = LinearRegression().fit(X_train, y_train)

y_train_pred = reg.predict(X_train)
y_test_pred = reg.predict(X_test)

# Evaluation Metrics
print("MSE train: ", mean_squared_error(y_train, y_train_pred))
print("MSE test: ", mean_squared_error(y_test, y_test_pred))

# Print the training score
# print("Training score: ", reg.score(X_test, y_test))
predictions = reg.predict(X_test)

# Use the model to make predictions on the test data
# X_transformed = svd.transform(X_test)
# predictions = X_transformed.predict(X_test)

# print(predictions)
# print(X_train.value_counts()) 
# print("X-test set set\n")
# print(X_test)
# print("y_training set\n")  
# print(y_train)
# print("y_test set\n")
# print(y_test)
#closing connections
cursor.close()
databaseConnection.close()

