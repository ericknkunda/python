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
from numpy.linalg import svd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import f1_score

#connecting to the database

databaseConnection =pymysql.connect(
    user="root", 
    password="", 
    host="localhost",
    database="final_project_new"
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
pandaFrame.to_csv("user_item_interactions.csv", index=False)

pandaFrame.columns = ['visit_id', 'user_id','phone_number', 'ages_range', 'item_id','item_visited', 'rating_given', 'visiting_times\
    ', 'gender', 'residence','parent_category', 'sub_category', 'item_location','visiting_status']

# ratings numeric
rating_mapping = {
    "Very_Satisfied": 5,
    "Satisfied": 4,
    "Neutral": 3,
    "Unsatisfied": 2,
    "Very_Unsatisfied": 1
}

#visit status on numeric 
visitinf_status_mapping = {
    "Open for Visit": 2,
    "Closed for Visit": 1
}

#ages on numeric
ages_mapping = {
    "10-15": 13,
    "16-35": 26,
    "36-":36
}

#numeric gender
gender_mapping = {
    "Male": 2,
    "Female": 1
}

#residence mapping
residence_mapping = {
    "Urban": 2,
    "Rural": 1
}
pandaFrame["visiting_status"] = pandaFrame["visiting_status"].map(visitinf_status_mapping)
# pandaFrame["ages_range"] = pandaFrame["ages_range"].map(ages_mapping)
pandaFrame["rating_given"] = pandaFrame["rating_given"].map(rating_mapping)
pandaFrame["gender"] = pandaFrame["gender"].map(gender_mapping)
pandaFrame["residence"] = pandaFrame["residence"].map(residence_mapping)


encoder_phone =OneHotEncoder()
encoder_items = OneHotEncoder()
encoder_location = OneHotEncoder()
encoder_ages = OneHotEncoder()

encoder_phone.fit(pandaFrame[['phone_number']])
encoded_user_phone =encoder_phone.transform(pandaFrame[['phone_number']])
df_encoded_user_phone = pd.DataFrame(encoded_user_phone.toarray())
df_encoded_user_phone.columns = encoder_phone.get_feature_names_out(['phone_number'])
# pandaFrame = pd.concat([pandaFrame, df_encoded_user_phone], axis=1)

encoder_ages.fit(pandaFrame[['ages_range']])
encoded_user_ages =encoder_ages.transform(pandaFrame[['ages_range']])
df_encoded_user_ages = pd.DataFrame(encoded_user_ages.toarray())
df_encoded_user_ages.columns = encoder_ages.get_feature_names_out(['ages_range'])
pandaFrame = pd.concat([pandaFrame, df_encoded_user_ages], axis=1)


#scalling item visited
encoder_items.fit(pandaFrame[['item_visited']])
encoder_item_visited =encoder_items.transform(pandaFrame[['item_visited']])

df_item_visited =pd.DataFrame(encoder_item_visited.toarray())
df_item_visited.columns = encoder_items.get_feature_names_out(['item_visited'])

pandaFrame = pd.concat([pandaFrame, df_item_visited], axis=1)
 
encoder_location.fit(pandaFrame[['item_location']])
encoder_item_location =encoder_location.transform(pandaFrame[['item_location']])
df_item_location =pd.DataFrame(encoder_item_location.toarray())
df_item_location.columns =encoder_location.get_feature_names_out(['item_location'])

pandaFrame = pd.concat([pandaFrame, df_item_location], axis=1)

pandaFrame =pandaFrame.drop(['phone_number','ages_range','item_visited'],axis =1)
newDataFrame =pandaFrame[['user_id','item_id', 'rating_given']]
print(newDataFrame.head(5000))
pandaFrameMatrix =pandaFrame.pivot(index=['visit_id','user_id'],columns='item_id', values='rating_given').fillna(0)
# pandaFrameMatrix.dropna()
# X = pandaFrameMatrix.drop(['rating_given'],axis=1)
y = pandaFrame['rating_given'].fillna(0)

svd = TruncatedSVD(n_components=100, random_state=42)
svd.fit(pandaFrameMatrix)

X_transformed = svd.transform(pandaFrameMatrix)
X_train, X_test, y_train, y_test = train_test_split(X_transformed, y, test_size=0.2, random_state=42)

# X_train =X_train.fill(0)
y_train =y_train.fillna(0)
# X_test =X_test.fillna(0)
y_test =y_test.fillna(0)
reg = LinearRegression()
reg.fit(X_train, y_train)
print("Training score using regression: ", reg.score(X_train, y_train))

y_train_pred = reg.predict(X_train)
y_test_pred = reg.predict(X_test)
    
# Evaluation Metrics
print("MSE train: ", mean_squared_error(y_train, y_train_pred))
print("MSE test: ", mean_squared_error(y_test, y_test_pred))

# ...

# Split the data into training and test sets
# X_train, X_test, y_train, y_test = train_test_split(X_train, y, test_size=0.2)

# Initialize and fit the random forest regressor
rf = RandomForestRegressor(n_estimators=100)
rf.fit(X_train, y_train)

# Print the training score
print("Training score using Random Forest: ", rf.score(X_train, y_train))

# Print the test score
print("Test score using Random Forest: ", rf.score(X_test, y_test))

from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression

# Create an instance of your model
model = LinearRegression()

# Use the cross_val_score function to perform k-fold cross-validation
scores = cross_val_score(reg, X_transformed, y, cv=5)

# Print the mean and standard deviation of the scores
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))


# print(pandaFrame.columns)

# #understanding the dataset

# #head 5 rows
# # print(X_train.columns)

# #tail 5 rows
# #print(pandaFrame.tail())

# #summary of the data
# # print(pandaFrame.info())

# #description of numerical columns
# # print(pandaFrame.describe())
# print(pandaFrame["rating_given"].value_counts())
# pandaFrame.dropna()
# print(pandaFrameMatrix)

# matrix = pandaFrameMatrix.values
# u, s, vh = svd(matrix, full_matrices=False)
# def cosine_similarity(v,u):
#     return (v @ u)/ (np.linalg.norm(v) * np.linalg.norm(u))
 
# highest_similarity = -np.inf
# highest_sim_col = -1
# for col in range(1,vh.shape[1]):
#     similarity = cosine_similarity(vh[:,0], vh[:,col])
#     if similarity >= highest_similarity:
#         highest_similarity = similarity
#         highest_sim_col = col
 
# print("Column %d (item id %s) is most similar to column 0 (item id %s)" %
#         (highest_sim_col, pandaFrameMatrix.columns[col], pandaFrameMatrix.columns[10]))
# Specify the features and target
# X = pandaFrame.drop(['rating_given'],axis=1)

# Split the data into a training set and a test set


# # Instantiate the SVD model
# print ("movie_features.shape = {0}".format(X_transformed.shape))
# train_predictions = svd.predict(X_test)

# calculate the MSE of the model on the training set
# mse_train = mean_squared_error(X_test, train_predictions)
# print("MSE train: ", mse_train)

# Calculate the cosine similarity between latent feature vectors
# item_similarities = cosine_similarity(X_transformed)

# item_id = 5
# top_n = 10
# similar_items = np.argsort(-item_similarities[item_id])[1:top_n+1]
# print(similar_items)

# print(svd.singular_values_)
# print(svd.explained_variance_ratio_)

# interactionMatrix =X_train.pivot(index='user_id', columns ='item_id', values ='rating_given').fillna(0)
# print(interactionMatrix)