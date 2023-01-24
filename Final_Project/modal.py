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
import csv
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import f1_score
from sklearn.metrics.pairwise import cosine_similarity


#connecting to the database

databaseConnection =pymysql.connect(
    user="root", 
    password="", 
    host="localhost",
    database="final_project_new"
)
cursor =databaseConnection.cursor()
cursor.execute("DESCRIBE fp_individual_visit_history")
columns = [col[0] for col in cursor.fetchall()]

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
with open('final_project.csv', 'w', newline='') as f:
    csv_writer = csv.writer(f)
    csv_writer.writerow(columns)
    csv_writer.writerows(historyDeatails)

ratings =pd.read_csv("final_project.csv", encoding = 'ISO-8859-1')
# ratings =ratings.drop(['user_id'])
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
ratings["item_visiting_status"] = ratings["item_visiting_status"].map(visitinf_status_mapping)
# pandaFrame["ages_range"] = pandaFrame["ages_range"].map(ages_mapping)
ratings["rating"] = ratings["rating"].map(rating_mapping)
ratings["user_gender"] = ratings["user_gender"].map(gender_mapping)
ratings["user_residence"] = ratings["user_residence"].map(residence_mapping)
ratings['user_phone'] = pd.to_numeric(ratings['user_phone'])

encoder_items = OneHotEncoder()
encoder_location = OneHotEncoder()
encoder_ages = OneHotEncoder()

encoder_ages.fit(ratings[['user_age']])
encoded_user_ages =encoder_ages.transform(ratings[['user_age']])
df_encoded_user_ages = pd.DataFrame(encoded_user_ages.toarray())
df_encoded_user_ages.columns = encoder_ages.get_feature_names_out(['user_age'])
ratings = pd.concat([ratings, df_encoded_user_ages], axis=1)


#scalling item visited
encoder_items.fit(ratings[['item_visited']])
encoder_item_visited =encoder_items.transform(ratings[['item_visited']])

df_item_visited =pd.DataFrame(encoder_item_visited.toarray())
df_item_visited.columns = encoder_items.get_feature_names_out(['item_visited'])

# ratings = pd.concat([ratings, df_item_visited], axis=1)
 
encoder_location.fit(ratings[['item_location']])
encoder_item_location =encoder_location.transform(ratings[['item_location']])
df_item_location =pd.DataFrame(encoder_item_location.toarray())
df_item_location.columns =encoder_location.get_feature_names_out(['item_location'])

ratings = pd.concat([ratings, df_item_location], axis=1)
nonrated =ratings

ratings =ratings.pivot(index=['id'],columns='item_visited', values='rating')
ratings =ratings.fillna(0)
# print(ratings.head())
# print(ratings.columns)

def standardiserow(row):
    new_row =(row-row.mean())/(row.max()-row.min())
    return new_row

ratings_std =ratings.apply(standardiserow)
item_similarity =cosine_similarity(ratings_std.T)

ratings_df =pd.DataFrame(item_similarity, index=ratings.columns, columns=ratings.columns)
# ratings_df =ratings_df.fillna(0)
# print(ratings_df.head())

def get_similar_items(itemId, ratings_given):
    similar_scores =ratings_df[itemId]*(ratings_given -2.5)
    similar_scores =similar_scores.sort_values(ascending=False)
    return similar_scores

item_lover =[('Amahamba',4),('Abiru',3),('Museum',5)]
similar_items =pd.DataFrame()
for itemId, ratings_given in item_lover:
    similar_items =similar_items.append(get_similar_items(itemId,ratings_given),ignore_index=True)
    

rowwise =similar_items.sum().sort_values(ascending=False)
print("______________________________________")
print(rowwise) 


y = nonrated['rating'].fillna(0)

svd = TruncatedSVD(n_components=100, random_state=42)
svd.fit(ratings)

X_transformed = svd.transform(ratings)
print(X_transformed.shape)
X_train, X_test, y_train, y_test = train_test_split(X_transformed, y, test_size=0.2, random_state=42)

# X_train =X_train.fill(0)
y_train =y_train.fillna(0)
# X_test =X_test.fillna(0)
y_test = y_test.fillna(0)
reg = LinearRegression()
reg.fit(X_train, y_train)

# print("Training score using regression: ", reg.score(X_train, y_train))

y_train_pred = reg.predict(X_train)
y_test_pred = reg.predict(X_test)
# print('predictions on X_train',y_train_pred)
# print('predictions on X_test',y_test_pred)

    
# Evaluation Metrics
# print("MSE train: ", mean_squared_error(y_train, y_train_pred))
# print("MSE test: ", mean_squared_error(y_test, y_test_pred))


rf = RandomForestRegressor(n_estimators=90)
rf.fit(X_test, y_test)

# Print the training score
print("Training score using Random Forest: ", rf.score(X_train, y_train))
print("Test score using Random Forest: ", rf.score(X_test, y_test))
# print("Random forest predictions ",rf.predict([10,'Ibere Rya Bigogwe']) )
# print("Random forest predictions ",rf.predict(X_train) )
# print("Random forest predictions ",rf.predict(X_test) )

from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression

# Use the cross_val_score function to perform k-fold cross-validation
scoresreg = cross_val_score(reg, X_transformed, y, cv=5)
scoresrand =cross_val_score(rf, X_transformed, y, cv=5)
# Print the mean and standard deviation of the scores
# print("Accuracy of linear regression: %0.2f (+/- %0.2f)" % (scoresreg.mean(), scoresreg.std() * 2))
print("Accuracy of random forest: %0.2f (+/- %0.2f)" % (scoresrand.mean(), scoresrand.std() * 2))
rf.predict([[5.]*100])

# import pickle

# with open('model.pkl','wb') as f:
#   pickle.dump(reg,f)
# print("------------------------------------------")
# print(get_similar_items(8,1))


cursor.close()
databaseConnection.close()