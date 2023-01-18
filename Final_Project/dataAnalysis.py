import pymysql
import csv


databaseConnection =pymysql.connect(
    user="root", 
    password="", 
    host="localhost",
    database="final_project_new"
)
cursor =databaseConnection.cursor()
cursor.execute("DESCRIBE fp_cultural_components_classification")
columns = [col[0] for col in cursor.fetchall()]

visitHistoryQuerry ="select *from fp_cultural_components_classification"

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
with open('clasification.csv', 'w', newline='') as f:
    csv_writer = csv.writer(f)
    csv_writer.writerow(columns)
    csv_writer.writerows(historyDeatails)