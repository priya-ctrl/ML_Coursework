import pandas as pd
import numpy as np
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split
from surprise import accuracy
from surprise.model_selection import cross_validate

#sample dataset of user ratings on movies
data_dict ={
    "user_id" : [1, 1, 1, 2, 2, 2, 3, 3, 4, 4],
    "item_id" : [101, 102, 103, 101, 103, 104, 102, 104, 101, 103],
    "rating"  : [5, 3, 4, 4, 5, 2, 3, 4, 2, 5]
}

df = pd.DataFrame(data_dict)
print(df)

reader = Reader(rating_scale=(1,5)) #Ratings range from 1 to 5
data = Dataset.load_from_df([df['user_id', 'item_id', 'rating']], reader)

trainset, testset = train_test_split(data, test_size=0.2)

#Train the SVD Model
model = SVD()
model.fit(trainset)

#Make Predictions

predictions = model.test(testset)

#Evaluate the model performance using RMSE
rmse = accuracy.rmse(predictions)
print(f"Root Mean Square Error (RMSE) : {rmse}")

#Recommend Items for a User

user_id = 1
item_id = 104

predicted_rating = model.predict(user_id , item_id) . est
print(f"Predicted rating of user {user_id} for item {item_id} : {predicted_rating}")

