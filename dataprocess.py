import pandas as pd
import numpy as np

from gensim import corpora
from keras.preprocessing.text import Tokenizer
import gzip
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from scipy.sparse.linalg import svds
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

null = "Unknown"
house_data = pd.read_csv("../train.csv")
house_data = house_data.replace("Unknown", np.nan)
# house_data=house_data.drop(columns=['description','neighbourhood','latitude','longitude'])
print("Rows : {row:,}".format(row=house_data.shape[0]))
print("Columns: {0}".format(house_data.shape[1]))

print(house_data.head())
# print("Number of unique users: {user:,}".format(user=house_data["user_id"].nunique()))
# print("Number of unique items: {item:,}".format(item=house_data["item_id"].nunique()))
print(pd.DataFrame({"is_null_count": house_data.isna().sum()}))

house_data.columns = house_data.columns.str.replace(" ", "_")
house_data.dropna(subset=["target"], inplace=True)  # rating


# def height_to_inches_convert(col):
#     '''
#     function takes in column with feet and inches height format (e.g. 5 ' 5) and returns value in inches
#     col: value with initial string format
#     '''
#     # extracting all numbers from the height column so split from special characters
#     extract = col.str.extractall("(\d+)").reset_index()
#     # 2 values come out: one for feet and one for inches
#     # isolate both using boolean filter: match column indicates first and second value extracted where ft is index 0
#     ft = extract["match"] == 0
#     inches = extract["match"] == 1
#     # using boolean filter to get ft, inches and then convert both to numeric
#     # ft value multiplied by 12 to convert to inches
#     ft_conversion = extract[ft].drop(["level_0", "match"], axis=1).reset_index(drop=True).apply(pd.to_numeric) * 12
#     inch_conversion = extract[inches].drop(["level_0", "match"], axis=1).reset_index(drop=True).apply(pd.to_numeric)
#     # add both numbers for total height in inches
#     return ft_conversion[0] + inch_conversion[0]


# house_data["height"] = height_to_inches_convert(col=house_data["height"])
# house_data["weight"] = house_data["weight"].str.extract("(\d+)", expand=True)


# num_col_convert = ["age", "rating", "weight"]
# house_data[num_col_convert] = house_data[num_col_convert].apply(pd.to_numeric, errors="coerce")
house_data["review_rating"].fillna(0, inplace=True)
house_data["review_scores_A"].fillna(0, inplace=True)
house_data["review_scores_B"].fillna(0, inplace=True)
house_data["review_scores_C"].fillna(0, inplace=True)
house_data["review_scores_D"].fillna(0, inplace=True)

house_data["bedrooms"].fillna(1, inplace=True)
#fill nan in bathrooms with the most type
house_data["bathrooms"].fillna(house_data["bathrooms"].mode().iloc[0], inplace=True)

house_data["description"].fillna("none", inplace=True)
house_data["description"]=house_data["description"].str.slice(0,512)
data_class={'f':0,'t':1}
house_data['instant_bookable']=house_data['instant_bookable'].map(data_class)
# data_class={'Entire home/apt':0,'Private room':1,'Shared room':2,'Hotel room': 3}
# house_data['type']=house_data['type'].map(data_class)

print(pd.DataFrame({"is_null_count": house_data.isna().sum()}))

house_data.drop_duplicates(inplace=True);

print("Rows : {row:,}".format(row=house_data.shape[0]))
print("Columns: {0}".format(house_data.shape[1]))

house_data_df=house_data.drop(columns=["description", "latitude", "longitude"])
house_data_df=house_data_df.drop(columns=["neighbourhood", "type", "bathrooms", "amenities", "instant_bookable"])

# sns.set_style("whitegrid")
#
# sns.pairplot(house_data, vars=["instant_bookable", "type", "bedrooms","review_rating"], hue="target")
# plt.show()
print(house_data.info())

label = house_data["target"]
label.to_csv("train_label.csv", index=False, header=None)
house_data.to_csv("new_train.csv", index=False)

# tok = Tokenizer()
# tok.fit_on_texts(house_data['neighbourhood'])
# house_data['neighbourhood']=tok.texts_to_matrix(house_data['neighbourhood'], mode='freq')
# house_data['neighbourhood']=(house_data['neighbourhood'])


# house_data["amenities"] = house_data.amenities.str.replace("[^0-9a-zA-Z]+", " ")
# mediean
# house_data["height"].fillna(house_data["height"].median(), inplace=True)
# house_data["age"].fillna(house_data["age"].median(), inplace=True)
#
# age_logic = (house_data["age"] > 80) | (house_data["age"] == 0)
# house_data["age"] = np.where(age_logic == True, house_data["age"].median(), house_data["age"])
#
# weight_impute_vals = dict(house_data.groupby("height")["weight"].median())
# house_data["weight"] = house_data["weight"].fillna(house_data["height"].map(weight_impute_vals))
#
# body_impute_vals = dict(house_data.groupby("size")["body_type"].last())
# house_data["body_type"] = house_data["body_type"].fillna(house_data["size"].map(body_impute_vals))
#
# bust_impute_vals = dict(house_data.groupby("size")["bust_size"].last())
# house_data["bust_size"] = house_data["bust_size"].fillna(house_data["size"].map(bust_impute_vals))
#
# house_data["rented_for"].fillna(house_data["rented_for"].value_counts().index[0], inplace=True)

# house_data["description"] = house_data.description.str.replace("[^0-9a-zA-Z]+", " ")
#
#
# house_data["review_text"] = house_data.review_text.str.replace("[^0-9a-zA-Z]+", " ")
# house_data["review_text"] = house_data.review_text.str.replace("\n", ".")
# house_data["review_summary"] = house_data.review_summary.str.replace("[^0-9a-zA-Z]+", " ")
#
# house_data["rating"] = house_data["rating"] / 2
#
# house_data = house_data.drop(["review_date", "review_summary"], axis=1)
# house_data = house_data.dropna()
#
# print(house_data.info())
#
# if np.sum(house_data.isnull().sum()) == 0:
#     print("Data now has no remaining NaN entries")
# else:
#     print("Data still has NaN entries present")
#
# item_rating_count = pd.DataFrame(house_data.groupby("item_id")["fit"].count())
# item_rating_count = item_rating_count.loc[(item_rating_count["fit"] > 0)]
# item_rating_count.reset_index(inplace=True)
# fashion_reduced = house_data[house_data["item_id"].isin(item_rating_count["item_id"])]

# including users with more than 2 purchases
# user_rating_count = pd.DataFrame(fashion_reduced.groupby("user_id")["fit"].count())
# user_rating_count = user_rating_count.loc[(user_rating_count["fit"] > 0)]
# user_rating_count.reset_index(inplace=True)
# user_rating_count.drop("fit", axis=1, inplace=True)
#
# fashion_reduced = fashion_reduced[fashion_reduced["user_id"].isin(user_rating_count["user_id"])]
# fashion_reduced.reset_index(inplace=True, drop=True)
#
# print("Final number of transcations in reduced set: {val:,}".format(val=len(fashion_reduced)))
# print("Items with more than 1 ratings: {val:,}".format(val=fashion_reduced.item_id.nunique()))
# print("Users with more than 1 ratings: {val:,}".format(val=fashion_reduced.user_id.nunique()))

# sns.set_style("whitegrid")
#
# pairs_plot_df = house_data.drop(["item_id", "user_id", "review_summary", "review_text"], axis=1)

# sns.pairplot(pairs_plot_df, hue="fit", palette="Set2")
# plt.show()
# print(pd.DataFrame(house_data.groupby("age")["fit"].value_counts(normalize=True)))
# print(pd.DataFrame(house_data.groupby("height")["fit"].value_counts(normalize=True)))
# print(pd.DataFrame(house_data.groupby("weight")["fit"].value_counts(normalize=True)))
# print(pd.DataFrame(house_data.groupby("body_type")["fit"].value_counts(normalize=True)))
# print(pd.DataFrame(house_data.groupby("bust_size")["fit"].value_counts(normalize=True)))
# print(pd.DataFrame(house_data.groupby("rented_for")["fit"].value_counts(normalize=True)))
# print(pd.DataFrame(house_data.groupby("rating")["fit"].value_counts(normalize=True)))


# fitmap = {"fit": 0, "large": 1, "small": 2}
# fashion_reduced["class_fit"] = np.array([fitmap[x] for x in fashion_reduced["fit"]])

# category_vars = ["category"]
# #
# for variable in category_vars:
#     var_df = fashion_reduced.groupby(variable)["class_fit"].agg(["mean", "count"]).reset_index()
#     var_df.sort_values(["mean"], ascending = False, inplace = True)
#     sns.factorplot(y  = variable, x = "mean",
#                    orient = "h",
#                    hue = "count",
#                    palette = "Set2",
#                    size = 3, aspect = 3,
#                    data = var_df)
#     plt.title(variable + " " + "class fit factorplot")
# plt.show()


# label = fashion_reduced["fit"]
# label.to_csv("train_label.txt", index=False, header=None)
# fashion_reduced.to_csv("new_train.txt", index=False)
#
# review_text = fashion_reduced["review_text"]
# review_text.to_csv("review_text.txt", index=False, header=None)
#
# testcsv = pd.read_csv("../product_fit/test.txt")
#
# testcsv["review_text"] = testcsv.review_text.str.replace("[^0-9a-zA-Z]+", " ")
# testcsv["review_text"] = testcsv.review_text.str.replace("\n", ".")
# reviewtest_text = testcsv["review_text"]
# reviewtest_text.to_csv("test_text.txt", index=False, header=None)

# item_counts = house_data["item_id"].value_counts()
# item_counts.sort_values(ascending = False, inplace = True)
# item_x_axis = list(range(1, len(item_counts) + 1))
# median_item_buy = np.median(item_counts)
#
# user_counts = house_data["user_id"].value_counts()
# user_counts.sort_values(ascending = False, inplace = True)
# user_x_axis = list(range(1, len(user_counts) + 1))
# median_user_buy = np.median(user_counts)
#
# plt.figure(figsize=(14, 7))
# plt.subplot(1, 2, 1)
#
# plt.plot(item_x_axis, item_counts, color="purple", linewidth = 3)
# plt.xlabel("Item")
# plt.axhline(median_item_buy)
# plt.ylabel("Unique Item Transaction Count")
# plt.title("Low number of items have been purchased repeatedly \n median item buy count = {mdn} (blue line)".format(mdn = median_item_buy))
#
# plt.subplot(1, 2, 2)
#
# plt.plot(user_x_axis, user_counts, color="purple", linewidth = 3)
# plt.xlabel("User")
# plt.axhline(median_user_buy)
# plt.ylabel("Unique User Transaction Count")
# plt.title("Low number of users have been purchased repeatedly \n median item buy count = {mdn} (blue line)".format(mdn = median_user_buy))
# plt.show()

# including items with more than 2 ratings

# print(pd.DataFrame(fashion_reduced["class_fit"].value_counts(normalize=True)))

# fashion_train, fashion_test = train_test_split(fashion_reduced,
#                                                stratify=fashion_reduced["user_id"],
#                                                test_size=.2,
#                                                random_state=1017)
#
# print("Train size: {row:,}".format(row=fashion_train.shape[0]))
# print("Test size: {row:,}".format(row=fashion_test.shape[0]))
