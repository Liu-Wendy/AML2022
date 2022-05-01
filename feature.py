# get feature embedding
import pandas as pd
import numpy as np
import torch
from numpy import *
from torch.autograd import Variable

# FeatureDict = {}
#
# HeightTypeDict = {}
# WeightTypeDict = {}
# BodyTypeDict = {}
# BustsizeTypeDict = {}
# CategoryTypeDict = {}
# Item_idlabelsTypeDict = {}
# User_idlabelsTypeDict = {}

NeighbourhoodDict = {}
TypeDict = {}
AccommodateDict={}
BathroomDict = {}
# def ReadTextTrainmbedding():
#     f = open("bert_text/save_encoding_sen_review_train.txt", "r")
#     TEXTEmbedding = []
#     for x in f.readlines():
#         if x == "\n":
#             continue
#         temp = x.replace(" \n","").split(" ")
#         for i in range(0,len(temp)):
#             temp[i] = float(temp[i])
#         TEXTEmbedding.append(temp)
#     print(TEXTEmbedding.__len__())
#     return TEXTEmbedding
#
#
# def ReadTextTestembedding():
#     f = open("bert_text/save_encoding_sen_review_test.txt", "r")
#     TEXTEmbedding = []
#     for x in f.readlines():
#         if x == "\n":
#             continue
#         temp = x.replace(" \n","").split(" ")
#         for i in range(0,len(temp)):
#             temp[i] = float(temp[i])
#         TEXTEmbedding.append(temp)
#     print(TEXTEmbedding.__len__())
#     return TEXTEmbedding
#
#
# def Heightlabel(num):
#     if num <= 55:
#         return 0
#     elif num <= 60:
#         return 1
#     elif num <= 65:
#         return 2
#     elif num <= 70:
#         return 3
#     elif num <= 75:
#         return 4
#     elif num <= 80:
#         return 5
#     else:
#         return 6
#
# def Weightlabel(num):
#     if num <= 50:
#         return 0
#     elif num <= 100:
#         return 1
#     elif num <= 150:
#         return 2
#     elif num <= 200:
#         return 3
#     elif num <= 250:
#         return 4
#     elif num <= 300:
#         return 5
#     else:
#         return 6
#
# def Agelabel(num):
#     if num <= 10:
#         return 0
#     elif num <= 20:
#         return 1
#     elif num <= 25:
#         return 2
#     elif num <= 30:
#         return 3
#     elif num <= 35:
#         return 4
#     elif num <= 40:
#         return 5
#     elif num <= 50:
#         return 6
#     elif num <= 60:
#         return 7
#     elif num <= 70:
#         return 8
#     else:
#         return 9

def getRandom(j): # 0-j random
    return random.randint(0, j)

def processTrain():
    data = pd.read_csv("new_train.txt")
    # data.drop(["target"],axis=1)

    # neighbourhood 64
    neighbourhood = data["neighbourhood"].values
    neighbourhoodlabels = list(set(neighbourhood))
    for i in range(0, len(neighbourhoodlabels)):
        NeighbourhoodDict[neighbourhoodlabels[i]] = i
    neighbourhoodlabel = [NeighbourhoodDict[neighbourhood[i]] for i in range(0, len(neighbourhood))]
    neighbourhood_embedding = torch.nn.Embedding(neighbourhoodlabels.__len__(), 64)
    # TYPE 64
    type_types = data["type"].values
    type_types_labels = list(set(type_types))
    for i in range(0, len(type_types_labels)):
        TypeDict[type_types_labels[i]] = i
    typelabel = [TypeDict[type_types[i]] for i in range(0, len(type_types))]
    type_embedding = torch.nn.Embedding(type_types_labels.__len__(), 64)

    # BATHROOMS 64
    bathroom_types = data["bathrooms"].values
    bathroom_types_labels = list(set(bathroom_types))
    for i in range(0, len(bathroom_types_labels)):
        BathroomDict[bathroom_types_labels[i]] = i
    bathroomlabel = [BathroomDict[bathroom_types[i]] for i in range(0, len(bathroom_types))]
    bathroom_embedding = torch.nn.Embedding(bathroom_types_labels.__len__(), 64)

    RES_embedding = []
    RES_f = []
    for i in range(0, len(data["target"])):
        finalembedding = neighbourhood_embedding(Variable(torch.LongTensor([neighbourhoodlabel[i]])))[0]
        finalembedding = torch.cat((finalembedding, type_embedding(Variable(torch.LongTensor([typelabel[i]])))[0]),0)
        finalembedding = torch.cat((finalembedding, bathroom_embedding(Variable(torch.LongTensor([bathroomlabel[i]])))[0]), 0)
        print(finalembedding.shape)

        finalembedding = list(finalembedding.detach().numpy())
        RES_embedding.append(finalembedding)

    # output final embedding 192
    train_vector = open("vector_feature_train.txt", "w")
    # TEXTembedding = ReadTextTrainmbedding()
    for i in range(0, len(RES_embedding)):
        # RES_embedding[i] = TEXTembedding[i] + RES_embedding[i]
        for x in RES_embedding[i]:
            train_vector.write(str(x) + " ")
        train_vector.write("\n")
        if i % 1000 == 0:
            print(i)


    #
    #
    #
    # # bmi height type
    # height = data["height"].values
    # heightlabel = [Heightlabel(height[i]) for i in range(0,len(height))]
    # height_embedding = torch.nn.Embedding(7, 64)
    #
    # weight = data["weight"].values
    # weightlabel = [Weightlabel(weight[i]) for i in range(0,len(weight))]
    # weight_embedding = torch.nn.Embedding(7, 64)
    #
    # # age
    # age = data["age"].values
    # agelabel = [Agelabel(age[i]) for i in range(0,len(age))]
    # age_embedding = torch.nn.Embedding(10, 32)
    #
    # # body type
    # neighbourhood = data["body_type"].values
    # neighbourhoodlabels = list(set(neighbourhood))
    # for i in range(0, len(neighbourhoodlabels)):
    #     BodyTypeDict[neighbourhoodlabels[i]] = i
    # bodytypelabel = [BodyTypeDict[neighbourhood[i]] for i in range(0,len(neighbourhood))]
    # body_embedding = torch.nn.Embedding(neighbourhoodlabels.__len__(), 64)
    # #print(bodytypelabel)
    #
    # # bust size
    # bustsize = data["bust_size"].values
    # bustsizelabels = list(set(bustsize))
    # for i in range(0, len(bustsizelabels)):
    #     BustsizeTypeDict[bustsizelabels[i]] = i
    # bustsizelabel = [BustsizeTypeDict[bustsize[i]] for i in range(0,len(bustsize))]
    # bust_embedding = torch.nn.Embedding(bustsizelabels.__len__(), 64)
    #
    # # category
    # category = data["category"].values
    # categorylabels = list(set(category))
    # #print(categorylabels)
    # for i in range(0, len(categorylabels)):
    #     CategoryTypeDict[categorylabels[i]] = i
    # categorylabel = [CategoryTypeDict[category[i]] for i in range(0,len(category))]
    # category_embedding = torch.nn.Embedding(categorylabels.__len__(), 64)
    # #print(categorylabel)
    #
    # #item_id
    # item_id = data["item_id"].values
    # item_idlabels = list(set(item_id))
    # # print(len(item_idlabels))
    # for i in range(0, len(item_idlabels)):
    #     Item_idlabelsTypeDict[item_idlabels[i]] = i
    # item_idlabel = [Item_idlabelsTypeDict[item_id[i]] for i in range(0,len(item_id))]
    # item_embedding = torch.nn.Embedding(item_idlabels.__len__(), 64)
    # #print(item_idlabel)
    #
    # #user_id
    # user_id = data["user_id"].values
    # user_idlabels = list(set(user_id))
    # for i in range(0, len(user_idlabels)):
    #     User_idlabelsTypeDict[user_idlabels[i]] = i
    # user_idlabel = [User_idlabelsTypeDict[user_id[i]] for i in range(0,len(user_id))]
    # user_embedding = torch.nn.Embedding(user_idlabels.__len__(), 64)
    #
    # # rating
    # rating = data["rating"].values
    # ratinglabel = rating
    # rating_embedding = torch.nn.Embedding(5, 64)
    # # print(ratinglabel)
    #
    # # size
    # size = data["size"].values
    # sizelabellabels = list(set(size))
    # #print(sizelabellabels, len(sizelabellabels))
    # sizelabel = size
    # size_embedding = torch.nn.Embedding(59, 64)
    #
    # RES_embedding = []
    # RES_f = []
    # for i in range(0, len(data["rating"])):
    #     finalembedding = height_embedding(Variable(torch.LongTensor([heightlabel[i]])))[0]
    #     finalembedding = torch.cat((finalembedding, weight_embedding(Variable(torch.LongTensor([weightlabel[i]])))[0]),0)
    #     finalembedding = torch.cat((finalembedding, age_embedding(Variable(torch.LongTensor([agelabel[i]])))[0]),0)
    #     finalembedding = torch.cat((finalembedding, body_embedding(Variable(torch.LongTensor([bodytypelabel[i]])))[0]), 0)
    #     finalembedding = torch.cat((finalembedding, bust_embedding(Variable(torch.LongTensor([bustsizelabel[i]])))[0]),0)
    #     finalembedding = torch.cat((finalembedding, category_embedding(Variable(torch.LongTensor([categorylabel[i]])))[0]),0)
    #     finalembedding = torch.cat((finalembedding, item_embedding(Variable(torch.LongTensor([item_idlabel[i]])))[0]),0)
    #     #finalembedding = torch.cat((finalembedding, user_embedding(Variable(torch.LongTensor([user_idlabel[i]])))[0]),0)
    #     finalembedding = torch.cat((finalembedding, rating_embedding(Variable(torch.LongTensor([ratinglabel[i] - 1])))[0]),0)
    #     finalembedding = torch.cat((finalembedding, size_embedding(Variable(torch.LongTensor([sizelabel[i]])))[0]),0)
    #     print(finalembedding.shape)
    #     # final_f = [bmi_label[i] ,  bodytypelabel[i], bustsizelabel[i], categorylabel[i], item_idlabel[i], ratinglabel[i], sizelabel[i]]
    #     # RES_f.append(final_f)
    #     finalembedding = list(finalembedding.detach().numpy())
    #     RES_embedding.append(finalembedding)
    #
    # # output final embedding 768 + 448 = 1216
    # train_vector = open("vector_feature_train.txt", "w")
    # #TEXTembedding = ReadTextTrainmbedding()
    # for i in range(0, len(RES_embedding)):
    #     #RES_embedding[i] = TEXTembedding[i] + RES_embedding[i]
    #     for x in RES_embedding[i]:
    #         train_vector.write(str(x) + " ")
    #     train_vector.write("\n")
    #     if i % 1000 == 0:
    #         print(i)

    # train_file = open("feature_train.txt", "w")
    # # TEXTembedding = ReadTextTrainmbedding()
    # for i in range(0, len(RES_f)):
    #     # RES_embedding[i] = TEXTembedding[i] + RES_embedding[i]
    #     for x in RES_f[i]:
    #         train_file.write(str(x) + " ")
    #     train_file.write("\n")
    #     if i % 1000 == 0:
    #         print(i)


    # process TEST
    # Test

    # test = pd.read_csv("new_test.txt")
    # test_vector = open("vector_feature_test.txt", "w")
    # # testTextEmbedding = ReadTextTestembedding()
    # # assert(testTextEmbedding.__len__() == 57544)
    # for i in range(0, len(test["size"])):
    #     ######## BMI ##################
    #     if pd.isna(test["height"].values[i]) or pd.isna(test["weight"].values[i]):
    #         tempbmi = bmi_mean
    #     else:
    #         tempbmi = (heightTransform(test["height"].values[i])  * 703) / (float(test["weight"].values[i].replace("lbs","")) * float(test["weight"].values[i].replace("lbs","")))
    #     tempbmi_label = BMIlabel(tempbmi)
    #     tempbmi_embedding = bmi_embedding(Variable(torch.LongTensor([tempbmi_label])))[0]
    #     ######## body type ##################
    #     if test["body type"].values[i] not in BodyTypeDict.keys():
    #         tempbodytype_label = getRandom(bodylabels.__len__() - 1)
    #     else:
    #         tempbodytype_label = BodyTypeDict[test["body type"].values[i]]
    #     tempbodytype_embedding = body_embedding(Variable(torch.LongTensor([tempbodytype_label])))[0]
    #     ######## bust size ##################
    #     if pd.isna(test["bust size"].values[i]):
    #         tempbust_label = getRandom(3)
    #     else:
    #         tempbust_label = BUSTlabel(test["bust size"].values[i])
    #     #print(tempbust_label)
    #     tempbust_embedding = bust_embedding(Variable(torch.LongTensor([tempbust_label])))[0]
    #     ######## category ##################
    #     if test["category"].values[i] not in CategoryTypeDict.keys():
    #         tempcategorytype_label = getRandom(categorylabels.__len__() - 1)
    #     else:
    #         tempcategorytype_label = CategoryTypeDict[test["category"].values[i]]
    #     #print(tempcategorytype_label)
    #     tempcategorytype_embedding = category_embedding(Variable(torch.LongTensor([tempcategorytype_label])))[0]
    #     ######## item_id ##################
    #     if test["item_id"].values[i] not in Item_idlabelsTypeDict.keys():
    #         tempitem_id_label = getRandom(item_idlabels.__len__() - 1)
    #     else:
    #         tempitem_id_label = Item_idlabelsTypeDict[test["item_id"].values[i]]
    #     # print(tempcategorytype_label)
    #     tempitem_id_embedding = item_embedding(Variable(torch.LongTensor([tempitem_id_label])))[0]
    #     ######## rating ###################
    #     if pd.isna(test["rating"].values[i]):
    #         temprating_label = getRandom(1)
    #     else:
    #         temprating_label = RATINGlabel(test["rating"].values[i])
    #     temprating_embedding = rating_embedding(Variable(torch.LongTensor([temprating_label])))[0]
    #     ######## size ###################
    #     if pd.isna(test["rating"].values[i]):
    #         tempsize_label = getRandom(58)
    #     else:
    #         tempsize_label = test["rating"].values[i]
    #     #print(test["rating"].values[i], tempsize_label)
    #     tempsize_embedding = size_embedding(Variable(torch.LongTensor([tempsize_label])))[0]
    #
    #     tempfinalembedding = torch.cat((tempbmi_embedding, tempbodytype_embedding, tempbust_embedding, tempcategorytype_embedding, tempitem_id_embedding
    #                                     , temprating_embedding, tempsize_embedding ), 0)
    #     #tempres = testTextEmbedding[i] + list(tempfinalembedding.detach().numpy())
    #     tempres = list(tempfinalembedding.detach().numpy())
    #     # assert(tempres.__len__() == 446)
    #     for x in tempres:
    #         test_vector.write(str(x) + " ")
    #     test_vector.write("\n")
    #     if i % 1000 == 0:
    #         print(i)


if __name__ == '__main__':
    processTrain()