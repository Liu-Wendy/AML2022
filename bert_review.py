import pandas as pd
import numpy as np
from transformers import BertTokenizer, BertModel
# Press the green button in the gutter to run the script.

# bert transformers
tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
model = BertModel.from_pretrained("bert-base-cased")

def getSentenceEncoding(text):
    encoded_input = tokenizer(text, return_tensors='pt')
    # print(encoded_input)
    output = model(**encoded_input)
    # print(output)
    return output[1].data.cpu().numpy()

def getTrainText():
    data = pd.read_csv("new_train.csv")
    cols = data.columns
    # print(data.head(0),type(cols))
    # get embedding of sentences
    SentenFileDescription = open("save_encoding_sen_description_train.txt", "w")
    SentenFileAmenity = open("save_encoding_sen_Amenity_train.txt", "w")

    description_sum = data["description"]
    print(len(description_sum))
    amenities_sum = data["amenities"]
    print(len(amenities_sum))

    for i in range(0,len(description_sum)):
        if(type(description_sum[i]) != type(description_sum[0])):
            description_sum[i] = "none"
        code = getSentenceEncoding(description_sum[i])
        print(i)
        for x in code[0]:
            SentenFileDescription.write(str(x) + " ")
        SentenFileDescription.write("\n")

    for i in range(0,len(amenities_sum)):
        if(type(amenities_sum[i]) != type(amenities_sum[0])):
            amenities_sum[i] = "none"
        code = getSentenceEncoding(amenities_sum[i])
        print(i)
        for x in code[0]:
            SentenFileAmenity.write(str(x) + " ")
        SentenFileAmenity.write("\n")

def getTestText():
    data = pd.read_csv("new_test.txt")
    cols = data.columns
    print(data.head(0),type(cols))
    # get embedding of sentences
    SentenFile = open("bert_text/save_encoding_sen_review_test.txt", "w")
    review_sum = data["review_summary"]
    for i in range(0,len(review_sum)):
        #print(review_sum[i])
        if(type(review_sum[i]) != type(review_sum[0])):
            review_sum[i] = "none"
        code = getSentenceEncoding(review_sum[i])
        for x in code[0]:
            SentenFile.write(str(x) + " ")
        SentenFile.write("\n")
        if (i + 1) % 100 == 0:
            print(i)

if __name__ == '__main__':
    getTrainText()
    # getTestText()

    # getSentenceEncoding("Replace me by any Text you'd like.")