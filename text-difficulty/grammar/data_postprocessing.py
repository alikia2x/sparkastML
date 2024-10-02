import pandas as pd

df = pd.read_csv("EGP_Derivied.csv")
newdf = pd.DataFrame()

levels_list=[]
sentences_list=[]
category_list=[]
for line in range(len(df.index)):
    examples = list(filter(None, df["Example"][line].split("\n")))
    lvl = df["Level"][line]
    cat = df["SuperCategory"][line] + '/' + df["SubCategory"][line]
    for sentence in examples:
        sentences_list.append(sentence)
        levels_list.append(lvl)
        category_list.append(cat)


newdf["Level"] = levels_list
newdf["Category"] = category_list
newdf["Sentence"] = sentences_list

newdf.to_csv("data.csv", index=False)