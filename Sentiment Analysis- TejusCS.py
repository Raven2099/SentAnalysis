#Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('ggplot')
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
SIA=SentimentIntensityAnalyzer()
def Custom(inputtext):
    sen = SIA.polarity_scores(inputtext)
    pos = sen["pos"]
    neg = sen["neg"]
    neutral = sen["neu"]
    print("Positive:\t",sen["pos"],"Negative:\t", sen["neg"],"Neutral:\t", sen["neu"],sep='\n')
    print("Compounded Score:\t",sen['compound'])
    if pos > neg:
        print("Mostly positive")
    elif neg > pos:
        print("Mostly negative")
    else:
        pass
    # Pie chart
    labels = 'POSITIVE', 'NEGATIVE', 'NEUTRAL'
    sizes = [pos, neg, neutral]
    explode = (0.1, 0, 0)  # only "explode" the 2nd slice (i.e. 'Hogs')

    fig1, ax1 = plt.subplots()
    ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
            shadow=True, startangle=90)
    ax1.axis('equal')

    plt.show()

def SenA():
    #reviews
    df=pd.read_csv('ProductReviews.csv')
    #Reading the csv file
    df=df.head(500) # since the csv file has too many reviews, we have limited the number of inputs for faster execution
    #print(df.shape)



    res = {}
    for i, row in df.iterrows():
        text = row['Text']
        myid = row['Id']
        res[myid] = SIA.polarity_scores(text)
    v = pd.DataFrame(res).T
    dv = v.reset_index().rename(columns={'index': 'Id'})
    dv = dv.merge(df, how='left')
    print(v)

    l=[]
    bestcase=''
    worstcase=''
    for col in v[['compound']]:
        cso=v[col]
        for i in cso.values:
            #print(i, sep='\n')
            l.append(i)
        l.sort()
        print("Best Score: ",l[-1])
        print("Worst Score: ",l[0])

    ax = df['Score'].value_counts().sort_index().plot(kind='bar',title='RATINGS',figsize=(10, 5))
    ax.set_xlabel('No of Stars')
    plt.show()
    axis=sns.barplot(data=dv,x='Score',y='compound')
    axis.set_title("Sentiment Analysis of Amazon Dataset")
    plt.show()

#main
ch=int(input("1.Enter a custom review\n2.Sentiment Analysis of Review Dataset"))
if ch==1:
    inp=input("Enter custom review")
    Custom(inp)
elif ch==2:
    SenA()
else:
    print("invalid input")
    exit()