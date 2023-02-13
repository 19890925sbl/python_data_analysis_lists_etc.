from this import d

import pandas as pd
s=pd.series([10,77,12,4,5])
type(s)
s.index
s.dtype
s.size
s.ndim
s.values
type(s.values)
s.head(3)
s.tail(3)

#Veri Okuma(Reading Data)
import pandas as pd
df=pd.read_csv
("datasets/advertising.csv")
df.head()

#Veriye Hızlı Bakış(Quick Look At Data)

import pandas as pd
import seaborn as sns
sns.bad_dataset("titanic")
sns.bad_dataset("titanic")
df.head()
df.tail()
df.shape
df.info()
df.columns
df.index
df.describe().T
df.isnull().values.any()
df.isnull().sum()
df["sex"].head()
df["sex"].value_count()

#Pandas'ta Seçim İşlemleri(Selection In Pandas)#

 import pandas as pd
 import seaborn as sns

 df=sns.load_dataset("titanic")
 df.head()
 df.index
 df[0:13]
 df.drop(0,axis=0).head()
 delete_indexes=[1,3,5,7]
df.drop(delete_indexes,axis=0).head(0)

#Değişkenleri İndex'e Çevirmek
df["age"].head()
df.age.head()
df.index=df["age"]
df.drop("age",axis=1).head()
df.drop("age",axis=1,inplace=True)
df.head()

df.index
df["age"]=df.index
df.head()

df.reset_index().head()
df=df.reset_index()
df.head()

#Değişken Üzerinde İşlemler
 import pandas as pd
 import seaborn as sns
 pd.set_option('display.max_columns',None)
 df=sns.load_dataset("titanic")

 "age" in df
 df["age"].head()
 df.age.head()

 df["age"].head()
 type(df["age"].head())

 df[["age"]].head()
 type([["age"]].head())

 df[["age","alive"]]
 col_names=["age","adult_male","alive"]
 df[col_names]

 df["age2"]=df["age"]**2
 df["age3"]=df["age"]/df["age2"]
 df.drop("age3",axis=1).head()
 df.drop(col_names,axis=1).head()
 df.loc[:,~df.columns.str.contains("age")].head()


 #Loc&Iloc

import pandas as pd
import seaborn as sns
pd.set_option('display.max_columns',None)
df=sns.load_dataset("titanic")
df.head()

#iloc :integer based selection
df.iloc[0:3]
df.iloc[0:0]

#loc:label based selection
df.loc[0:3]
df.iloc[0:3,0:3]
df.loc[0:3,"age"]
col_names=["age","embarked","alive"]
df.loc[0:3,col_names]

#Koşullu Seçim

import pandas as pd
import seaborn as sns
pd.set_option('display.max_columns',None)
df=sns.load_dataset("titanic")
df.head()

df[df["age"]>50].head()

df[df["age"]>50]["age"].count

df.loc[df["age"]>50,["age","class"]].head()

df.loc[(df["age"]>50)&(df["sex"]=="male"),["age","class"]].head()

[(df["age"]>50)&(df["sex"]=="male")&(df["embark_town"]=="Cherbourg"),["age","class","embark_town"]].head()

df.loc[(df["age"]>50)
&(df["sex"]=="male")
&(df["embark_town"]=="Charbourg"),
["age","class","embark_town"]].head()

df["embark_town"].value_counts()

df.new=df.loc[(df["age"]>50)
&(df["sex"]=="male")
&((df["embark_town"]=="Cherbourg")|(df["embark_town"]=="Southampton")),["age","class","embark_town"]]

df.new["embark_town"].value_counts()

import pandas as pd
import seaborn as sns
pd.set_option('display.max_columns',None)
df=sns.load_dataset("titanic")
df.head()
df["age"].mean()
df.groupby("sex")["age"].mean()
df.groupby("sex").agg({"age":"mean"})
df.groupby("sex").agg({"age":["mean","sum"]})
df.groupby("sex").agg({"age":["mean","sum"],"survived":"mean"})
df.groupby(["sex","embark_town","class"]).agg({
    "age":["mean"],
    "survived":"mean",
    "sex":"count"
})

#Pivot Table

import pandas as pd
import seaborn as sns
pd.set_option('display.max_columns',None)
df=sns.load_dataset("titanic")
df.head()
df.pivot_table("survived","sex","embarked")
df.pivot_table("survived","sex","embarked",aggfunc="std")
df.pivot_table("survived","sex",["embarked","class"])
df["new_age"]=pd.cut(df["age"],[0,10,18,25,40,90])
df.pivot_table("survived","sex",["new_age","class"])
pd.set_option('display.width',500)

#Apply ve Lambda
import pandas as pd
import seaborn as sns
pd.set_option('display.max_columns',None)
pd.set_option('display.width',500)
df=sns.load_dataset("titanic")
df.head()
df["age2"]=df["age"]*2
df["age3"]=df["age"]*5
(df["age"]/10).head()
(df["age2"]/10).head()
(df["age3"]/10).head()

for col in df.columns:
    if "age" in col:
        print(col)

for col in df.columns:

    if "age" in col:
        print((df[col]/10).head())

for col in df.columns:
    if "age" in col:
        df[col]=df[col]/10
        df.head()

df[["age","age2","age3"]].apply(lambda x:x/10).head()
df.loc[:,df.columns.str.contains("age")].apply(lambda x:x/10).head()
df.loc[:,df.columns.str.contains("age")].apply(lambda x:(x-x.mean())/x.std()).mean()

def standart_scale(col_name):
    return (col_name-col_name.mean())/col_name.std()
df.loc[:,["age","age2","age3"]]=df.loc[:,df.columns.str.contains("age")].apply(standart_scale(2)).head()

#df.loc[:,["age","age2","age3"]]=df.loc[:,df.columns.str.contains("age")].apply(standart_scale(1))

#Birleştirme (Join) İşlemleri
import numpy as np
import pandas as pd
m=np.random.randint(1,30,size=(5,3))
df1=pd.DataFrame(m,columns=["var1","var2","var3"])
df2=df1+99
pd.concat([df1,df2])
pd.concat([df1,df2],ignore_index=True)

#Merge İle Birleştirme İşlemleri
df1=pd.DataFrame({'employees':['john','dennis','mark','maria'],
                  'group':['accounting','engineering','engineering','hr']})

df2=pd.DataFrame({'employees':['mark','john','dennis','maria'],
                  'start_date':[2010,2009,2014,2019]})
pd.merge(df1,df2)
d.merge(df1,df2,on='employees')
df3=pd.merge(df1,df2)
df4=pd.DataFrame({'group':['accounting','engineering','hr'],
                  'manager':['caner','mustafa','berkcan']})
pd.merge(df3,df4)
