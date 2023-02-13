#GÖREV1:List Comprehensions yapısı kullanarak car_crashes verisindeki numeric değişkenlerin
#isimlerini büyük harfe çeviriniz ve başka NUM ekleyiniz.

import seaborn as sns
import pandas as pd
pd.set_option('display.max_rows',None)
pd.set_option('display.max_columns',None)
pd.set_option('display.width',None)
df=sns.load_dataset("car_crashes")
df.columns
df.info()

["NUM_"+col.upper() if df[col].dtype!="0" else col.upper()
 for col in df.columns]
#BEKLENEN ÇIKTI:
['NUM_TOTAL','NUM_SPEEDING','NUM_ALCOHOL','NUM_NOT_DISTRACTED','NUM_NO_PREVIOUS','NUM_INS_PREMIUM','NUM_INS_COSSES',
 'ABBREV']
#Numeric olmayan değişkenlerin de isimleri büyümeli tek bir list comprehension yapısı kullanılmalı.

#GÖREV2:List Comprehensions yapısı kullanarak car_crashes verisinde isminde "no" barındırmayan
#değişkenlerin isimlerinin sonuna  "FLAG" yazınız
#BEKLENEN ÇIKTI:

[col.upper()+"_FLAG" if "no" not in col else col.upper() for col in df.columns]


['TOTAL_FLAG','SPEEDING_FLAG','ALCOHOL_FLAG','NOT_DISTRACTED','NO_PREVIOUS','INS_PREMIUM_FLAG',
 'INS_LOSSES_FLAG','ABBREV_FLAG']
#Tüm değişkenlerin isimleri büyük harf olmalı.Tek bir list comprehensions yapısı ile yapılmalı.

#GÖREV3:List Comprehensions yapısı kullanarak aşağıda verilen değişkenin isimlerinden FARKLI OLAN
#değişkenlerin isimlerini seçiniz  ve yeni bir dataframe oluşturunuz.

ag_list=["abbrev","no_previous"]
new_cols=[colfor col in df.columns if col not in ag_list]
new_df=df[new_cols]
new_df.head()


#Önce verilen listeye göre list comprehensions kullanarak new_cols adında yeni liste oluşturunuz.
#sonra df[new_cols]nile bu değişkenleri seçerek yeni bir df oluşturunuz ve adını new_df olarak isimlendiriniz.

import numpy as np
import  seaborn as sns
import pandas as pd
pd.set_option('display.max_rows',None)
pd.set_option('display.max_columns',None)
pd.set_option('display.width',1000)

###########################################PANDAS ALIŞTIRMALAR###################################################3
#1)GÖREV:Seaborn kütüphanesi içerisinden Titanic Veri setini tanımllayınız.

df=sns.load_dataset("titanic")
df.head()
df.shape()

#2)Görev:Titanic Veri setindeki kadın ve erkek yolcuların sayısını bulunuz.

df["sex"].value_counts()

#3)Görev:Her bir sütuna ait unique değerlerin sayısını bulunuz.

df.nunique()

#4)Görev:pclass değişkenlerinin unique değerlerinin sayısını bulunuz.

df["pclass"].unique()

#5)Görev:pclass ve parch değişkenlerinin unique değerlerinin sayısını bulunuz.

df[["pclass","parch"]].nunique()

#6)Görev:embarked değişkeninin tipini kontrol ediniz.Tipini category olarak değiştiriniz ve tekrar
#kontrol ediniz.

df["embarked"].dtype

#7)Görev:embarked değeri C olanlatın tüm bilgilerini gösteriniz.

df[df["embarked"]=="C"].head(10)

#8)Görev:embarked değeri 5 olmayanların tüm bilgilerini gösteriniz.

df[df["embarked"]=="S"].head(10)

df[df["embarked"]=="S"]["embarked"].unique()

df[~(df["embarked"]=="S")]["embarked"].unique()

#9)Görev:Yaşı 30'dan küçük ve kadın olan yolcuların tüm bilgilerini gösteriniz.

df[(df["age"]<30)&(df["sex"]=="female")].head()

#10)Görev:Fare'i 500'dab büyük vaya yaşı 70'den büyük yolcuların bilgilerini gösteriniz.

df[(df["fare"]>500)|(df["age"]>70)].head()

#11)Görev:Herbir değişkendeki boş değerlerin toplamını bulunuz.

df.isnull().sum()

#12)Görev:who değişkenini dataframe'den çıkartınız.

df.drop("who",axis=1,inplace=True)
df.head()


#13)Görev:deck değişkenindeki boş değerleri deck değişkeninin en çok tekrar eden değeri (mode)ile doldurunuz.

type(df["deck"].mode())
df["deck"].mode()[0]
df["deck"].fillna(df["deck"].mode()[0],inplace=True)
df["deck"].isnull().sum()

#14)Görev:age değişkenindeki boş değerleri age değişkeninin medyanı ile doldurunuz.

df["age"].fillna(df["age"].median(),inplace=True)
df.isnull().sum()

#15)Görev:survived değişkeninin boş değerleri age değişkeninin medyanı ile doldurunuz.

df.groupby(["pclass","sex"]).agg({"survived":["sum","count","mean"]})

#16)Görev:30 yaş altında olanlar 1,30'a eşit ve üstünde olanlara 0 vericek bir fonksiyon yazınız.Yazdığınız
#fonksiyonu kullanarak titanik veri setinde age_flag adında bir değişken oluşturunuz.
#(apply ve lambda yapılarını kullanınız)

def age_30(age):
 if age<30:
  return 1
 else:
  return 0
 df["age_flag"]=df["age"].apply(lambda  x:age_30(x))
 df["age_flag"]=df["age"].apply(lambda  x:1 if x<30 else 0)

#17)Görev:Seaborn kütüphanesi içerisinden Tips veri setini tanımlayınız.

df=sns.load_dataset("tips")
df.head()
df.shape

#18)Görev:Time değişkeninin kategorilerine(Dinner,Lunch)göre total_bill değerlerinin toplamını min,
#max ve ortalamasını bulunuz.

df.groupby("time").agg({"total_bill":["sum","min","mean","max"]})

#19)Görev:Güçlere ve time a göre total_bill değerlerinin toplamını,min,max ve ortalamasını bulunuz.

df.groupby(["day","time"]).agg({"total_bill":["sum","min","mean","max"]})

#20)Görev:Lunch zamanına ve kadın müşterilere ait total_bill ve tip değerlerinin day'e göre toplamını,
#min,max ve ortalamasını bulunuz.

df[(df["time"]=="lunch")&(df["sex"]=="Female")].groupby("day").agg({"total_bill":["sum","min","max","mean"],
                                                                    "tip":["sum","min","max","mean"]})

#21)Görev:size 3'ten küçük,total_bill'i 10'dan büyük olan siparişlerin ortalaması nedir?(loc kullanınız.)

df.loc[(df["size"]<3)&(df["total_bill"]>10),"total_bill"].mean()

#22)Görev:total_bill_tip_sum adında yeni bir değişken oluşturunuz.Herbir müşterinin ödediği totalbill ve tip'in
#toplamını versin.

df["total_bill_tip_sum"]=df["total_bill"]+df["tip"]
df.head()

#23)Görev:total_bill_tip_sum değişkenine göre büyükten küçüğe sıralayınız ve ilk 30 kişiyi yeni bir dataframe'e
#atayınız.

new_df=df.sort_values("total_bill_tip_sum",ascending=False[:30])
new_df.shape