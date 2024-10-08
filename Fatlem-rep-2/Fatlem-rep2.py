#!/usr/bin/env python
# coding: utf-8

# In[34]:


import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import f_regression
from sklearn.feature_selection import SelectKBest
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt


# In[9]:


full_data = pd.read_csv("suicide_rates_1990-2022.csv")


# In[15]:


japan_suicide_full_data = full_data.loc[full_data["CountryName"]=="Japan"]
japan_suicide_full_data


# In[11]:


japan_suicide_full_data.info()


# In[13]:


japan_suicide_relevan_data = japan_suicide_full_data.drop_duplicates()
print("Jumlah row yang dihilangkan karena duplicate: ", japan_suicide_full_data.size-japan_suicide_relevan_data.size)


# In[14]:


unrelevan_columns = ["RegionCode", "RegionName", "CountryCode", "CountryName", "CauseSpecificDeathPercentage"]
japan_suicide_relevan_data = japan_suicide_relevan_data.drop(unrelevan_columns, axis=1)
japan_suicide_relevan_data


# In[16]:


japan_suicide_relevan_data.isnull().sum()


# In[17]:


test = japan_suicide_relevan_data.loc[:, ["Year", "Population"]].drop_duplicates()
print(test)
for i in range(1990, 2022):
    if i not in test["Year"].values:
        print("Tahun yang tidak ada: ", i)


# In[18]:


japan_suicide_relevan_data = japan_suicide_relevan_data[japan_suicide_relevan_data["Year"] != 1990]
japan_suicide_relevan_data.isnull().sum()


# In[19]:


japan_suicide_relevan_data = japan_suicide_relevan_data[(japan_suicide_relevan_data["AgeGroup"]!="Unknown") | (japan_suicide_relevan_data["Generation"]!="Unknown")]
japan_suicide_relevan_data


# In[20]:


japan_suicide_relevan_data.isnull().sum()


# In[28]:


age_encoder = LabelEncoder()
age_encoder.fit(sorted(japan_suicide_relevan_data["AgeGroup"].unique()))
japan_suicide_relevan_data["AgeGroup"] = age_encoder.transform(japan_suicide_relevan_data["AgeGroup"])

sex_encoder = LabelEncoder()
sex_encoder.fit(sorted(japan_suicide_relevan_data["Sex"].unique()))
japan_suicide_relevan_data["Sex"] = sex_encoder.transform(japan_suicide_relevan_data["Sex"])

gen_encoder = LabelEncoder()
gen_encoder.fit(sorted(japan_suicide_relevan_data["Generation"].unique()))
japan_suicide_relevan_data["Generation"] = gen_encoder.transform(japan_suicide_relevan_data["Generation"])

japan_suicide_relevan_data


# In[29]:


X = japan_suicide_relevan_data.drop(["DeathRatePer100K", "SuicideCount"], axis=1)
y = japan_suicide_relevan_data["DeathRatePer100K"]


# In[30]:


bestfeatures = SelectKBest(score_func=f_regression, k=10)
fit = bestfeatures.fit(X,y)

dfscores = pd.DataFrame(fit.scores_)
dfcolumns = pd.DataFrame(X.columns)


# In[31]:


# menggabungkan 2 dataframe
featureScores = pd.concat([dfcolumns,dfscores],axis=1)
featureScores.columns = ['Specs','Score']

print(featureScores.nlargest(10,'Score')) 


# In[35]:


plt.figure(figsize=(16, 4))

plt.bar(featureScores["Specs"], featureScores["Score"], width=1, edgecolor="white", linewidth=0.7)
plt.title("Feature Score Importance Comparison for Analyzing Suicide Rate in Japan")

plt.xlabel("Feature")
plt.ylabel("Score")
plt.xticks(rotation=350)
plt.tick_params(axis='both', which='major', labelsize=8)

plt.show()


# In[36]:


# Memilih 5 Score teratas dari secore sebelumnya
selected_features = ["Sex", "AgeGroup", "Generation", "GNIPerCapita", "Year"]


# In[37]:


copy_relevan_data = pd.DataFrame()
copy_relevan_data["Year"] = japan_suicide_relevan_data["Year"]
copy_relevan_data["AgeGroup"] = age_encoder.inverse_transform(japan_suicide_relevan_data["AgeGroup"])

predict_purpose_dataset = copy_relevan_data
predict_purpose_dataset["AgeGroup"] = copy_relevan_data["AgeGroup"].apply(lambda x: x.replace(" years", "")) 
predict_purpose_dataset = predict_purpose_dataset.groupby(['AgeGroup', "Year"]).size().to_frame(name="Number")

predict_purpose_dataset = predict_purpose_dataset.reset_index()
predict_purpose_dataset.head(5)


# In[38]:


plt.figure(figsize=(16, 4))

plt.title("Change of Numbers of Suicide in Japan over 1991-2021")
plt.xlabel("Year")
plt.ylabel("Suicide Count")

plt.xticks(predict_purpose_dataset["Year"])
plt.tick_params(axis='both', which='major', labelsize=8)

age_groups = predict_purpose_dataset["AgeGroup"].unique()
for age in age_groups:
    certain_age = predict_purpose_dataset.loc[predict_purpose_dataset["AgeGroup"] == age]
    plt.plot(certain_age["Year"], certain_age["Number"], label=age)

plt.legend()
plt.show()


# In[39]:


new_age_encoder = LabelEncoder()
new_age_encoder.fit(predict_purpose_dataset["AgeGroup"])

predict_purpose_dataset["AgeGroup"] = new_age_encoder.transform(predict_purpose_dataset["AgeGroup"])
predict_purpose_dataset


# In[40]:


linear_predict_model = LinearRegression()
linear_predict_model.fit(predict_purpose_dataset.loc[:, ["AgeGroup", "Year"]], predict_purpose_dataset["Number"])


# In[41]:


age_groups = predict_purpose_dataset["AgeGroup"].unique()
new_predicted_year = [2022, 2023, 2024]

for age in age_groups:
    for year in new_predicted_year:
        predicted_suicide_rate = linear_predict_model.predict([[age, year]])[0]
        new_row = {'AgeGroup': age, 'Year': year, 'Number': predicted_suicide_rate}
        predict_purpose_dataset.loc[len(predict_purpose_dataset)] = new_row

predict_purpose_dataset.loc[predict_purpose_dataset['Year'].isin(new_predicted_year)].head(6)


# In[42]:


predict_purpose_dataset["AgeGroup"] = new_age_encoder.inverse_transform(predict_purpose_dataset["AgeGroup"])
predict_purpose_dataset


# In[43]:


plt.figure(figsize=(16, 4))
plt.title("Change of Numbers of Suicide in Japan over 1991-2021")
plt.xlabel("Year", fontsize=12)
plt.ylabel("Suicide Count", fontsize=12)


plt.xticks(predict_purpose_dataset["Year"])
plt.tick_params(axis='both', which='major', labelsize=8)

unencoded_age_groups = predict_purpose_dataset["AgeGroup"].unique()
color_choice = ["red", "blue", "purple", "black", "brown", "green", ""]

for age in unencoded_age_groups:
    certain_age = predict_purpose_dataset.loc[predict_purpose_dataset["AgeGroup"] == age]

    past_data = certain_age.loc[:certain_age[certain_age['Year'] == 2021].index[0]]
    predicted_data = certain_age.loc[certain_age[certain_age['Year'] == 2021].index[0]:]

    plt.plot(past_data["Year"], past_data["Number"], label=age, color=color_choice[0])
    plt.plot(predicted_data["Year"], predicted_data["Number"], linestyle='dotted', color=color_choice[0])
    color_choice = color_choice[1:]


plt.legend(bbox_to_anchor=(1.08, 1), loc='upper right')
plt.show()

