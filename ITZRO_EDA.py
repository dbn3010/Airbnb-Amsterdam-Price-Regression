#!/usr/bin/env python
# coding: utf-8

# In[1]:


#title Load Packages
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'
import numpy as np
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
import requests
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:





# In[2]:


url ='http://data.insideairbnb.com/the-netherlands/north-holland/amsterdam/2021-11-04/data/listings.csv.gz'
r = requests.get(url)
with open('listings.csv.gz', 'wb') as fo:
    fo.write(r.content)
pd_listings = pd.read_csv("listings.csv.gz")


# In[3]:


pd_listings


# In[5]:


pd_listings.neighbourhood_cleansed


# In[6]:



# select columns from pd_listings
pd_listings = pd_listings[['id','name','neighbourhood_cleansed','latitude','longitude','property_type','room_type','accommodates','bathrooms','bedrooms','beds','amenities','price','minimum_nights','number_of_reviews','review_scores_rating','review_scores_accuracy','review_scores_cleanliness','review_scores_checkin','review_scores_communication','review_scores_location','review_scores_value']]

# basic data cleaning
pd_listings['price'] = pd_listings['price'].str.replace("[$, ]", "").astype("float")

pd_listings.at[pd_listings['bathrooms'].isnull(), 'bathrooms'] = 0
pd_listings.at[pd_listings['bedrooms'].isnull(), 'bedrooms'] = 0 # yea there are 6 that has no bedrooms, but they do have 1 bathrooms
pd_listings.at[pd_listings['beds'].isnull(), 'beds'] = 0 # there's one listing for 1 guest, without any beds

pd_listings.at[pd_listings['review_scores_rating'].isnull(), 'review_scores_rating'] = 0
pd_listings.at[pd_listings['review_scores_accuracy'].isnull(), 'review_scores_accuracy'] = 0
pd_listings.at[pd_listings['review_scores_cleanliness'].isnull(), 'review_scores_cleanliness'] = 0
pd_listings.at[pd_listings['review_scores_checkin'].isnull(), 'review_scores_checkin'] = 0
pd_listings.at[pd_listings['review_scores_communication'].isnull(), 'review_scores_communication'] = 0
pd_listings.at[pd_listings['review_scores_location'].isnull(), 'review_scores_location'] = 0
pd_listings.at[pd_listings['review_scores_value'].isnull(), 'review_scores_value'] = 0

pd_listings.rename(columns={'id':'listing_id'}, inplace=True)


# In[7]:


# Load data from reviews.csv
url='http://data.insideairbnb.com/the-netherlands/north-holland/amsterdam/2021-11-04/data/reviews.csv.gz'
r =requests.get(url)
with open('reviews.csv.gz','wb') as fo:
    fo.write(r.content)
pd_reviews = pd.read_csv("reviews.csv.gz")

pd_reviews = pd_reviews[['id','listing_id','date']]

# basic conversions
pd_reviews['date'] = pd.to_datetime(pd_reviews['date'])

# pd_reviews.head()
print('reviews.csv loaded into pd_reviews')


# In[8]:


#check listings, reviews data count
print(sum(pd_listings['number_of_reviews']),len(pd_reviews))


# In[9]:


pd_listing_count_reviws = pd_reviews[['listing_id','id']].groupby(['listing_id']).count()
pd_listing_count_reviws.columns = ['# of reviews']
pd_listings_plus_reviews = pd.merge(pd_listings, pd_listing_count_reviws, on='listing_id')
pd_listings_plus_reviews.at[pd_listings_plus_reviews['# of reviews'].isnull(), '# of reviews'] = 0
pd_listings_plus_reviews[ pd_listings_plus_reviews['# of reviews'] != pd_listings_plus_reviews['number_of_reviews']]


# In[10]:


# Calculate estimated revenue for each listing

# get estimated bookings based on reviews
pd_bookings = pd.merge(pd_reviews, pd_listings, on='listing_id')
pd_bookings['estimated_revenue'] = pd_bookings['price'] * pd_bookings['minimum_nights']

# get revenue by listings
pd_listings_revenue = pd_bookings[['listing_id','estimated_revenue']].groupby(['listing_id']).sum()


pd_listings = pd.merge(pd_listings, pd_listings_revenue, on='listing_id', how='left')
pd_listings.at[pd_listings['estimated_revenue'].isnull(), 'estimated_revenue'] = 0


# In[11]:


# Top 5 highest revenue listings
pd_listings[['listing_id','number_of_reviews','minimum_nights','accommodates','bedrooms','beds','estimated_revenue']].sort_values('estimated_revenue', ascending=False).head()


# In[12]:


pd_listings[['listing_id','minimum_nights']].groupby(['minimum_nights']).count().sort_values('minimum_nights')


# In[13]:


# Top 5 highest revenue listings (minimum_nights <= 7)
pd_listings.loc[pd_listings['minimum_nights']<=7, ['listing_id','number_of_reviews','minimum_nights','accommodates','bedrooms','beds','estimated_revenue']].sort_values('estimated_revenue', ascending=False).head()


# In[14]:


# Showing 5 highest revenue listings (minimum_nights <= 4)
pd_listings.loc[pd_listings['minimum_nights']<=4, ['listing_id','number_of_reviews','minimum_nights','accommodates','bedrooms','beds','estimated_revenue']].sort_values('estimated_revenue', ascending=False).head()


# In[15]:


# Correlation between minimum nights and estimated revenue (not filtering min night 1000)
pd_listings[['minimum_nights','estimated_revenue']].corr()


# In[16]:


# Correlation between minimum nights and estimated revenue (filter min night 1000,1001)
pd_listings.loc[pd_listings['minimum_nights']<=7, ['minimum_nights','estimated_revenue']].corr()


# In[17]:


plt.scatter(pd_listings['longitude'], pd_listings['latitude'])


# In[18]:


# Best months for rental?

plt.figure(figsize=(15, 5))

# # bookings by month
plotdata = pd_reviews[['date']].groupby(pd_reviews["date"].dt.month).count()
plotdata.rename(columns={'date':'# of bookings'}, inplace=True)

ax = plt.subplot(1, 3, 1)
ax.set_title("# bookings by month")
plt.bar(plotdata.index, plotdata['# of bookings'])

# revenue by month
plotdata2 = pd_bookings[['date','estimated_revenue']].groupby(pd_bookings["date"].dt.month).sum()
plotdata2.rename(columns={'estimated_revenue':'revenue'}, inplace=True)

ax = plt.subplot(1, 3, 2)
ax.set_title("revenue by month")
plt.bar(plotdata2.index, plotdata2['revenue'])

# avg booking price by month
plotdata3 = pd.concat([plotdata, plotdata2], axis=1)
plotdata3['avg booking price'] = plotdata3['revenue'] / plotdata3['# of bookings']
plotdata3.head()

ax = plt.subplot(1, 3, 3)
ax.set_title("avg booking price by month")
plt.bar(plotdata3.index, plotdata3['avg booking price'])

_ = plt.plot()


# In[19]:


plt.figure(figsize=(13,7))
plt.title("Type of Room")
sns.countplot(pd_listings.room_type, palette="muted")
fig = plt.gcf()
plt.show()


# In[20]:


pd_listings.columns


# In[21]:


min_threshold,max_threshold= pd_listings.price.quantile([0.01,0.999])
min_threshold,max_threshold


# In[22]:


df_air_pnw= pd_listings[(pd_listings.price>min_threshold)&(pd_listings.price<max_threshold)]


# In[28]:


sns.set(rc={"figure.figsize": (10, 8)})
ax= sns.scatterplot(x=df_air_pnw.longitude, y=df_air_pnw.latitude,hue=pd_listings.room_type,palette='muted')
ax.set_title('Distribution of type of rooms across Amsterdam')


# In[33]:


pd_neighbourhood_revenue = pd_listings[['neighbourhood_cleansed','estimated_revenue']].groupby(['neighbourhood_cleansed']).mean().sort_values('estimated_revenue', ascending=False)
print(pd_neighbourhood_revenue)

pd_listings_plot = pd_listings[['neighbourhood_cleansed','longitude','latitude','estimated_revenue']]
pd_listings_plot.loc[:,'color'] = 0

color_value = 1
for neighbourhood in pd_neighbourhood_revenue[0:3].index:
  pd_listings_plot.at[pd_listings_plot['neighbourhood_cleansed'] == neighbourhood, 'color'] = color_value
  color_value -= 0.2
# plot
plt.figure(figsize=(4, 7))
ax = plt.subplot(1, 1, 1)
ax.set_title("Top 3 revenue neighbourhoods")
plt.scatter(pd_listings_plot['longitude'],
            pd_listings_plot['latitude'],
            cmap="coolwarm",
            c=pd_listings_plot['color']
           )

_ = plt.plot()    


# In[ ]:




