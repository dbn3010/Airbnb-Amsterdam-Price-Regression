# This is UI application of the project
from cProfile import label
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, LabelEncoder
import joblib
import plotly.express as px


st.set_page_config(
        page_title="AirBnB Prediction",
)
st.title("AirBnB Price Estimation")

features = ['accommodates', 'availability_365', 'bedrooms', 'host_identity_verified', 'instant_bookable', 'minimum_nights', 'number_of_reviews', 'reviews_per_month', 
'room_type', 'price', 'latitude', 'longitude']

listing_links = '''http://data.insideairbnb.com/united-states/nc/asheville/2021-10-18/data/listings.csv.gz
http://data.insideairbnb.com/united-states/tx/austin/2021-10-14/data/listings.csv.gz
http://data.insideairbnb.com/united-states/ma/boston/2021-10-19/data/listings.csv.gz
http://data.insideairbnb.com/united-states/mn/twin-cities-msa/2021-10-22/data/listings.csv.gz
http://data.insideairbnb.com/united-states/dc/washington-dc/2021-10-18/data/listings.csv.gz'''

listing_links = listing_links.split('\n')

@st.cache(allow_output_mutation=True)
def load_data():
    input_df = pd.DataFrame()
    for link in listing_links:
        city_df = pd.read_csv(filepath_or_buffer = link, header = 0)
        input_df = input_df.append(city_df)
    df = input_df[features]
    df['price'] = df['price'].str.replace(',', '')
    df['price'] = df['price'].str.replace('$', '')
    df['price'] = df['price'].astype(float)
    df.columns = df.columns.str.replace(' ', '_').str.strip().str.lower()
    df = df.dropna()
    df = df[df.price <= 1000]
    return df

my_df = load_data()
plot_df = my_df
my_df = my_df.drop(['price', 'latitude', 'longitude'], axis=1)




def data_transformation(df):
    df['host_identity_verified'] = df["host_identity_verified"].replace(['f','t'],[0, 1])
    df['instant_bookable'] = df["instant_bookable"].replace(['f','t'],[0, 1])
    df = pd.get_dummies(df, columns = ['room_type'])
    df.columns = df.columns.str.replace(' ', '_').str.strip().str.lower()   
    df = df.reset_index(drop = True)
    df = df.drop(labels=['room_type_entire_home/apt', 'room_type_hotel_room', 'room_type_shared_room'], axis=1)
    return df


df = data_transformation(my_df)


# Adding input fields to sidebar panel
accommodates = st.sidebar.number_input('Accomodates', 
    step = 1, value = 3
)

room_type = st.sidebar.selectbox('Room Type',
    my_df['room_type'].unique()
)

availability_365 = st.sidebar.select_slider('Room Availability',
    sorted(my_df['availability_365'].unique())
)

bedrooms = st.sidebar.number_input('Number of Bedrooms',
    step = 1, value = 1
)

host_identity_verified = st.sidebar.selectbox('Verified Host',
    my_df['host_identity_verified'].unique()
)

instant_bookable = st.sidebar.selectbox('Availbale Now',
    my_df['instant_bookable'].unique()
)

minimum_nights = st.sidebar.select_slider('Minimum Nights',
    sorted(my_df['minimum_nights'].unique())
)

number_of_reviews = st.sidebar.select_slider('number of Reviews',
    sorted(my_df['number_of_reviews'].unique())
)

reviews_per_month = st.sidebar.select_slider('# of Monthly Reviews',
    sorted(my_df['reviews_per_month'].unique())
)


# Creating a test data
test_data = {'accommodates': [accommodates], 'availability_365': [availability_365], 'bedrooms': [bedrooms], 'host_identity_verified': [host_identity_verified],
 'instant_bookable': [instant_bookable], 'minimum_nights': [minimum_nights], 'number_of_reviews': [number_of_reviews], 'reviews_per_month': [reviews_per_month],
 'room_type': [room_type]}

test_df = pd.DataFrame(test_data)


def test_transformation(df):
    df.rename(columns= {"room_type": "room_type_private_room"})
    df['room_type_private_room'] = 1
    df = df.drop(['room_type'], axis = 1)
    return df


test_df_tf = test_transformation(test_df)

# # # # Feature transformation
rc = RobustScaler()
X_train = rc.fit_transform(df)
X_test = rc.transform(test_df_tf)

def load_model():
    loaded_model = joblib.load("final_model_lr.joblib")
    return loaded_model

loaded_model = load_model()

y_test = loaded_model.predict(X_test)


# Creating the front end of the application
col1, col2 = st.columns(2)

plot_df_t = plot_df[(plot_df.accommodates == accommodates) & (plot_df.bedrooms == bedrooms)] 
avg_price = plot_df_t['price'].mean()
col1.metric("Predicted Price in USD", value = round(y_test[0],2))
col2.metric("Average Price in USD", value = round(avg_price, 2))


# Adding chart element to the UI

df = px.data.tips()
fig1 = px.histogram(plot_df_t['price'], x="price")
st.plotly_chart(fig1, use_container_width=True)


df = px.data.tips()
fig2 = px.histogram(plot_df_t, x="price",  color="room_type", hover_data=plot_df_t.columns)
fig2.update_layout(legend=dict(
    yanchor="top",
    xanchor="center"
))
st.plotly_chart(fig2, use_container_width=True)


map_data = plot_df[['latitude', 'longitude']]
st.map(map_data)
