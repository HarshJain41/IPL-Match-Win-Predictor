import streamlit as st
import pandas as pd
import pickle
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

# Declaring the teams

teams = ['Sunrisers Hyderabad',
         'Mumbai Indians',
         'Royal Challengers Bangalore',
         'Kolkata Knight Riders',
         'Kings XI Punjab',
         'Chennai Super Kings',
         'Rajasthan Royals',
         'Delhi Capitals']

# declaring the venues

cities = ['Hyderabad', 'Bangalore', 'Mumbai', 'Indore', 'Kolkata', 'Delhi',
          'Chandigarh', 'Jaipur', 'Chennai', 'Cape Town', 'Port Elizabeth',
          'Durban', 'Centurion', 'East London', 'Johannesburg', 'Kimberley',
          'Bloemfontein', 'Ahmedabad', 'Cuttack', 'Nagpur', 'Dharamsala',
          'Visakhapatnam', 'Pune', 'Raipur', 'Ranchi', 'Abu Dhabi',
          'Sharjah', 'Mohali', 'Bengaluru']


pipe = pickle.load(open('pipe.pkl', 'rb'))
st.title('IPL Win Predictor')


col1, col2 = st.columns(2)

with col1:
    battingteam = st.selectbox('Select the batting team', sorted(teams))

with col2:

    bowlingteam = st.selectbox('Select the bowling team', sorted(teams))


city = st.selectbox(
    'Select the city where the match is being played', sorted(cities))


target = st.number_input('Target')

col3, col4, col5 = st.columns(3)

with col3:
    score = st.number_input('Score')

with col4:
    overs = st.number_input('Overs Completed')

with col5:
    wickets = st.number_input('Wickets Fallen')


if st.button('Predict Probability'):

    runs_left = target-score
    balls_left = 120-(overs*6)
    wickets = 10-wickets
    currentrunrate = score/overs
    requiredrunrate = (runs_left*6)/balls_left

    input_df = pd.DataFrame({'batting_team': [battingteam], 'bowling_team': [bowlingteam], 'city': [city], 'runs_left': [runs_left], 'balls_left': [
                            balls_left], 'wickets': [wickets], 'total_runs_x': [target], 'cur_run_rate': [currentrunrate], 'req_run_rate': [requiredrunrate]})

    # st.table(input_df)
#     cols_to_transform = ['batting_team','bowling_team','city']
#     ct = ColumnTransformer(
#     transformers=[
#         ('onehot', OneHotEncoder(), cols_to_transform)
#     ],
#     remainder='passthrough'
# )
    # input_df = pd.get_dummies(input_df, drop_first = True)
    # st.table(input_df)
    # encoder = OneHotEncoder()
    # encoder.fit(input_df)
    # one_hot_encoded_data = encoder.transform(input_df)
    # st.table(one_hot_encoded_data)
    result = pipe.predict_proba(input_df)
    lossprob = result[0][0]
    winprob = result[0][1]

    st.header(battingteam+"- "+str(round(winprob*100))+"%")

    st.header(bowlingteam+"- "+str(round(lossprob*100))+"%")
