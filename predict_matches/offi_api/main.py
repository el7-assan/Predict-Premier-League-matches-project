from fastapi import FastAPI,HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np 
import pickle
from pydantic import BaseModel
import traceback
import sklearn


app=FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins. Use specific domains for better security.
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"]   # Allow all headers
)

#loading models 
with open('../models/teams.pkl','rb')as f:
    teams=pickle.load(f)
teams=pd.DataFrame(teams.value_counts())
teams.drop('count',axis=1,inplace=True)
teams.reset_index(inplace=True)

with open ('../models/venue.pkl','rb')as f:
    venue=pickle.load(f)
venue=pd.DataFrame(venue.value_counts())
venue.drop('count',axis=1,inplace=True)
venue.reset_index(inplace=True)

with open('../models/statistics.pkl','rb')as f:
    statistics=pickle.load(f)

with open('../models/final_data.pkl','rb')as f:
    final_data=pickle.load(f)

class game(BaseModel):
    home :str
    away:str
    studium:str|None=None
    attendence:int|None=None




@app.post("/predict/")
async def predict(Match:game):
    try:
        if(Match.studium is not None and Match.attendence is not None):
                      match_data = {
                      'home_numeric_name': teams.loc[teams['team'] == Match.home, 'numeric_name'].values[0],
                      'away_numeric_name': teams.loc[teams['team'] == Match.away, 'numeric_name'].values[0],
                      'Venue_n': venue.loc[venue['Venue'] == Match.studium, 'Venue_n'].values[0],
                      'Attendance': Match.attendence}
        elif(Match.studium is not None and Match.attendence is None):
                       match_data = {
                      'home_numeric_name': teams.loc[teams['team'] == Match.home, 'numeric_name'].values[0],
                      'away_numeric_name': teams.loc[teams['team'] == Match.away, 'numeric_name'].values[0],
                      'Venue_n': venue.loc[venue['Venue'] == Match.studium, 'Venue_n'].values[0]}
        elif(Match.studium is  None and Match.attendence is not None):
                       match_data = {
                      'home_numeric_name': teams.loc[teams['team'] == Match.home, 'numeric_name'].values[0],
                      'away_numeric_name': teams.loc[teams['team'] == Match.away, 'numeric_name'].values[0],
                      'Attendance': Match.attendence}
        else:
                       match_data = {
                      'home_numeric_name': teams.loc[teams['team'] == Match.home, 'numeric_name'].values[0],
                      'away_numeric_name': teams.loc[teams['team'] == Match.away, 'numeric_name'].values[0]}


        match = pd.DataFrame([match_data])

        if(Match.studium is not None and Match.attendence is not None):
            match = match.merge(
            statistics[['numeric_name', 'age', 'expected_goals', 'win', 'loss', 'draw', 'weekly', 'goals_x', 'conceded', 'points']],
            left_on='home_numeric_name',right_on='numeric_name',how='left').rename(columns={
            'age': 'home_age','expected_goals': 'home_xG','win': 'home_wins','loss': 'home_losses','draw': 'home_draws','weekly': 'home_weekly_salary',
            'goals_x': 'home_goals_total','conceded': 'home_conceded','points': 'home_points'}).drop('numeric_name', axis=1)
            match = match.merge(
            statistics[['numeric_name', 'age', 'expected_goals', 'win', 'loss', 'draw', 'weekly', 'goals_x', 'conceded', 'points']],
            left_on='away_numeric_name',right_on='numeric_name',how='left').rename(columns={
            'age': 'away_age','expected_goals': 'away_xG','win': 'away_wins','loss': 'away_losses','draw': 'away_draws','weekly': 'away_weekly_salary',
            'goals_x': 'away_goals_total','conceded': 'away_conceded','points': 'away_points'}).drop('numeric_name', axis=1)
    
        elif(Match.studium is  None and Match.attendence is not None):
            match = match.merge(
            statistics[['numeric_name','expected_goals', 'win', 'loss', 'draw','conceded', 'points']],
            left_on='home_numeric_name',right_on='numeric_name',how='left').rename(columns={
            'expected_goals': 'home_xG','win': 'home_wins','loss': 'home_losses','draw': 'home_draws',
            'conceded': 'home_conceded','points': 'home_points'}).drop('numeric_name', axis=1)
            match = match.merge(
            statistics[['numeric_name','expected_goals', 'win', 'loss', 'draw','conceded', 'points']],
            left_on='away_numeric_name',right_on='numeric_name',how='left').rename(columns={
            'expected_goals': 'away_xG','win': 'away_wins','loss': 'away_losses','draw': 'away_draws',
            'conceded': 'away_conceded','points': 'away_points'}).drop(['numeric_name','Venue_n'], axis=1,errors='ignore')

        else:
            match = match.merge(
            statistics[['numeric_name', 'age', 'expected_goals', 'win', 'loss', 'draw', 'weekly', 'goals_x', 'conceded', 'points']],
            left_on='home_numeric_name',right_on='numeric_name',how='left').rename(columns={
            'age': 'home_age','expected_goals': 'home_xG','win': 'home_wins','loss': 'home_losses','draw': 'home_draws','weekly': 'home_weekly_salary',
            'goals_x': 'home_goals_total','conceded': 'home_conceded','points': 'home_points'}).drop('numeric_name', axis=1)
            match = match.merge(
            statistics[['numeric_name', 'age', 'expected_goals', 'win', 'loss', 'draw', 'weekly', 'goals_x', 'conceded', 'points']],
            left_on='away_numeric_name',right_on='numeric_name',how='left').rename(columns={
            'age': 'away_age','expected_goals': 'away_xG','win': 'away_wins','loss': 'away_losses','draw': 'away_draws','weekly': 'away_weekly_salary',
            'goals_x': 'away_goals_total','conceded': 'away_conceded','points': 'away_points'}).drop('numeric_name', axis=1)
            match.drop(['Venue_n','Attendance'], axis=1, errors='ignore', inplace=True)


        if 'Attendance'in match.columns and 'Venue_n'in match.columns:
            #Logistic Regression + Filter_3
            with open('../models/model_1_lr.pkl', 'rb')as f:
                      model_1 = pickle.load(f)
            y_pred=model_1.predict(match)
            if(y_pred==1):
                      result=Match.home
            elif(y_pred==2):
                      result=Match.away
            else :
                  result="Drow"  
        elif('Attendance'in match.columns):
            #Logistic Regression + Filter_2
            with open('../models/model_2_lr.pkl', 'rb')as f:
                      model_2 = pickle.load(f)
            y_pred=model_2.predict(match)
            if(y_pred==1):
                  result=Match.home
            elif(y_pred==2):
                  result=Match.away
            else :
                  result="Drow"       
        else:
            # Random Forest + Filter_4
            with open('../models/model_3_rf.pkl', 'rb')as f:
                      model_3 = pickle.load(f)
            y_pred=model_3.predict(match)
            if(y_pred==1):
                  result=Match.home
            elif(y_pred==2):
                  result=Match.away
            else :
                  result="Drow" 


        return {"prediction": result}
 
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=400, detail=str(e))







