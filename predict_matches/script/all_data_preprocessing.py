import pandas as pd
import numpy as np 



with open('../models/label_encoders.pkl','rb')as f:
    label_encoders=pickle.load(f)
def data_cleaning(data):
  data=data.copy()
  data['result']=np.where(data['HomeScore']==data['AwayScore'],0,np.where(data['HomeScore']>data['AwayScore'],1,2))# drow = 0 , homewin=1 , awaywin=2

  # creating day_of_week column and is_weekend column
  data['Date'] = pd.to_datetime(data['Date'])
  data['Time'] = pd.to_datetime(data['Time'])
  data['day_of_week'] = data['Date'].dt.dayofweek
  data['is_weekend']=np.where(data['day_of_week'].isin([5,6]),1,0)

  # dealling with null values in Attendance column in data
  data['Attendance']=data['Attendance'].fillna(data['Attendance'].mean())

  #encoding  venue and referee columns 
  data['Venue_n']=label_encoder.fit_transform(data['Venue'])
  data['Referee_n']=label_encoder.fit_transform(data['Referee'])

  # tunning hame and away columns 
  data['Home'] = data['Home'].str.strip().str.lower()
  data['Away'] = data['Away'].str.strip().str.lower()


  return data

def data_4_cleaning(data_4):
  data_4=data_4.copy()
  data_4['numeric_name']=label_encoder.fit_transform(data_4['team'])
  data_4['team'] = data_4['team'].str.strip().str.lower()

  return data_4

def data_5_cleaning(data_5):
  data_5=data_5.copy()
  data_5['team'] = data_5['team'].str.strip().str.lower()
  return data_5

def data_6_cleaning(data_6):
  data_6=data_6.copy()
  data_6['team'] = data_6['team'].str.strip().str.lower()

  return data_6

def creat_final_data(data,data_4,data_5,data_6):
  data=data_cleaning(data)
  data_4=data_4_cleaning(data_4)
  data_5=data_5_cleaning(data_5)
  data_6=data_6_cleaning(data_6)

  #merging data5 and data6
  merged_d5_d6 = pd.merge(
    left=data_6[['team', 'players', 'age', 'possession', 'goals', 'assists','penalty_kicks', 'penalty_kick_attempts', 'yellows', 'reds',
       'expected_goals', 'expected_assists', 'progressive_carries','progressive_passes']],
    right=data_5[['team','weekly']],
    
    left_on=['team'],
    
    right_on=['team'],
    
    how='left')

  #merging data4,data5 and data6
  merged_d4_d5_d6 = pd.merge(
  left=merged_d5_d6[['team', 'players', 'age', 'possession', 'goals', 'assists','penalty_kicks', 'penalty_kick_attempts', 'yellows', 'reds',
       'expected_goals', 'expected_assists', 'progressive_carries','progressive_passes', 'weekly']],
    right=data_4[['team','rank',  'win', 'loss', 'draw', 'goals', 'conceded', 'points','numeric_name']],
    
    left_on=['team'],
    
    right_on=['team'],
    
    how='left' )

  #mapping
  team_mapping = data_4[['team', 'numeric_name']].copy()
  data = data.merge(
    team_mapping.rename(columns={'team': 'Home', 'numeric_name': 'home_numeric_name'}),
    on='Home',
    how='left')
  data = data.merge(
    team_mapping.rename(columns={'team': 'Away', 'numeric_name': 'away_numeric_name'}),
    on='Away',
    how='left')

  #all work
  base_data = data[[
    'Date', 'Time', 'day_of_week',
    'Home', 'Away',
    'home_numeric_name', 'away_numeric_name',
    'Venue_n', 'Attendance', 'Referee_n',
    'is_weekend', 'result'
  ]]

  team_stats = merged_d4_d5_d6[[
    'numeric_name',
    'players', 'age', 'possession',
    'goals_x', 'assists', 'penalty_kicks', 'penalty_kick_attempts',
    'yellows', 'reds', 'expected_goals', 'expected_assists',
    'progressive_carries', 'progressive_passes', 'weekly',
    'rank', 'win', 'loss', 'draw', 'goals_y', 'conceded', 'points'
  ]].copy()

  final_data = base_data.merge(
    team_stats,
    left_on='home_numeric_name',
    right_on='numeric_name',
    how='left',
    suffixes=('', '_home')  # إضافة _home للأعمدة المكررة
  )

  home_columns_rename = {
    'players': 'home_players',
    'age': 'home_age',
    'possession': 'home_possession',
    'goals_x': 'home_goals_scored',
    'assists': 'home_assists',
    'penalty_kicks': 'home_penalty_kicks',
    'penalty_kick_attempts': 'home_penalty_attempts',
    'yellows': 'home_yellows',
    'reds': 'home_reds',
    'expected_goals': 'home_xG',
    'expected_assists': 'home_xA',
    'progressive_carries': 'home_prog_carries',
    'progressive_passes': 'home_prog_passes',
    'weekly': 'home_weekly_salary',
    'rank': 'home_rank',
    'win': 'home_wins',
    'loss': 'home_losses',
    'draw': 'home_draws',
    'goals_y': 'home_goals_total',
    'conceded': 'home_conceded',
    'points': 'home_points'
  }

  final_data.rename(columns=home_columns_rename, inplace=True)

  final_data.drop('numeric_name', axis=1, inplace=True)

  final_data = final_data.merge(
    team_stats,
    left_on='away_numeric_name',
    right_on='numeric_name',
    how='left',
    suffixes=('', '_away')
  )

  # إعادة تسمية الأعمدة للفريق الضيف
  away_columns_rename = {
    'players': 'away_players',
    'age': 'away_age',
    'possession': 'away_possession',
    'goals_x': 'away_goals_scored',
    'assists': 'away_assists',
    'penalty_kicks': 'away_penalty_kicks',
    'penalty_kick_attempts': 'away_penalty_attempts',
    'yellows': 'away_yellows',
    'reds': 'away_reds',
    'expected_goals': 'away_xG',
    'expected_assists': 'away_xA',
    'progressive_carries': 'away_prog_carries',
    'progressive_passes': 'away_prog_passes',
    'weekly': 'away_weekly_salary',
    'rank': 'away_rank',
    'win': 'away_wins',
    'loss': 'away_losses',
    'draw': 'away_draws',
    'goals_y': 'away_goals_total',
    'conceded': 'away_conceded',
    'points': 'away_points'}

  final_data.rename(columns=away_columns_rename, inplace=True)

  final_data.drop('numeric_name', axis=1, inplace=True)
  final_data=final_data[['home_numeric_name',
       'away_numeric_name', 'Venue_n', 'Attendance','home_age', 'home_possession',
       'home_goals_scored', 'home_assists', 'home_penalty_kicks',
       'home_penalty_attempts', 'home_yellows', 'home_reds', 'home_xG',
       'home_xA', 'home_prog_carries', 'home_prog_passes',
       'home_weekly_salary', 'home_rank', 'home_wins', 'home_losses',
       'home_draws', 'home_goals_total', 'home_conceded', 'home_points',
       'away_age', 'away_possession', 'away_goals_scored',
       'away_assists', 'away_penalty_kicks', 'away_penalty_attempts',
       'away_yellows', 'away_reds', 'away_xG', 'away_xA', 'away_prog_carries',
       'away_prog_passes', 'away_weekly_salary', 'away_rank', 'away_wins',
       'away_losses', 'away_draws', 'away_goals_total', 'away_conceded',
       'away_points',
       'result']]
  return final_data

def preprocessing_steps(data,data_4,data_5,data_6):
  print(" data preparing and cleaning ...\n")
  data=data_cleaning(data)
  print("done\n")
  print(" data_4 preparing and cleaning ...\n")
  data_4=data_4_cleaning(data_4)
  print("done\n")
  print(" data_5 preparing and cleaning ...\n")
  data_5=data_5_cleaning(data_5)
  print("done\n")
  print(" data_6 preparing and cleaning ...\n")
  data_6=data_6_cleaning(data_6)
  print("done\n")
  print("data merging and finding final data ...\n")
  final_data=creat_final_data(data,data_4,data_5,data_6)
  print("done\n")
  return final_data

  

  