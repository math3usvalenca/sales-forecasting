import pickle
import pandas as pd
from math import isnan
import numpy as np
import datetime
import os

class Rossmann(object):
    
    def __init__(self):
          
        self.competition_distance_scaler=pickle.load(open(
            os.path.join(os.getcwd(),'parameter/competition_distance_scaler.pkl')
         ,'rb'
        ))
        self.competition_time_month_scaler=pickle.load(open(
            os.path.join(os.getcwd(),'parameter/competition_time_month_scaler.pkl')
            ,'rb'
        ))
        self.promo_time_week_scaler=pickle.load(open(
            os.path.join(os.getcwd(),'parameter/promo_time_week_scaler.pkl')
            ,'rb'
        ))
        self.year_scaler=pickle.load(open(
            os.path.join(os.getcwd(),'parameter/year_scaler.pkl')
            ,'rb'
        ))
        self.store_type_scaler=pickle.load(open(
            os.path.join(os.getcwd(),'parameter/store_type_scaler.pkl')
            ,'rb'
        ))
      
    def data_cleaning( self, df1 ):
         ##  Data Types
        df1['Date'] = pd.to_datetime( df1['Date'] )
        # Fillout Na
        df1['CompetitionDistance']  = df1['CompetitionDistance'].apply(lambda x: 200000.0 if isnan(x) else x)

        df1['CompetitionOpenSinceMonth'] = df1.apply(
    lambda x: x['Date'].month if isnan(x['CompetitionOpenSinceMonth']) else x['CompetitionOpenSinceMonth'],axis=1)
        
        df1['CompetitionOpenSinceYear'] = df1.apply(
    lambda x: x['Date'].year if isnan(x['CompetitionOpenSinceYear']) else x['CompetitionOpenSinceYear'],axis=1)
        
        df1['Promo2SinceWeek'] = df1.apply(
    lambda x: x['Date'].week if isnan(x['Promo2SinceWeek']) else x['Promo2SinceWeek'],axis=1)
        df1['Promo2SinceYear'] = df1.apply(
    lambda x: x['Date'].year if isnan(x['Promo2SinceYear']) else x['Promo2SinceYear'],axis=1)

        # dicionário para o map
        month_map = {
            1:'Jan',
            2:'Fev',
            3:'Mar',
            4:'Apr',
            5:'May',
            6:'Jun',
            7:'Jul',
            8:'Aug',
            9:'Sep',
            10:'Oct',
            11:'Nov',
            12:'Dec'
        }

        df1['PromoInterval'].fillna(0,inplace=True)
        # mapeando os nomes dos meses
        df1['MonthMap'] = df1['Date'].dt.month.map(month_map)
        # agora verificar se um determinado mês existe em PromoInterval
        df1['IsPromo'] = df1[['PromoInterval','MonthMap']].apply(
    lambda x: 0 if x['PromoInterval']==0 else 1 if x['MonthMap'] in x['PromoInterval'].split(',') else 0, axis=1)
        
        # change data types
        df1['CompetitionOpenSinceMonth'] = df1['CompetitionOpenSinceMonth'].astype( int )
        df1['CompetitionOpenSinceYear'] = df1['CompetitionOpenSinceYear'].astype(int)
        # promo2
        df1['Promo2SinceWeek'] = df1['Promo2SinceWeek'].astype( int )
        df1['Promo2SinceYear'] = df1['Promo2SinceYear'].astype( int )
        
        return df1
    
    def feature_engineering( self, df2 ):
        # year
        df2['Year'] = df2['Date'].dt.year
        # month
        df2['Month'] = df2['Date'].dt.month
        # day
        df2['Day'] = df2['Date'].dt.day
        # week of year
        df2['WeekOfYear'] = df2['Date'].dt.weekofyear
        # year week
        df2['YearWeek'] = df2['Date'].dt.strftime( '%Y-%W' )
        # competition since
        df2['CompetitionSince'] = df2.apply( lambda x: datetime.datetime(year=x['CompetitionOpenSinceYear'],
        month=x['CompetitionOpenSinceMonth'],day=1 ), axis=1 )
        
        df2['CompetitionTimeMonth'] = ( ( df2['Date'] - df2['CompetitionSince'] )/30).apply(
        lambda x: x.days ).astype( int )
        
        # promo since
        df2['PromoSince'] = df2['Promo2SinceYear'].astype( str ) + '-' + df2['Promo2SinceWeek'].astype( str )
        df2['PromoSince'] = df2['PromoSince'].apply( lambda x: datetime.datetime.strptime(
            x + '-1', '%Y-%W-%w' ) - datetime.timedelta( days=7 ) )
        df2['PromoTimeWeek'] = ( ( df2['Date'] - df2['PromoSince'] )/7 ).apply(
            lambda x: x.days ).astype( int )
        # assortment
        df2['Assortment'] = df2['Assortment'].apply( lambda x: 'basic' if x == 'a' else 
                                                                'extra' if x == 'b' else 'extended' )
        # state holiday
        df2['StateHoliday'] = df2['StateHoliday'].apply( 
            lambda x: 'PublicHoliday' if x == 'a' else 'EasterHoliday' if x == 'b' else 'Christmas' if x == 'c'
        else 'RegularDay' )
        
        # FILTRAGEM DE VARIÁVEIS
        ##  Filtragem das Linhas
        df2 = df2[(df2['Open'] != 0)]

        ## Selecao das Colunas
        cols_drop = ['Open', 'PromoInterval', 'MonthMap']
        df2 = df2.drop( cols_drop, axis=1 )
        
        return df2
    
    def data_preparation( self, df5 ):
        
        # Rescaling 
        # competition distance
        df5['CompetitionDistance'] = self.competition_distance_scaler.fit_transform( df5[['CompetitionDistance']].values )
    
        # competition time month
        df5['CompetitionTimeMonth'] = self.competition_time_month_scaler.fit_transform( df5[['CompetitionTimeMonth']].values )

        # promo time week
        df5['PromoTimeWeek'] = self.promo_time_week_scaler.fit_transform( df5[['PromoTimeWeek']].values )
        
        # year
        df5['Year'] = self.year_scaler.fit_transform( df5[['Year']].values )
        
        # StateHoliday - One Hot Encoding
        df5 = pd.get_dummies( df5, prefix=['StateHoliday'], columns=['StateHoliday'] )

        # StoreType - Label Encoding
        df5['StoreType'] = self.store_type_scaler.fit_transform( df5['StoreType'] )

        # Assortment - Ordinal Encoding
        assortment_dict = {'basic': 1, 'extra': 2, 'extended': 3}
        df5['Assortment'] = df5['Assortment'].map( assortment_dict )
        
        # transformações
        # day of week
        df5['DayOfWeekSin'] = df5['DayOfWeek'].apply( lambda x: np.sin( x * ( 2. * np.pi/7 ) ) )
        df5['day_of_week_cos'] = df5['DayOfWeek'].apply( lambda x: np.cos( x * ( 2. * np.pi/7 ) ) )
        # month
        df5['month_sin'] = df5['Month'].apply( lambda x: np.sin( x * ( 2. * np.pi/12 )) )
        df5['month_cos'] = df5['Month'].apply( lambda x: np.cos( x * ( 2. * np.pi/12 )) )
        # day
        df5['day_sin'] = df5['Day'].apply( lambda x: np.sin( x * ( 2. * np.pi/30 ) ) )
        df5['day_cos'] = df5['Day'].apply( lambda x: np.cos( x * ( 2. * np.pi/30 ) ) )
        # week of year
        df5['week_of_year_sin'] = df5['WeekOfYear'].apply( lambda x: np.sin( x * ( 2. * np.pi/52 ) ) )
        df5['week_of_year_cos'] = df5['WeekOfYear'].apply( lambda x: np.cos( x * ( 2. * np.pi/52 ) ))
        
        # variáveis finais
        cols_selected = [
            'Store',
            'Promo',
            'StoreType',
            'Assortment',
            'CompetitionDistance',
            'CompetitionOpenSinceMonth',
            'CompetitionOpenSinceYear',
            'Promo2',
            'Promo2SinceWeek',
            'Promo2SinceYear',
            'CompetitionTimeMonth',
            'PromoTimeWeek',
            'DayOfWeekSin',
            'day_of_week_cos',
            'month_sin',
            'month_cos',
            'day_sin',
            'day_cos',
            'week_of_year_sin',
            'week_of_year_cos']
        return df5[cols_selected]
    
    
    def get_prediction( self, model, original_data, test_data ):
        # prediction
        pred = model.predict( test_data )
        
        # join pred into the original data
        original_data['Prediction'] = np.expm1( pred )
        
        return original_data.to_json( orient='records', date_format='iso' )
