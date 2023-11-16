import pandas as pd
import numpy as np
import pgeocode
from sklearn.impute import KNNImputer

def read_data() -> pd.DataFrame:
    """
    
    """
    train_set = pd.read_excel(
        io='data/ecoshare_sales_v3.xlsx',
        sheet_name='Data',
        header=0,
        true_values=['Y'],
        false_values=['N']
    )

    test_set = pd.read_csv(
        filepath_or_buffer='data/ecoshare_sales_test.csv',
        header=0,
        true_values=['Y'],
        false_values=['N'],
        parse_dates=[0]
    )

    train_set['set'] = 'train'
    test_set['set'] = 'test'
    test_set['accept'] = np.nan

    return pd.concat([train_set, test_set])

def preprocessing(data: pd.DataFrame) -> pd.DataFrame:
    """

    """
    data['order_day'] = pd.to_datetime(data['order_day'])
    data.sort_values(by='order_day', ascending=True, inplace=True)
    data['day_of_year'] = data['order_day'].dt.day_of_year
    data['day_of_week'] = data['order_day'].dt.dayofweek

    data['term_length'] = pd.to_numeric(data['term_length'], errors='coerce')
    data['tos_flg'].fillna(False, inplace=True)

    data['load_profile_PV'] = data['load_profile'].str.endswith(pat='PV', na=False)
    data['load_profile'] = data['load_profile'].str[:2]

    return data

def add_household_income(data: pd.DataFrame) -> pd.DataFrame:
    """
    
    """
    county_income = pd.read_html(io='https://www.texascounties.net/statistics/householdincome2020.htm')[1]
    county_income.drop(
        columns=[
            'Rank',
            'Population 2020'
        ],
        inplace=True
    )
    county_income.rename(
        columns={
            'Name': 'county',
            'Median Household Income, 2020 ($)': 'county_median_household_income'
        },
        inplace=True
    )
    county_income['county'] = county_income['county'].str.upper()
    data = data.merge(county_income, on='county', how='left')

    return data

def geocode(data: pd.DataFrame) -> pd.DataFrame:
    """
    
    """
    nomi = pgeocode.Nominatim('us')
    data['zipcode'] = data['zipcode'].astype('string').str.slice(0, 5)
    latlong = nomi.query_postal_code(data['zipcode'].to_numpy())[['latitude', 'longitude']]
    data = data.merge(latlong, how='outer', left_index=True, right_index=True)

    return data

def dummy_variables(data: pd.DataFrame) -> pd.DataFrame:
    """
    
    """
    cols = [
    'dwelling_type_cd',
    'product_type_cd',
    'tdsp',
    'segment',
    'pool',
    'risk_level',
    'load_profile'
    ]
    data = pd.get_dummies(data, columns=cols, dummy_na=True, drop_first=True)

    data['sap_productname'] = data['sap_productname'].fillna('').str.lower()
    sap_keywords = [
        'easy',
        'flex'
        'secure',
        'weekends',
        'conservation',
        'save',
        'simple',
        'solarsparc',
        'pollution free',
        'wind',
        'business',
        'apartment',
        'smart',
        'discount',
        'solar'
    ]
    for kw in sap_keywords:
        data[f'sap_productname_{kw}'] = data['sap_productname'].str.contains(kw)

    return data

def add_customer_features(data: pd.DataFrame) -> pd.DataFrame:
    """
    
    """
    # num of meters tied to each customer
    customer_num_meters = data[['customer_id', 'meter_id']].groupby(by='customer_id', as_index=False).count().rename(columns={'meter_id': 'customer_num_meters'})
    data = data.merge(customer_num_meters, on='customer_id', how='left')

    # determine if an eligible customer/meter was previously enrolled in EcoShare
    prev_enrolled = data.sort_values(by=['customer_id', 'meter_id', 'order_day']).groupby(by=['customer_id', 'meter_id'])[['accept']].shift(1).fillna(0).rename(columns={'accept': 'prev_enrolled'})
    prev_enrolled = data.merge(prev_enrolled, left_index=True, right_index=True).sort_values(by=['customer_id', 'meter_id', 'order_day']).groupby(by=['customer_id', 'meter_id'])[['prev_enrolled']].cumsum().astype(bool)
    data = data.merge(prev_enrolled, left_index=True, right_index=True)

    # determine count of previous attempts to upsell EcoShare
    prev_attempts = data.sort_values(by=['customer_id', 'meter_id', 'order_day']).groupby(by=['customer_id', 'meter_id'])[['accept']].cumcount().rename('prev_attempts')
    data = data.merge(prev_attempts, left_index=True, right_index=True)

    # call volume on a given order day
    calls_by_order_day = data[['order_day']].groupby(by='order_day', as_index=False).size().rename(columns={'size': 'total_calls_on_day'})
    data = data.merge(calls_by_order_day, on='order_day')

    return data

def drop_columns(data: pd.DataFrame) -> pd.DataFrame:
    """
    
    """
    cols_to_drop = [
        'order_day', # parsed to day_of_year
        'city', # geocoded
        'county', # geocoded
        'dma', # geocoded
        'zipcode', # geocoded
        'customer_id', # internal identifier
        'meter_id', # internal identifier

        # sparse cols
        'deposit_onhand_amt',
        'home_value',
        
        # coded by keywords
        'sap_productname'
        
    ]

    return data.drop(columns=cols_to_drop)

def prep_data() -> pd.DataFrame:
    """
    
    """
    data = read_data()
    data = preprocessing(data)
    data = add_household_income(data)
    data = geocode(data)
    data = dummy_variables(data)
    data = add_customer_features(data)
    data = drop_columns(data)

    train = data[data['set'] == 'train'].drop(columns='set')
    test = data[data['set'] == 'test'].drop(columns=['set', 'accept'])

    train['accept'] = train['accept'].astype(int)
    return train, test

def create_imputer(X_train: pd.DataFrame) -> pd.DataFrame:
    """
    
    """
    imputer = KNNImputer(
        n_neighbors=3,
        weights='distance'
    )
    imputer.fit(X_train)
    imputer.set_output(transform='pandas')
    return imputer
