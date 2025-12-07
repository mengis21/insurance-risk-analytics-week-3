import pandas as pd
from datetime import datetime

def main():
    df = pd.DataFrame({
        'TotalPremium':[1000,1200,900,1100],
        'TotalClaims':[200,300,100,500],
        'CalculatedPremiumPerTerm':[100,120,90,110],
        'CustomValueEstimate':[50000,60000,45000,55000],
        'Province':['Gauteng','Western Cape','Gauteng','KwaZulu-Natal'],
        'VehicleType':['Sedan','SUV','Hatchback','SUV'],
        'Gender':['Male','Female','Female','Male'],
        'TransactionMonth':[datetime(2014,2,1),datetime(2014,3,1),datetime(2014,4,1),datetime(2014,5,1)]
    })
    df.to_csv('SM/data/insurance.csv', index=False)

if __name__ == '__main__':
    main()
