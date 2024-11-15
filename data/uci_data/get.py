from ucimlrepo import fetch_ucirepo 
import pandas as pd
  
# https://archive.ics.uci.edu/dataset/242/energy+efficiency
def get_data(id=242):
    energy_efficiency = fetch_ucirepo(id=id) 
    
    # data (as pandas dataframes) 
    X: pd.DataFrame = energy_efficiency.data.features 
    y: pd.DataFrame = energy_efficiency.data.targets
    return X.to_numpy(), y.to_numpy()