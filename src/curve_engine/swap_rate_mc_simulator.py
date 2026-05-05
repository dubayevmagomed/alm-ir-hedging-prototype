from __future__ import annotations
from typing import Dict
from pathlib import Path
import numpy as np
import pandas as pd 

class SwapRateMCSimulator:
    """
    Monte Carlo simulator for interest rate curve scenarios based on historical covariance.

    This class generates stochastic scenarios for par/swap rates using a multivariate
    normal framework calibrated to historical rate shifts. The covariance matrix is
    estimated from historical data and used in conjunction with Cholesky decomposition
    to produce correlated rate shocks.

    The simulated rate scenarios are used for curve calibration,
    portfolio valuation, and risk analysis (e.g. VaR, stress testing).

    Parameters
    ----------
    ts_df_dict : Dict[str, pd.DataFrame]
        Dictionary containing historical time series:
        - "rates": level time series of swap rates
        - "shifts": first differences of rates
        - "cshifts": centered rate series (demeaned)

    n_scenarios : int
        Number of Monte Carlo scenarios to generate.

    t_scale_factor : float, optional
        Time scaling factor applied to shocks. Scenarios are scaled by sqrt(T).
        Default is 1.0 (1-day horizon).

    rng_seed : int, optional
        Seed for random number generator to ensure reproducibility.

    Attributes
    ----------
    rates : pd.DataFrame
        Historical rate levels.

    shifts : pd.DataFrame
        Historical rate changes.

    cov_mat : pd.DataFrame
        Sample covariance matrix of rate shifts.

    rate_scenarios : pd.DataFrame
        Simulated rate levels across scenarios.

    shift_scenarios : pd.DataFrame
        Simulated rate shocks across scenarios.

    Notes
    -----
    - Assumes multivariate normality of rate shifts.
    - Correlation structure is preserved via Cholesky decomposition.
    - Time scaling assumes Brownian motion (sqrt(T) scaling).
    """
    
    @classmethod 
    def from_csv(
       cls, 
       n_scenarios : float = 1000, 
       t_scale_factor : float = 1.0, 
       rng_seed : int = 1, 
       swap_ts_csv_path : Path | None = None 
      ) -> SwapRateMCSimulator: 
   
      # default to project data, if path to csv file with time series for eur swap rates is not provided
      if swap_ts_csv_path is None: 
          
          project_root = Path(__file__).resolve().parent.parent.parent
          csv_path = project_root / "curve_data" / "eur_swap_ts.csv"
      
      else: 
          
          csv_path = swap_ts_csv_path 
   
      rates = (  
       pd.read_csv(csv_path)
      .astype({ 'Tenor' : 'Int64', 
                'Rate'  : 'float64' })
      .assign( Date = lambda df_ : pd.to_datetime(df_.Date))
      .pivot( index   = 'Date', 
              columns = 'Tenor', 
              values  = 'Rate') 
      .sort_index(axis=1)
      .rename(columns = lambda clmn : str(clmn) + 'Y')
      .rename_axis(None, axis=0)
      .rename_axis(None, axis=1) )
      
      shifts = rates.diff().dropna()
      cshifts = rates - rates.mean()
      
      ts_df_dict = { 'rates'   : rates, 
                     'shifts'  : shifts, 
                     'cshifts' : cshifts
                   } 
      
      return cls( ts_df_dict = ts_df_dict, 
                  n_scenarios = n_scenarios, 
                  t_scale_factor = t_scale_factor, 
                  rng_seed = rng_seed 
                ) 
      
    def __init__( 
       self, 
       ts_df_dict : Dict[str, pd.DataFrame],               
       n_scenarios : int, 
       t_scale_factor : float = 1.0, 
       rng_seed : int = 1 
      ) -> None :

      self.ltst_rates = ts_df_dict['rates'].iloc[[-1],:]     
      self.rates = ts_df_dict['rates']         
      self.shifts = ts_df_dict['shifts']
      self.cshifts = ts_df_dict['cshifts']      

      
      self.cov_mat = (
         self.shifts
        .cov()
        .reset_index()
        .rename(columns = {'index' : 'Node'}) )

      self.corr_mat = (
         self.shifts
        .corr()
        .reset_index()
        .rename(columns = {'index' : 'Node'}) )      
      
      # local, instance-specific rn generator
      self.rng = np.random.default_rng(rng_seed)
      
      # create scenarios and save in a dict attr. which can be passed to a curve bootsrapper/calibrator
      self.n_scenarios = n_scenarios 
      self.t_scale_factor = t_scale_factor       
      self._mc_scenarios()
              
    def _mc_scenarios(
       self
     ) -> None : 
     
     df_clmns = self.cov_mat.iloc[:,1:].columns   
     cov_arr = self.cov_mat[df_clmns].values

     # cholesky decomposition lower triangular matrix, L
     L = np.linalg.cholesky(cov_arr) 
  
     # generate n-scenarios x n-tenors matrix of std. normal draws
     n_tenors = cov_arr.shape[0]
     Z = self.rng.normal( size = (self.n_scenarios, n_tenors) )   

     # generate n-scenarios x n-tenors matrix of simulated (1d) rate shifts - (T) Time scaled by sqrt(T)
     S_shifts = np.sqrt(self.t_scale_factor) * Z @ L.T   

     # a df with n rate shift scenarios; df with n rate level scenarios from adding shifts to last rates observed in data. 
     S_rates = S_shifts + self.ltst_rates[df_clmns].values
     
     shift_scens = pd.DataFrame( data = S_shifts, columns = df_clmns )
     rate_scens = pd.DataFrame( data = S_rates, columns = df_clmns )

     shift_scens.insert(0, "Scenario", shift_scens.index + 1)
     rate_scens.insert(0, "Scenario", rate_scens.index + 1)
     
     self.shift_scenarios = shift_scens 
     self.rate_scenarios = rate_scens