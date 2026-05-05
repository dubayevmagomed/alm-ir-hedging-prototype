from __future__ import annotations
from typing import Dict
from pathlib import Path
import numpy as np
import pandas as pd 

class SwapRatePCAEstimator:
    """
    Principal Component Analysis (PCA) estimator for interest rate curves.

    This class performs PCA on historical swap rate shifts to identify the main
    drivers of yield curve movements (e.g. level, slope, curvature).

    The PCA is based on the sample covariance matrix of rate shifts and uses
    eigen-decomposition to extract orthogonal risk factors.

    Parameters
    ----------
    ts_df_dict : Dict[str, pd.DataFrame]
        Dictionary containing:
        - "rates": historical rate levels
        - "shifts": rate differences
        - "cshifts": centered rates

    Attributes
    ----------
    cov_mat : pd.DataFrame
        Sample covariance matrix of rate shifts.

    corr_mat : pd.DataFrame
        Correlation matrix of rate shifts.

    pca_eigvals : pd.DataFrame
        Eigenvalues and explained variance ratios.

    pca_eigvecs : pd.DataFrame
        Eigenvectors representing principal components.

    pca_loadings : pd.DataFrame
        Factor loadings (scaled eigenvectors).

    Notes
    -----
    PCA decomposition:
        Σ v_i = λ_i v_i

    Loadings:
        L_i = v_i * sqrt(λ_i)

    These loadings represent factor sensitivities in original units.
    """
    @classmethod 
    def from_csv(
       cls, 
       swap_ts_csv_path : Path | None = None 
      ) -> SwapRatePCAEstimator: 

      # default to project data, if path to csv file with time series for eur swap rates is not provided
      if swap_ts_csv_path is None: 
          
          project_root = Path(__file__).resolve().parent.parent.parent
          csv_path = project_root / "curve_data" / "eur_swap_ts.csv"
      
      else: 
          
          csv_path = swap_ts_csv_path 
               
      rates_df = (  
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
      
      shifts_df = rates_df.diff().dropna()
      cshifts_df = rates_df - rates_df.mean()
      
      ts_df_dict = { 'rates'   : rates_df, 
                     'shifts'  : shifts_df, 
                     'cshifts' : cshifts_df
                   } 
      
      return cls( ts_df_dict = ts_df_dict )   
        
    def __init__( 
       self, 
       ts_df_dict : Dict[str, pd.DataFrame]
      ) -> None: 

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
      
      self._estimate_pca()
    
    def _estimate_pca(
        self 
      ) -> None:

      _eigvals, _eigvecs = np.linalg.eigh(self.cov_mat.iloc[:,1:].values) # numpy saves eigvals from smallest to largest     
      idx = np.argsort(_eigvals)[::-1] # index the eigval elements from largest to smallest
      eigvals = _eigvals[idx] # reorder eigenvalues
      eigval_ratios = eigvals / eigvals.sum()
      eigvals_df = pd.DataFrame( eigvals.reshape(1, len(eigvals) ), index = ['eigenvalue'] )
      eigval_ratios_df = pd.DataFrame( eigval_ratios.reshape(1, len(eigval_ratios)) * 100, index = ['% of total variance'] )
       
      self.pca_eigvals    = ( pd.concat([  eigvals_df
                                         , eigval_ratios_df
                                        ], axis = 0 )
                             .rename( columns = lambda i : f"PC{i+1}" )
                            )   

      eigvecs  = _eigvecs[:, idx] # reorder correspoinding eigenvectors - slice the column elements in accordance to index of eigenvalues          
      loadings = eigvecs * np.sqrt(eigvals) # scale with sqrt( eigval ) - i.e. std. deviation of each PC factor 
     
      self.pca_eigvecs    = ( pd.concat([  self.cov_mat[['Node']] 
                                         , pd.DataFrame( eigvecs ).rename( columns = lambda i : f"PC{i+1}" )
                                        ], axis = 1 )
                            )
      
      self.pca_loadings   = ( pd.concat([  self.cov_mat[['Node']] 
                                          , pd.DataFrame( loadings ).rename( columns = lambda i : f"PC{i+1}" )
                                         ], axis = 1 )
                            )