from __future__ import annotations
from typing import Dict, Tuple
from pathlib import Path
from datetime import datetime 
import copy
import numpy as np
import pandas as pd 
from rateslib import IRS as rlIRS, Curve as rlCurve, Solver as rlSolver, calendars as rlCalendars, Frequency as rlFrequency 
from curve_engine.swap_curve_solver import SwapCurveSolver
from curve_engine.swap_rate_pca_estimator import SwapRatePCAEstimator 

class CurveError(Exception):
  pass
 
class SwapCurveBuilder:
    """
    Analytical wrapper for a calibrated interest rate curve.

    This class provides derived curve analytics and risk transformation tools
    based on a calibrated curve obtained from `SwapCurveSolver`.

    It exposes commonly used representations of the yield curve and supports
    transformations between different risk spaces (e.g. spot rates, forward rates,
    and PCA factors).

    Parameters
    ----------
    curve_id : str
        Identifier of the curve.

    curve_solver : SwapCurveSolver
        Solver instance containing calibrated curve and instruments.

    Attributes
    ----------
    curve : rateslib Curve
        Calibrated discount curve.

    solver : rateslib Solver
        Solver used for pricing and sensitivity calculations.

    nodes : pd.DataFrame
        Curve node information (tenors, discount factors).

    zero_rates : pd.DataFrame
        Spot zero rates derived from discount factors.

    par_rates : pd.DataFrame
        Par swap rates implied by the curve.

    fwd_rates : pd.DataFrame
        Forward rates derived from the curve.

    par_fwd_jacobian : pd.DataFrame
        Jacobian matrix mapping par rate sensitivities to forward rate sensitivities.

    par_pc_jacobian : pd.DataFrame
        Jacobian matrix mapping par rate sensitivities to PCA factor sensitivities.

    pca_loadings : pd.DataFrame
        PCA factor loadings associated with the curve.

    Methods
    -------
    calc_zero_rates()
        Computes spot zero rates from discount factors.

    calc_par_rates()
        Computes par swap rates implied by the curve.

    calc_forward_rates()
        Computes forward rates between tenors.

    calc_par_fwd_jacobian()
        Builds Jacobian transforming par rate sensitivities into forward rate sensitivities.

    calc_par_pc_jacobian()
        Builds Jacobian transforming par rate sensitivities into PCA factor sensitivities.

    Notes
    -----
    - This class separates curve calibration from analytics, enabling reuse of the
      same curve across different risk representations.
    - Jacobians are central for transforming sensitivities between risk spaces.
    - Designed to support portfolio-level risk aggregation and hedging analysis.
    """
    
    @classmethod
    def from_solver(    
            cls,
            curve_id : str,  
            curve_solver : SwapCurveSolver
        ) -> SwapCurveBuilder:
        
        return cls( curve_id = curve_id, 
                    curve_date = curve_solver.curve_date,
                    curve = curve_solver.curve,                    
                    curve_nodes = curve_solver.curve_nodes,
                    solver = curve_solver.solver, 
                    instruments = curve_solver.instruments,                   
                    solver_id = curve_solver.solver_id, 
                    solver_currency = curve_solver.config.currency, 
                    solver_calendar = curve_solver.config.calendar,                     
                    solver_leg_convention = curve_solver.config.leg_convention, 
                    solver_float_leg_convention = curve_solver.config.float_leg_convention, 
                    solver_leg_frequency = curve_solver.config.leg_frequency, 
                    solver_float_leg_frequency = curve_solver.config.float_leg_frequency                                                                                 
                  )
        
    def __init__( 
            self,
            curve_id : str, 
            curve_date : datetime,
            curve : rlCurve,   
            curve_nodes : Dict[str, Tuple],
            instruments : Dict[str, rlIRS],             
            solver : rlSolver,
            solver_id : str,                         
            solver_currency : str = None, 
            solver_calendar : str = None, 
            solver_leg_convention : str = None, 
            solver_float_leg_convention : str = None,
            solver_leg_frequency : str = None,
            solver_float_leg_frequency : str = None
        ) -> None:


        self.curve_id = curve_id                 
        self.curve_date = curve_date                        
        self.curve = curve 
        self.curve_nodes  = curve_nodes
        self.solver = solver                
        self.solver_id = solver_id                
        self.solver_currency = solver_currency                
        self.solver_calendar = solver_calendar                
        self.solver_leg_convention = solver_leg_convention                
        self.solver_float_leg_convention = solver_float_leg_convention                
        self.solver_leg_frequency = solver_leg_frequency                
        self.solver_float_leg_frequency = solver_float_leg_frequency                

        self._is_fwd_jacobian_built = False 
        self._is_pc_jacobian_built = False 
        
        self._instruments       = instruments          
        self._node_dates        = {node_tenor : date_tuple[0] for node_tenor, date_tuple in curve_nodes.items()}
        self._node_years        = {node_tenor : date_tuple[1] for node_tenor, date_tuple in curve_nodes.items()}
        self._node_disc_factors = self._set_disc_factors()        
        self._spot_zero_rates = self._calc_spot_zero_rates()
        self._spot_par_rates  = self._calc_spot_par_rates()
        
    def _set_disc_factors( 
            self
        ) -> Dict[str, float]:
                    
        return { node_tenor : self.curve.nodes.nodes[node_date].real 
                        for node_tenor, node_date in self._node_dates.items() 
               }   
           
    def _calc_spot_zero_rates( 
            self
        ) -> Dict[str, float]:
        
        for node_years in self._node_years.values():
            if node_years == 0:
                raise CurveError("Zero rate calc. for a curve node set to curve date with 0 years - division by zero!")
        
        return { node_tenor : float( -1/node_years * np.log( self._node_disc_factors[node_tenor] ) )
                        for node_tenor, node_years in self._node_years.items() 
               }           
        
    def _calc_spot_par_rates(
            self
        ) -> Dict[str, float]:
        
        return { tenor : instrument.rate( curves = [self.curve] ).real / 100
                   for tenor, instrument in self._instruments.items()
               }

    def calc_par_fwd_jacobian(
            self
        ) -> None:
        
        nodes = list(self._node_years.keys())

        fwd_instrument_labels = (
            [f"0Y{nodes[0]}"] + 
            [f"{start}{int(end[:-1]) - int(start[:-1])}Y"
         for start, end in zip(nodes, nodes[1:])] )

        fwd_instrument_tenors = (
            [nodes[0]] + 
            [f"{int(end[:-1]) - int(start[:-1])}Y"
         for start, end in zip(nodes, nodes[1:])] )

        fwd_instrument_dates = (
            [self.curve_date] + 
            [self._node_dates[_] for _ in nodes[:-1]] )

        try: fwd_instruments = ( [ 
                 rlIRS(   effective = fwd_instrument_dates[idx], 
                        termination = fwd_instrument_tenors[idx], 
                           calendar = rlCalendars.get(self.solver_calendar), 
                           currency = self.solver_currency, 
                             curves = self.solver_id, 
                          frequency = rlFrequency.Months(self.solver_leg_frequency, None), 
                     leg2_frequency = rlFrequency.Months(self.solver_float_leg_frequency, None), 
                         convention = self.solver_leg_convention, 
                    leg2_convention = self.solver_float_leg_convention  
                      ) 
                    for idx in range(len(fwd_instrument_labels)) ] )
        except Exception as e: 
                raise CurveError(f"error creating instrument for forward par rates: {e}")     
    
        fwd_instrument_rates = (
            [ float(_.rate(solver = self.solver)) for _ in fwd_instruments ] )

        fwd_curve = copy.deepcopy(self.curve)

        fwd_solver = rlSolver(
                      curves = [fwd_curve],
                  instruments = fwd_instruments,
                            s = fwd_instrument_rates,
                           id = self.solver_id,
            instrument_labels = fwd_instrument_labels )

        par_fwd_jacobian_df = self.solver.jacobian(fwd_solver).droplevel(0, axis = 0)
        par_fwd_jacobian_df = par_fwd_jacobian_df.droplevel(0, axis = 1)
        par_fwd_jacobian_df.columns.name = 'Node'
        
        self._par_fwd_jacobian     = par_fwd_jacobian_df
        self._fwd_instruments      = {tenor : inst for tenor, inst in zip(fwd_instrument_labels, fwd_instruments)}
        self._fwd_instrument_rates = {tenor : inst for tenor, inst in zip(fwd_instrument_labels, fwd_instrument_rates)}
        
        self._is_fwd_jacobian_built = True         
        
    def calc_par_pc_jacobian(
            self, 
            pca : SwapRatePCAEstimator
        ) -> None:
       
        self.pca_eigvals  = pca.pca_eigvals
        self.pca_eigvecs  = pca.pca_eigvecs 
        self.pca_loadings = pca.pca_loadings
        
        self._is_pc_jacobian_built = True 
               
    @property
    def nodes( 
        self
    ) -> pd.DataFrame:
        
        df =  pd.DataFrame( {
                "Node": [str.upper() for str in self._node_dates.keys()],
                "Date": pd.to_datetime( list( self._node_dates.values() ) ),
                "Years": list( self._node_years.values() )
               } )      
        
        df["Years"] = df["Years"].astype(float)  
        
        return df         

    @property
    def disc_factors(
        self
    ) -> pd.DataFrame:
        
        df =  pd.DataFrame( {
                "Node": [str.upper() for str in self._node_dates.keys()],
                "DiscFactor": list( self._node_disc_factors.values() )
               } )      
        
        df["DiscFactor"] = df["DiscFactor"].astype(float)  
        
        return df   

    @property
    def spot_zero_rates( 
        self
    ) -> pd.DataFrame:
        
        df =  pd.DataFrame( {
                "Node": [str.upper() for str in self._node_dates.keys()],
                "SpotZeroRate": list( self._spot_zero_rates.values() )
               } )      
        
        df = ( df.astype( {"SpotZeroRate" : float} )
                 .assign( SpotZeroRate = lambda df_ : df_.SpotZeroRate * 100 )
             )
                
        return df           
        
    @property
    def spot_par_rates( 
        self
    ) -> pd.DataFrame:
        
        df =  pd.DataFrame( {
                "Node": [str.upper() for str in self._node_dates.keys()],
                "SpotParRate": list( self._spot_par_rates.values() )
               } )      
        
        df = ( df.astype( {"SpotParRate" : float} )
                 .assign( SpotParRate = lambda df_ : df_.SpotParRate * 100 )
             )
        
        return df            
           
    @property
    def fwd_par_rates( 
        self
    ) -> pd.DataFrame:
        
        if self._is_fwd_jacobian_built:
            
            df =  pd.DataFrame( {
                    "Node": list( self._fwd_instrument_rates.keys() ),
                    "FwdParRate": list( self._fwd_instrument_rates.values() )
                   } )      
        
            df["FwdParRate"] = df["FwdParRate"].astype(float)          
            return df
        else: 
             raise CurveError(" par >> fwd Jacobian has not been calculated, see method .calc_par_fwd_jacobian()")   
              
    @property
    def par_fwd_jacobian( 
        self
    ) -> pd.DataFrame:
        
        if self._is_fwd_jacobian_built:
            
            return self._par_fwd_jacobian
        else: 
             raise CurveError(" par >> fwd Jacobian has not been calculated, see method .calc_par_fwd_jacobian()")                 
               
    @property
    def par_pc_jacobian( 
        self
    ) -> pd.DataFrame:
        
        if self._is_pc_jacobian_built:
            
            pc_clmns = ['PC1', 'PC2', 'PC3', 'PC4']
            nodes_clmns = self.pca_eigvecs['Node'].values.tolist()
            
            par_pc_jacobian_df = (
                 self.pca_eigvecs[pc_clmns]
                    .pivot_table(columns = nodes_clmns, values = pc_clmns)
                     [nodes_clmns]
                             )
            par_pc_jacobian_df = par_pc_jacobian_df.rename_axis('Component', axis=1)
            return par_pc_jacobian_df        
        else: 
             raise CurveError(" par >> PC Jacobian has not been calculated, see method .calc_par_pc_jacobian()")