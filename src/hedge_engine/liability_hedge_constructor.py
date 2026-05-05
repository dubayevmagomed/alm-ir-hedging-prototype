from __future__ import annotations
from typing import Dict, List
import copy 
from pathlib import Path
from datetime import datetime 
import numpy as np
import pandas as pd 
from curve_engine.swap_curve_solver import SwapCurveSolver
from curve_engine.swap_curve_builder import SwapCurveBuilder
from curve_engine.swap_curve_scenario_generator import SwapCurveScenarioGenerator
from position_engine.liability_position import LiabilityPosition
from position_engine.euribor6m_irs_position import Euribor6mIRSPosition
from position_engine.liability_portfolio import LiabilityPortfolio

class HedgeConstructorError(Exception):
    pass 

class LiabilityHedgeConstructor:
    """
    Constructs hedging strategies for liability portfolios using IRS instruments.

    This class computes hedge weights by matching or minimizing interest rate
    exposure using different methodologies.

    Parameters
    ----------
    hedge_dt : datetime
        Hedge construction date.

    liability_position : LiabilityPosition
        Target liability exposure.

    liability_curve : SwapCurveBuilder
        Curve used for valuation and sensitivities.

    liquid_irs_tenors : List[str]
        Set of liquid instruments available for proxy hedging.

    swap_rate_cov_mat : pd.DataFrame, optional
        Covariance matrix for construction of liquid proxy hedge.

    Methods
    -------
    calc_benchmark_hedge()
        Exact PV01 neutralization.

    calc_liquid_proxy_hedge()
        Variance-minimizing proxy hedge using covariance weighting, and liquid instruments only (2Y,5Y,10Y,30Y)

    Notes
    -----
    Hedge formulations:

    Benchmark:
        Ax = -b

    Variance-optimal Liquid Proxy:
        min xᵀ Aᵀ Σ A x + 2bᵀ Σ A x

    where:
        A = hedge instrument sensitivities
        b = liability sensitivities
        Σ = Covariance matrix for swap rates
        x = Notional weight for the hedge instruments
    """
    def __init__(
       self, 
       hedge_dt : datetime, 
       liability_position : LiabilityPosition, 
       liability_curve : SwapCurveBuilder, 
       liquid_irs_tenors : List[str], 
       swap_rate_cov_mat : pd.DataFrame 
       ) -> None :
        
        self.hedge_dt = hedge_dt
        self.liability_position = liability_position
        self.liability_curve = liability_curve
        self.liquid_irs_tenors = liquid_irs_tenors
        self.swap_rate_cov_mat = swap_rate_cov_mat
        self.curve_node_tenors = liability_curve.nodes['Node'].to_list()
                
        self._benchmark_hedge_instruments = {
             tenor : 
             Euribor6mIRSPosition(
                  pos_id = f"EUR6M {tenor}", 
                  pos_grp = 'Hedge', 
                  effective_dt = self.hedge_dt, 
                  fixed_rate = 0, 
                  notional = 1e6, 
                  termination = tenor, 
                  is_receiver = True, 
                  is_par_swap = True, 
                  position_curve_builder = self.liability_curve)
             for tenor in self.curve_node_tenors }
        
        self._liquid_proxy_hedge_instruments = {
             tenor : copy.deepcopy(self._benchmark_hedge_instruments[tenor])
             for tenor in self.liquid_irs_tenors }
        
        self._pv01_clmns_idx = (
            self.liability_position
                .spot_delta( calc_curve_builder = self.liability_curve )
                .columns.to_list()
                .index('Curve') + 1 )
        
        self._lby_pv01_vec = (
            self.liability_position
                .spot_delta()
                .iloc[ : , self._pv01_clmns_idx : ]
                .values
                .reshape(-1,1) )
        
        benchmark_irs_pv01_deltas = [
           irs.spot_delta(calc_curve_builder = self.liability_curve)
              .iloc[:, self._pv01_clmns_idx :]
              .T
              .values 
           for irs in self._benchmark_hedge_instruments.values() ]

        liquid_irs_pv01_deltas = [
           irs.spot_delta(calc_curve_builder = self.liability_curve)
              .iloc[:, self._pv01_clmns_idx :]
              .T
              .values 
           for irs in self._liquid_proxy_hedge_instruments.values() ]
        
        self._benchmark_irs_pv01_mat = np.hstack(benchmark_irs_pv01_deltas)    
        self._liquid_irs_pv01_mat = np.hstack(liquid_irs_pv01_deltas)

        self.calc_benchmark_hedge()
        self.calc_liquid_proxy_hedge(self.swap_rate_cov_mat)
   
    def calc_benchmark_hedge(
        self, 
       ) -> None : 
        
        hedge_weights = np.linalg.solve(self._benchmark_irs_pv01_mat, -self._lby_pv01_vec)
        hedge_df = (
            pd.DataFrame( data = hedge_weights,
                          index = self.curve_node_tenors,
                          columns = ['Hedge Weight'] 
                        )
              .assign( **{'Hedge Type' : 'Benchmark IRS Hedge'},  
                       **{'Notional' : lambda df: df['Hedge Weight'] * 1e6}
                     ) ) 
        
        self.benchmark_hedge_positions = {}     
        for irs_tenor, irs in self._benchmark_hedge_instruments.items():
            pos_grp = 'Benchmark IRS Hedge'
            hedge_notional = hedge_df.loc[irs_tenor, 'Notional']
            fixed_rate = irs.fixed_rate
            if hedge_notional < 0:
                is_receiver = False 
                pos_id = f"EURIBOR6M {irs_tenor} ATM Payer @ {round(fixed_rate,3)}%"      
            else: 
                is_receiver = True 
                pos_id = f"EURIBOR6M {irs_tenor} ATM Receiver @ {round(fixed_rate,3)}%"         
            irs_position = Euribor6mIRSPosition(
                pos_id = pos_id, 
                pos_grp = pos_grp, 
                notional = hedge_notional, 
                termination = irs_tenor, 
                is_receiver = is_receiver, 
                is_par_swap = True, 
                fixed_rate = fixed_rate, 
                effective_dt = self.hedge_dt,
                position_curve_builder = self.liability_curve)  
            self.benchmark_hedge_positions[irs_tenor] = irs_position 
                
        hedge_df['Hedge Position ID'] = [
          pos.pos_id for pos in (self.benchmark_hedge_positions[irs_tenor] for irs_tenor in self.curve_node_tenors)
        ]
        hedge_df.index.name = 'Hedge Tenor'
        hedge_df.reset_index(inplace = True)                
        self._benchmark_hedge_df = hedge_df.copy()
  
    def calc_liquid_proxy_hedge(
        self, 
        swap_rate_cov_mat : pd.DataFrame
       ) -> None : 

        cov_mat = swap_rate_cov_mat.iloc[:,1:].values        
        hedge_weights = np.linalg.solve(  
            self._liquid_irs_pv01_mat.T @ cov_mat @ self._liquid_irs_pv01_mat,
          - self._liquid_irs_pv01_mat.T @ cov_mat @ self._lby_pv01_vec )
        
        hedge_df = (
            pd.DataFrame( data = hedge_weights,
                          index = self.liquid_irs_tenors,
                          columns = ['Hedge Weight'] 
                        )
              .assign( **{'Hedge Type' : 'Liquid Proxy Hedge'},  
                       **{'Notional' : lambda df: df['Hedge Weight'] * 1e6}
                     ) ) 
        
        self.liquid_proxy_hedge_positions = {}                
        for irs_tenor, irs in self._liquid_proxy_hedge_instruments.items():
            pos_grp = 'Liquid Proxy Hedge'           
            hedge_notional = hedge_df.loc[irs_tenor, 'Notional']
            fixed_rate = irs.fixed_rate
            
            if hedge_notional < 0:
               is_receiver = False 
               pos_id = f"EURIBOR6M {irs_tenor} ATM Payer @ {round(fixed_rate,3)}%"
            else: 
               is_receiver = True 
               pos_id = f"EURIBOR6M {irs_tenor} ATM Receiver @ {round(fixed_rate,3)}%"
               
            irs_position = Euribor6mIRSPosition(
            pos_id = pos_id, 
            pos_grp = pos_grp, 
            notional = hedge_notional, 
            termination = irs_tenor, 
            is_receiver = is_receiver, 
            is_par_swap = True, 
            fixed_rate = fixed_rate, 
            effective_dt = self.hedge_dt,                   
            position_curve_builder = self.liability_curve) 
            self.liquid_proxy_hedge_positions[irs_tenor] = irs_position 
            
        hedge_df['Hedge Position ID'] = [
          pos.pos_id for pos in (self.liquid_proxy_hedge_positions[irs_tenor] for irs_tenor in self.liquid_irs_tenors)
        ]
        hedge_df.index.name = 'Hedge Tenor'
        hedge_df.reset_index(inplace = True)        
        self._liquid_proxy_hedge_df = hedge_df.copy()
    
    def create_hedged_portfolio(
        self, 
        prtfl_id : str, 
        effective_dt : datetime, 
        liability_position : LiabilityPosition, 
        hedge_positions : Dict[str, Euribor6mIRSPosition]
        ) -> LiabilityPortfolio : 
        
        positions = [liability_position]
        for hedge_position in hedge_positions.values():
            positions.append(hedge_position)
        
        return LiabilityPortfolio(
                prtfl_id = prtfl_id, 
                positions = positions, 
                effective_dt = effective_dt)

    @property
    def full_hedge_instruments(
        self
        ) -> pd.DataFrame: 
        
        dfs = []
        
        for tenor, inst in self._benchmark_hedge_instruments.items():
            
            df = pd.concat(
                [ pd.DataFrame(
                    { 'IRS Hedge Instrument' : f"EURIBOR6M {tenor} ATM Receiver @ {round(inst.fixed_rate,3)}%" ,
                      'IRS Tenor' : tenor, 
                      'Notional' : inst.notional}, 
                        index = [0] 
                    ), 
                    inst.spot_delta().loc[:,'Sensitivity':]
                ],  
                axis = 1
            )    
            dfs.append(df)

        return pd.concat(dfs).reset_index(drop = True)
            
    @property
    def liquid_hedge_instruments(
        self
        ) -> pd.DataFrame: 
        
        df = self.full_hedge_instruments
        
        return df[df['IRS Tenor'].isin(self.liquid_irs_tenors)].reset_index(drop = True)    
            
    @property 
    def benchmark_hedge(
        self     
    ) -> pd.DataFrame : 
        
        clmn_list = ['Hedge Type', 'Hedge Tenor', 'Hedge Weight', 'Hedge Position ID', 'Notional']
        return self._benchmark_hedge_df[clmn_list]
    
    @property 
    def liquid_proxy_hedge(
        self     
    ) -> pd.DataFrame : 
        
        clmn_list = ['Hedge Type','Hedge Tenor', 'Hedge Weight', 'Hedge Position ID', 'Notional']
        
        return self._liquid_proxy_hedge_df[clmn_list]
