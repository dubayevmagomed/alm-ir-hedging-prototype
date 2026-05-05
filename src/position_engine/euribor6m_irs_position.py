from __future__ import annotations
from pathlib import Path
from datetime import datetime 
import numpy as np
import pandas as pd 
from rateslib import IRS as rlIRS, dcf as rlDCF 
from curve_engine.swap_curve_solver import SwapCurveSolver
from curve_engine.swap_curve_builder import SwapCurveBuilder
from curve_engine.swap_rate_pca_estimator import SwapRatePCAEstimator

class PositionError(Exception):
  pass 

class Euribor6mIRSPosition:
    """
    Representation of a vanilla EUR interest rate swap (IRS).

    This class wraps a RatesLib IRS instrument and provides methods for
    valuation and risk analysis under a given curve.

    Parameters
    ----------
    pos_id : str
        Unique identifier for the position.

    pos_grp : str
        Position grouping (e.g. Hedge, Liability).

    effective_dt : datetime
        Start date of the swap.

    termination : str
        Maturity tenor (e.g. '5Y').

    fixed_rate : float
        Fixed leg rate.

    notional : float
        Trade notional.

    is_receiver : bool
        True if receiving fixed, False if paying fixed.

    position_curve_builder : SwapCurveBuilder
        Curve used for valuation.

    is_par_swap : bool, optional
        If True, fixed rate is set to par rate.

    Methods
    -------
    npv()
        Calculates present value.

    spot_delta()
        Computes PV01 with respect to spot rates.

    fwd_delta()
        Computes forward rate sensitivities.

    pc_multiplier_delta()
        Computes PCA factor sensitivities.

    Notes
    -----
    Sensitivities are computed using RatesLib solver and transformed
    via curve Jacobians where necessary.
    """
    
    def __init__(
        self, 
        pos_id : str, 
        pos_grp : str, 
        effective_dt : datetime, 
        termination : str, 
        fixed_rate : float, 
        notional : float,  
        is_receiver : bool,              
        position_curve_builder : SwapCurveBuilder,        
        is_par_swap : bool = False,        

        ) -> None: 
              
        self.effective_dt = effective_dt
        self.pos_id = pos_id
        self.pos_grp = pos_grp
        self.is_receiver = is_receiver 
        self.position_curve_builder = position_curve_builder
        self.notional = abs(notional)
        
        if is_receiver: 
          rl_notional = abs(notional) * -1
        else: 
          rl_notional = abs(notional)  

        if is_par_swap: 
          self.fixed_rate = float(
              rlIRS( effective = effective_dt,
                     termination = termination,
                     spec = 'eur_irs6' ).rate(curves = self.position_curve_builder.curve) )
        else: 
           self.fixed_rate = fixed_rate 
                     
        self.rl_irs = rlIRS( effective = effective_dt, 
                             termination = termination, 
                             notional = rl_notional, 
                             fixed_rate = self.fixed_rate, 
                             spec = 'eur_irs6'
                           )    
        
        self.maturity_dt = self.rl_irs.cashflows()['Payment'].max()
                
    def cashflows(
        self, 
        calc_curve_builder : SwapCurveBuilder | None = None
        ) -> pd.DataFrame: 
      
      if calc_curve_builder: 
        curve_builder = calc_curve_builder
      else:
        curve_builder = self.position_curve_builder
      
      rl_clmns = [ 'Leg', 'Payment', 'PaymentYears', 'Rate', 'Base Ccy', 'Cashflow', 'DF', 'NPV Ccy']            
      rl_clmns_map = { 'Payment' : 'Payment Date',
                       'PaymentYears': 'Payment Years', 
                       'NPV Ccy' : 'PV', 
                       'Base Ccy' : 'Ccy' 
                    }

      cashflows_df = (
        self.rl_irs.cashflows(curves = curve_builder.curve)
          .assign( PaymentYears = lambda df: 
                       df.apply(  lambda row: 
                                rlDCF( start = self.effective_dt, 
                                         end = row['Payment'],
                                      convention = row['Convention']),
                                axis = 1 
                               ), 
                   Leg = lambda df: 
                       df.apply( lambda row: 
                                 'Fixed' if row['Type'] == 'FixedPeriod' else 'Float', 
                                 axis = 1
                               ) 
                  ) 
          .reset_index()[rl_clmns]
          .rename(columns = rl_clmns_map)
          )  
      
      cashflows_df['Curve'] = curve_builder.curve_id 
      cashflows_df.insert(0, "Position ID", self.pos_id)
      cashflows_df.insert(0, "Position Grp", self.pos_grp)
      cashflows_df.insert(0, "Date", curve_builder.curve_date)

      return cashflows_df

    def spot_delta(
        self, 
        calc_curve_builder : SwapCurveBuilder | None = None
        ) -> pd.DataFrame: 
      
      if calc_curve_builder: 
        curve_builder = calc_curve_builder
      else:
        curve_builder = self.position_curve_builder     
        
      rl_spot_delta = (
          self.rl_irs.delta( curves = [curve_builder.curve],
                             solver = curve_builder.solver)
          .reset_index()    
          .iloc[:,-2:] )        
      
      deltas = rl_spot_delta.iloc[:,-1:].values.reshape(1,-1)
      nodes = rl_spot_delta.iloc[:,-2:-1].values.flatten().tolist()
      spot_delta = pd.DataFrame(data = deltas, columns = nodes)

      spot_delta.columns.name = None 
      
      spot_delta.insert(0, 'Curve', curve_builder.curve_id)            
      spot_delta.insert(0, 'Ccy', curve_builder.solver_currency.upper())
      spot_delta.insert(0, 'Sensitivity', 'Spot PV01')
      spot_delta.insert(0, 'Position ID', self.pos_id)
      spot_delta.insert(0, 'Position Grp', self.pos_grp)
      spot_delta.insert(0, 'Date', curve_builder.curve_date)

      return spot_delta 

    def fwd_delta(
        self, 
        calc_curve_builder : SwapCurveBuilder | None = None
        ) -> pd.DataFrame: 
      
      if calc_curve_builder: 
        curve_builder = calc_curve_builder
      else:
        curve_builder = self.position_curve_builder           
      
      spot_delta = self.spot_delta(calc_curve_builder = curve_builder)

      curve_builder.calc_par_fwd_jacobian()
      par_fwd_jacobian = curve_builder.par_fwd_jacobian.copy()
      spot_delta_vec = spot_delta.iloc[:,6:].values.reshape( par_fwd_jacobian.shape[0], 1)

      fwd_delta_vec = ( 
        par_fwd_jacobian.dot(spot_delta_vec)
        .reset_index() )

      fwd_delta_vec.columns = ['Node','Forward PV01']
      fwd_node_array = fwd_delta_vec['Node'].values.tolist()
      fwd_delta_array = fwd_delta_vec['Forward PV01'].values.reshape(1,-1).tolist() 

      fwd_delta_df = pd.DataFrame(data = fwd_delta_array, columns = fwd_node_array)

      fwd_delta_df.insert(0, 'Curve', curve_builder.curve_id)            
      fwd_delta_df.insert(0, 'Ccy', curve_builder.solver_currency.upper())
      fwd_delta_df.insert(0, 'Sensitivity', 'Forward PV01')
      fwd_delta_df.insert(0, 'Position ID', self.pos_id)
      fwd_delta_df.insert(0, 'Position Grp', self.pos_grp)
      fwd_delta_df.insert(0, 'Date', curve_builder.curve_date)
      
      return fwd_delta_df 
    
    def pc_multiplier_delta(
        self, 
        pca : SwapRatePCAEstimator,         
        calc_curve_builder : SwapCurveBuilder | None = None

      ) -> pd.DataFrame: 
      
      if calc_curve_builder: 
        curve_builder = calc_curve_builder
      else:
        curve_builder = self.position_curve_builder           
  
      spot_delta = self.spot_delta(calc_curve_builder = curve_builder)  
  
      curve_builder.calc_par_pc_jacobian( pca = pca )
      
      par_pc_jacobian = curve_builder.par_pc_jacobian.copy()
      spot_delta_vec = spot_delta.iloc[:,6:].values.reshape( par_pc_jacobian.shape[1], 1)

      pc_multiplier_delta_vec = ( 
        par_pc_jacobian.dot(spot_delta_vec)
        .reset_index() )

      pc_multiplier_delta_vec.columns = ['PC','PC Multiplier Delta']
      pc_array = pc_multiplier_delta_vec['PC'].values.tolist()
      pcm_delta_array = pc_multiplier_delta_vec['PC Multiplier Delta'].values.reshape(1,-1).tolist() 

      pc_multiplier_delta_df = pd.DataFrame(data = pcm_delta_array, columns = pc_array)

      pc_multiplier_delta_df.insert(0, 'Curve', curve_builder.curve_id)            
      pc_multiplier_delta_df.insert(0, 'Ccy', curve_builder.solver_currency.upper())
      pc_multiplier_delta_df.insert(0, 'Sensitivity', 'PC Multiplier Delta')
      pc_multiplier_delta_df.insert(0, 'Position ID', self.pos_id)
      pc_multiplier_delta_df.insert(0, 'Position Grp', self.pos_grp)
      pc_multiplier_delta_df.insert(0, 'Date', curve_builder.curve_date)

      return pc_multiplier_delta_df      
                      
    def spot_gamma(
        self, 
        calc_curve_builder : SwapCurveBuilder | None = None
        ) -> pd.DataFrame: 
      
      if calc_curve_builder: 
        curve_builder = calc_curve_builder
      else:
        curve_builder = self.position_curve_builder                 
      
      rl_spot_gamma = self.rl_irs.gamma(curves = [curve_builder.curve], solver = curve_builder.solver )                                  
      rl_spot_gamma = rl_spot_gamma.reset_index(drop = True)
      rl_spot_gamma.columns = rl_spot_gamma.columns.get_level_values(2)      

      nodes = []
      node_gamma_sums = [] 
      for node, node_gamma in rl_spot_gamma.items():
          nodes.append(node)
          node_gamma_sums.append( float(node_gamma.sum()) )
      
      spot_gamma_df = pd.DataFrame(
        data = np.array(node_gamma_sums).reshape(1,-1), 
        columns = nodes )

      spot_gamma_df.insert(0, 'Curve', curve_builder.curve_id)            
      spot_gamma_df.insert(0, 'Ccy', curve_builder.solver_currency.upper())
      spot_gamma_df.insert(0, 'Sensitivity', 'Gamma')
      spot_gamma_df.insert(0, 'Position ID', self.pos_id)
      spot_gamma_df.insert(0, 'Position Grp', self.pos_grp)
      spot_gamma_df.insert(0, 'Date', curve_builder.curve_date)   

      return spot_gamma_df
    
    def npv(
        self, 
        calc_curve_builder : SwapCurveBuilder | None = None
        ) -> pd.DataFrame:                                 
      
      if calc_curve_builder: 
        curve_builder = calc_curve_builder
      else:
        curve_builder = self.position_curve_builder                       
       
      pv_df = ( pd.DataFrame(
                columns = ['PV'], 
                data = [ float( self.rl_irs.npv( curves = [curve_builder.curve] ) ) ] )
              )

      pv_df.insert(0, 'Curve', curve_builder.curve_id)            
      pv_df.insert(0, 'Ccy', curve_builder.solver_currency.upper())
      pv_df.insert(0, 'Position ID', self.pos_id)
      pv_df.insert(0, 'Position Grp', self.pos_grp)
      pv_df.insert(0, 'Date', curve_builder.curve_date)     
      
      return pv_df  
  
  