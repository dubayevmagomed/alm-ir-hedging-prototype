from __future__ import annotations
from typing import Dict
from pathlib import Path
from datetime import datetime 
import numpy as np
import pandas as pd 
from rateslib import FixedRateBond as rlFixedRateBond
from curve_engine.swap_curve_solver import SwapCurveSolver
from curve_engine.swap_curve_builder import SwapCurveBuilder
from curve_engine.swap_rate_pca_estimator import SwapRatePCAEstimator

class PositionError(Exception):
  pass 
 
class LiabilityPosition:
    """
    Representation of a pension liability cashflow profile.

    The liability is modeled as a portfolio of zero-coupon bonds, where
    each bond corresponds to a future payment.

    Parameters
    ----------
    pos_id : str
        Position identifier.

    pos_grp : str
        Position group (typically 'Liabilities').

    effective_dt : datetime
        Valuation date.

    currency : str
        Currency of cashflows.

    payment_schedule : Dict[str, float]
        Mapping of tenor to payment amount.

    position_curve_builder : SwapCurveBuilder
        Curve used for discounting.

    Attributes
    ----------
    zc_bonds : Dict[str, FixedRateBond]
        Underlying zero-coupon bond instruments.

    notional : float
        Total nominal exposure.

    Methods
    -------
    npv()
        Present value of liabilities.

    spot_delta()
        Sensitivity to par rate shifts.

    Notes
    -----
    This representation ensures compatibility with curve-based pricing
    and enables consistent risk aggregation with hedge instruments.
    """
    
    def __init__(
        self, 
        pos_id : str, 
        pos_grp : str, 
        effective_dt : datetime, 
        currency : str, 
        payment_schedule : Dict[str, float], 
        position_curve_builder : SwapCurveBuilder
        ) -> None: 
        
        if not payment_schedule: 
          raise PositionError("Payment schedule cant be empty")
        
        self.pos_id = pos_id
        self.pos_grp = pos_grp
        self.position_curve_builder = position_curve_builder
        self.currency = currency.upper()
        
        self.zc_bonds = {
             tenor : rlFixedRateBond( 
                        effective = effective_dt, 
                        termination = tenor, 
                        notional = payment, 
                        fixed_rate = 0.0, 
                        eom = False, 
                        currency = currency, 
                        convention = 'Act360', 
                        frequency = 'Z'   ) 
             for tenor, payment in payment_schedule.items() }
        
        self.notional = 0.0         
        for zcb in self.zc_bonds.values():
          cf_df = zcb.cashflows()
          cf = float( cf_df[cf_df.Type == 'Cashflow']['Cashflow'].iloc[0] )
          self.notional += cf
        
        self.maturity_dt = effective_dt
        for zcb in self.zc_bonds.values():
            payment_dt = zcb.cashflows()['Payment'].max()
            if payment_dt > self.maturity_dt: self.maturity_dt = payment_dt
                  
        self.fixed_rate = 0.0    
        
    def cashflows(
        self, 
        calc_curve_builder : SwapCurveBuilder | None = None
        ) -> pd.DataFrame: 
      
      if calc_curve_builder: 
        curve_builder = calc_curve_builder
      else:
        curve_builder = self.position_curve_builder
              
      zcb_cf_dfs = []
      rl_clmns   = ['Date', 'Leg', 'Payment', 'DCF', 'Rate', 'Ccy', 'Cashflow', 'DF', 'NPV' ,'Curve']
      rl_clmns_map = { 'Payment' : 'Payment Date', 
                       'DCF' : 'Payment Years', 
                       'NPV' : 'PV'}
            
      for zcb in self.zc_bonds.values():
        
        rl_cf_df = zcb.cashflows(curves = curve_builder.curve)
        zcb_cf_df = (
          pd.concat( [ rl_cf_df[['Payment', 'DCF', 'DF']].iloc[[0],:].reset_index( drop = True ),
                       rl_cf_df[['Cashflow','NPV', 'Ccy']].iloc[[1],:].reset_index( drop = True ) 
                     ], axis = 1 )
            .assign( Leg = 'Fixed',
                     Curve = curve_builder.curve_id,
                     Date = curve_builder.curve_date, 
                     Rate = self.fixed_rate
                    ) 
                    )
        zcb_cf_dfs.append(zcb_cf_df[rl_clmns])
        
      cashflows_df = ( pd.concat(zcb_cf_dfs)
                         .rename(columns = rl_clmns_map)
                         .reset_index(drop = True) )
      
      cashflows_df.insert(1,"Position ID" , self.pos_id)
      cashflows_df.insert(1,"Position Grp" , self.pos_grp)      
            
      return cashflows_df

    def spot_delta(
        self, 
        calc_curve_builder : SwapCurveBuilder | None = None
        ) -> pd.DataFrame: 
      
      if calc_curve_builder: 
        curve_builder = calc_curve_builder
      else:
        curve_builder = self.position_curve_builder
      
      zcb_deltas = []
      for zcb in self.zc_bonds.values():
          rl_delta_df = zcb.delta( curves = [curve_builder.curve], solver = curve_builder.solver ).reset_index()
          rl_delta_df.columns = rl_delta_df.columns.get_level_values(0)
          rl_delta_df.columns.name = None 
              
          zcb_delta = rl_delta_df.iloc[:,-2:]
          zcb_delta.columns = ['Node','Spot PV01']       
          zcb_deltas.append(zcb_delta)

      node_deltas = (
        pd.concat(zcb_deltas)
          .reset_index(drop=True)
          .groupby(['Node'])['Spot PV01'].sum() 
          .sort_index( key = lambda x: x.str[:-1].astype('int64'), ascending = True )
        )

      delta_df = pd.DataFrame( data = node_deltas.values.reshape(1,10), columns = node_deltas.index )
      delta_df.columns.name = None 
      
      delta_df.insert(0, 'Curve', curve_builder.curve_id)      
      delta_df.insert(0, 'Ccy', curve_builder.solver_currency.upper())
      delta_df.insert(0, 'Sensitivity', 'Spot PV01')
      delta_df.insert(0, 'Position ID', self.pos_id)
      delta_df.insert(0, 'Position Grp', self.pos_grp)
      delta_df.insert(0, 'Date', curve_builder.curve_date)

      return delta_df 

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
                                               
      zcb_gamma_dfs = []
      for zcb_tenor, zcb in self.zc_bonds.items():
          rl_spot_gamma_df = zcb.gamma(curves = [curve_builder.curve], solver = curve_builder.solver)
          rl_spot_gamma_df = rl_spot_gamma_df.reset_index(drop=True)
          rl_spot_gamma_df.columns = rl_spot_gamma_df.columns.get_level_values(2).tolist()
          rl_spot_gamma_df.index = rl_spot_gamma_df.columns
        
          nodes = []
          node_gamma_sums = [] 
          for node, node_gamma in rl_spot_gamma_df.items():
              nodes.append(node)
              node_gamma_sums.append( float(node_gamma.sum()) )    
          
          zcb_gamma_df = pd.DataFrame(
              data = np.array(node_gamma_sums).reshape(1,-1), 
              columns = nodes )
          zcb_gamma_df.index = [zcb_tenor]
          zcb_gamma_dfs.append(zcb_gamma_df)
          
      spot_gamma_df = pd.DataFrame(
        data = pd.concat(zcb_gamma_dfs).sum().values.reshape(1,-1),
        columns = pd.concat(zcb_gamma_dfs).sum().index )
      
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
        
      npv_df = ( 
      self.cashflows(curve_builder)[['Position Grp', 'Position ID', 'PV']]   
          .groupby(['Position Grp', 'Position ID'])[['PV']]
          .sum()
          .reset_index()
          [['Position Grp', 'Position ID' ,'PV']] )  

      npv_df.insert(2, 'Curve', curve_builder.curve_id)
      npv_df.insert(2, 'Ccy', curve_builder.solver_currency.upper())
      npv_df.insert(0, 'Date', curve_builder.curve_date)          
               
      return npv_df
    
    @property
    def cashflow_zcb_positions(
      self
    ) -> pd.DataFrame : 
      
      in_clmn_list = ['Payment Date', 'Payment Years', 'Ccy', 'Curve', 'Cashflow', 'DF', 'PV']
      out_clmn_list = ['Cashflow Zero Coupon Bond','Notional','Ccy','Maturity Date','Maturity Years','Curve','DF','PV']
      cashflow_zcb_positions = ( self.cashflows()
                                  .copy()[in_clmn_list]
                                  .rename( columns = { 'Payment Date' : 'Maturity Date',
                                                       'Payment Years' : 'Maturity Years', 
                                                       'Cashflow' : 'Notional'
                                                     } ) 
                                  .assign( **{ 'Cashflow Zero Coupon Bond' : 
                                                lambda df: 'ZCB-' + df['Maturity Years'].astype(int).astype('str') 
                                             } )
                                )
      return cashflow_zcb_positions[out_clmn_list]