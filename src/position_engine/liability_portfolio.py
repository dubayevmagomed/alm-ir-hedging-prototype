from __future__ import annotations
from pathlib import Path
from typing import List
from datetime import datetime 
import numpy as np
import pandas as pd 
from curve_engine.swap_curve_solver import SwapCurveSolver
from curve_engine.swap_curve_builder import SwapCurveBuilder
from curve_engine.swap_curve_scenario_generator import SwapCurveScenarioGenerator
from curve_engine.swap_rate_mc_simulator import SwapRateMCSimulator
from position_engine.liability_position import LiabilityPosition
from position_engine.euribor6m_irs_position import Euribor6mIRSPosition

class PortfolioError(Exception):
  pass 

class LiabilityPortfolio:
    """
    Portfolio of liability and hedge positions.

    This class aggregates positions and provides portfolio-level valuation,
    risk metrics, and scenario-based PnL analysis.

    Parameters
    ----------
    effective_dt : datetime
        Valuation date.

    prtfl_id : str
        Portfolio identifier.

    positions : List
        List of position objects.

    Attributes
    ----------
    pv : pd.DataFrame
        Portfolio present value.

    spot_delta : pd.DataFrame
        Aggregated PV01.

    sim_prtfl_pnl : pd.DataFrame
        Scenario-based PnL.

    Methods
    -------
    simulate_pnl()
        Reprices portfolio under MC scenarios.

    calc_cvar()
        Computes VaR and CVaR.

    Notes
    -----
    Supports full revaluation risk framework:
    scenario → pricing → aggregation → risk metrics.
    """
    
    def __init__(
        self, 
        effective_dt : datetime,         
        prtfl_id : str, 
        positions : List[LiabilityPosition | Euribor6mIRSPosition]
        ) -> None: 
              
        self.prtfl_id = prtfl_id
        self.effective_dt = effective_dt
        self._positions = { pos.pos_id : pos for pos in positions }

        pos_map = self._positions
        notional_map = {k: v.notional for k, v in pos_map.items()}
        rate_map = {k: v.fixed_rate for k, v in pos_map.items()}
        maturity_map = {k: v.maturity_dt for k, v in pos_map.items()}        
        positions_df = pd.concat( [ pos.npv() for pos in self._positions.values() ] )
        positions_df['Notional'] = positions_df['Position ID'].map(notional_map)
        positions_df['Rate'] = positions_df['Position ID'].map(rate_map)
        positions_df['Maturity'] = positions_df['Position ID'].map(maturity_map)   
        
        self.positions = positions_df.reset_index(drop=True)                
        self.pv = self.npv()        
        self.cashflows = self._cashflows()
        self.spot_delta = self._aggregate_sensitivity('spot_delta')
        self.fwd_delta = self._aggregate_sensitivity('fwd_delta')
        self.spot_gamma = self._aggregate_sensitivity('spot_gamma')
        
        self._sim_pnl = None 
        self._pos_cvar = None 
        self._pos_grp_cvar = None 
        self._prtfl_cvar = None 

    def _cashflows(
        self 
        ) -> pd.DataFrame: 
      
      liability_pos = [_.pos_id for _ in self._positions.values() if isinstance(_, LiabilityPosition) ]     
       
      if liability_pos: 
          cashflows = pd.concat( [_.cashflows() for _ in self._positions.values() if isinstance(_, LiabilityPosition) ] 
                               )         
          cashflows.insert(1, "Portfolio ID", self.prtfl_id)
          cashflow_clmns = ['Date', 'Portfolio ID', 'Position Grp', 'Position ID', 'Leg', 'Payment Date', 
                            'Payment Years', 'Rate', 'Ccy', 'Curve', 'Cashflow', 'DF', 'PV']
          return cashflows[cashflow_clmns]
          
    def _aggregate_sensitivity(
       self,
       method_name: str) -> pd.DataFrame:
      
      dfs = [getattr(pos, method_name)() for pos in self._positions.values()]
      df = pd.concat(dfs, ignore_index=True).drop(columns='Date')

      pos_cols = ['Position Grp','Position ID','Sensitivity','Ccy','Curve']
      node_cols = [c for c in df.columns if c not in pos_cols]
    
      total_row = df[node_cols].sum().to_frame().T.astype(object)
      total_row['Position Grp'] = ''
      total_row['Position ID'] = 'Total'
      total_row['Sensitivity'] = df['Sensitivity'].iloc[0]
      total_row['Ccy'] = df['Ccy'].iloc[0]
      total_row['Curve'] = df['Curve'].iloc[0]
      
      df = pd.concat([df, total_row], ignore_index=True)

      df.insert(0, 'Portfolio ID', self.prtfl_id)
      df.insert(0, 'Date', self.effective_dt)

      df.insert(df.columns.get_loc(node_cols[0]), 'Total', df[node_cols].sum(axis=1))

      return df

    def npv(
        self, 
        calc_curve_builder : SwapCurveBuilder | None = None
        ) -> pd.DataFrame:                         
       
      if calc_curve_builder:
        pos_pv_dfs = [ _.npv(calc_curve_builder = calc_curve_builder) for _ in self._positions.values() ]
      else:
        pos_pv_dfs = [ _.npv() for _ in self._positions.values() ]  
        
      prtfl_pv_df = pd.concat( pos_pv_dfs, ignore_index = True, axis = 0).reset_index(drop = True)
      prtfl_pv_df.drop(columns = 'Date', inplace = True)

      pos_cols = ['Position Grp','Position ID','Ccy','Curve']
      pv_cols = [clmn for clmn in prtfl_pv_df.columns if clmn not in pos_cols]
      pv_totals = prtfl_pv_df[pv_cols].sum()

      total_row = {col: None for col in prtfl_pv_df.columns}

      for col in pv_cols:
          total_row[col] = pv_totals[col]

      total_row['Position Grp'] = ''
      total_row['Position ID'] = 'Total'
      total_row['Ccy'] = prtfl_pv_df['Ccy'].iloc[0]
      total_row['Curve'] = prtfl_pv_df['Curve'].iloc[0]

      total_row = pd.DataFrame([total_row])

      prtfl_pv_df = pd.concat([prtfl_pv_df, total_row], ignore_index=True)
      prtfl_pv_df.insert(0, 'Portfolio ID', self.prtfl_id)
      prtfl_pv_df.insert(0, 'Date', self.effective_dt)

      return prtfl_pv_df
    
    def simulate_pnl(
        self, 
        mc_curve_scenarios : SwapCurveScenarioGenerator 
        ) -> None : 

      base_pvs = {
        p:v for p,v in zip(
          self.pv['Position ID'].iloc[:-1].to_numpy(),
          self.pv['PV'].iloc[:-1].to_numpy()) }

      sim_pnl = {} 

      for pos in self._positions.values():
        pos_base_pv = base_pvs[pos.pos_id]            
        
        pos_ccy = pos.position_curve_builder.solver_currency.upper()
        pos_curve_id = mc_curve_scenarios.mc_base_curve_id 
        mc_scenarios = []
        pos_mc_sim_pnls = []
        
        for scenario, scenario_curve in mc_curve_scenarios.mc_scenario_curves.items():  
          mc_scenarios.append(scenario)          
          pos_mc_sim_pnls.append( float( pos.npv(calc_curve_builder = scenario_curve)['PV'].iloc[0] ) - pos_base_pv )  
          
          
        sim_pnl[pos.pos_id] = {
          'Date': self.effective_dt, 
          'Portfolio ID' : self.prtfl_id,
          'Position Grp' : pos.pos_grp,     
          'Position ID' : pos.pos_id,      
          'Time Horizon (Days)' : mc_curve_scenarios.mc_t_scale_factor, 
          'Ccy' : pos_ccy, 
          'Curve' : pos_curve_id,        
          'Scenario' : mc_scenarios, 
          'PnL' : pos_mc_sim_pnls }

      self._sim_pnl = sim_pnl    
      self._mc_curve_scenarios = mc_curve_scenarios 
      
    @property
    def sim_pos_pnl(
        self
        ) -> pd.DataFrame : 
      
      if self._sim_pnl is None:
        raise PortfolioError("Portfolio PnL is not simulated, execute method simulate_pnl()")
         
      else: 
        return ( 
        pd.concat( [pd.DataFrame(_) for _ in self._sim_pnl.values()] )
          .reset_index(drop=True) 
          )
        
    @property
    def sim_pos_grp_pnl(
        self
        ) -> pd.DataFrame : 

      grp_clmns = ['Date','Portfolio ID','Position Grp', 'Time Horizon (Days)', 'Ccy', 'Curve', 'Scenario']
      sim_pos_grp_pnl = ( 
        self.sim_pos_pnl
            .groupby(grp_clmns)[['PnL']]
            .sum() 
            .reset_index() )
      
      return sim_pos_grp_pnl

    @property
    def sim_prtfl_pnl(
        self
        ) -> pd.DataFrame : 
          
      grp_clmns = ['Date','Portfolio ID', 'Time Horizon (Days)', 'Ccy', 'Curve', 'Scenario']          
      sim_prtfl_pnl = ( 
      self.sim_pos_pnl
          .groupby(grp_clmns)[['PnL']]
          .sum() 
          .reset_index() )
      
      return sim_prtfl_pnl
    
    def calc_cvar(
        self, 
        cvar_quantile : float = 0.95
       ) -> None :

      sim_prtfl_pnl = self.sim_prtfl_pnl
      sim_pos_grp_pnl = self.sim_pos_grp_pnl
      sim_pos_pnl = self.sim_pos_pnl

      quantile = cvar_quantile       
      date = sim_prtfl_pnl['Date'].unique()[0]
      prtfl_id = sim_prtfl_pnl['Portfolio ID'].unique()[0]
      time_horizon = sim_prtfl_pnl['Time Horizon (Days)'].unique()[0]
      ccy = sim_prtfl_pnl['Ccy'].unique()[0]
      curve = sim_prtfl_pnl['Curve'].unique()[0]

      var = sim_prtfl_pnl.groupby('Portfolio ID')['PnL'].quantile(1-quantile).to_numpy()[0] 
      cvar = sim_prtfl_pnl[sim_prtfl_pnl.PnL <= var].PnL.mean()
      tail = sim_prtfl_pnl[sim_prtfl_pnl.PnL <= var].Scenario.to_list() 


      pos_list = sim_pos_pnl['Position ID'].unique().tolist()
      pos_pnls = { pos : sim_pos_pnl[ sim_pos_pnl['Position ID'] == pos ] for pos in pos_list }
      pos_cvars = {}
      for pos, pos_pnl in pos_pnls.items():
          pos_var = pos_pnl.PnL.quantile(1-quantile) 
          pos_cvar = pos_pnl[pos_pnl.PnL <= pos_var].PnL.mean() 
          pos_component_cvar = pos_pnl[pos_pnl.Scenario.isin(tail)].PnL.mean() 
          pos_cvars[pos] = { 
            'Date' : date, 
            'Portfolio ID' : prtfl_id, 
            'Position Grp' : pos_pnl['Position Grp'].unique()[0],
            'Position ID' : pos, 
            'Time Horizon (Days)' : time_horizon, 
            'Ccy' : ccy, 
            'Curve' : curve,
            f"Standalone {round(quantile,3)*100}% MC VaR" : pos_var * -1, 
            f"Standalone {round(quantile,3)*100}% MC CVaR" : pos_cvar * -1, 
            f"Component {round(quantile,3)*100}% MC CVaR": pos_component_cvar * -1
                           }    

      pos_grp_list = sim_pos_grp_pnl['Position Grp'].unique().tolist()
      pos_grp_pnls = { pos_grp : sim_pos_grp_pnl[ sim_pos_grp_pnl['Position Grp'] == pos_grp ] for pos_grp in pos_grp_list }    
      pos_grp_cvars = {}
      for pos_grp, pos_grp_pnl in pos_grp_pnls.items():
          pos_grp_var = pos_grp_pnl.PnL.quantile(1-quantile) 
          pos_grp_cvar = pos_grp_pnl[pos_grp_pnl.PnL <= pos_grp_var].PnL.mean() 
          pos_grp_component_cvar = pos_grp_pnl[pos_grp_pnl.Scenario.isin(tail)].PnL.mean() 
          pos_grp_cvars[pos_grp] = { 
            'Date' : date, 
            'Portfolio ID' : prtfl_id, 
            'Position Grp' : pos_grp, 
            'Time Horizon (Days)' : time_horizon, 
            'Ccy' : ccy, 
            'Curve' : curve,  
            f"Standalone {round(quantile,3)*100}% MC VaR"  : pos_grp_var * -1, 
            f"Standalone {round(quantile,3)*100}% MC CVaR" : pos_grp_cvar * -1, 
            f"Component  {round(quantile,3)*100}% MC CVaR" : pos_grp_component_cvar * -1
                                  }    

      self._pos_cvar = pd.concat( 
          [pd.DataFrame(pos_cvar, index = [0]) for pos_cvar in pos_cvars.values()] 
          ).reset_index(drop = True)
      
      self._pos_grp_cvar = pd.concat( 
          [pd.DataFrame(pos_grp_cvar, index = [0]) for pos_grp_cvar in pos_grp_cvars.values()] 
          ).reset_index(drop = True)
      
      self._prtfl_cvar = ( pd.DataFrame( {
          'Date' : date, 
          'Portfolio ID' : prtfl_id, 
          'Time Horizon (Days)' : time_horizon, 
          'Ccy' : ccy, 
          'Curve' : curve, 
          f"Standalone {round(quantile,3)*100}% MC VaR" : var * -1, 
          f"Standalone {round(quantile,3)*100}% MC CVaR" : cvar * -1, 
          f"Component {round(quantile,3)*100}% MC CVaR": cvar * -1           
          } , index = [0]) )
      
      self._cvar_scenarios = ( pd.DataFrame( {
         'Date' : date, 
         'Scenario' : tail
          } ) ).reset_index(drop = True)   
      
      self._calc_cvar_attr()
      
    def _calc_cvar_attr(
        self
      ) -> None :
      
      cvar_total = self.prtfl_cvar.values.flatten()[-1] * -1  
      sim_shifts = self._mc_curve_scenarios.sim_shifts.copy()
      pv01_deltas = self.spot_delta.copy() 

      pv01_delta_clmns_start = self.spot_delta.columns.to_list().index('Total') + 1

      shift_clmns_start = sim_shifts.columns.to_list().index('Scenario') + 1

      pv01_delta_idx = self.spot_delta.loc[:, 'Position ID'] == 'Total' 

      cvar_tail_idx = sim_shifts['Scenario'].isin(self.cvar_scenarios['Scenario'].to_list())

      tail_shifts_mat = sim_shifts[cvar_tail_idx].iloc[:,shift_clmns_start:]

      pv01_delta_vec = pv01_deltas[pv01_delta_idx].iloc[: , pv01_delta_clmns_start : ].values 

      pv01_delta_tail_pnl = (tail_shifts_mat * pv01_delta_vec).mean().to_frame().T

      pv01_delta_tail_pnl_shr = pv01_delta_tail_pnl / cvar_total * 100 
      
      pv01_delta_row1_df = pd.DataFrame( {
        'Portfolio ID' : self.prtfl_id, 
        'Measure' : 'PV01 Delta Contribution in Tail Scenarios', 
        'Total' : pv01_delta_tail_pnl.sum().sum()
      }, index = [0])

      pv01_delta_row2_df = pd.DataFrame( {
        'Portfolio ID' : self.prtfl_id, 
        'Measure' : 'Same but as % of MC CVaR', 
        'Total' : pv01_delta_tail_pnl_shr.sum().sum()  
      }, index = [0])

      pv01_delta_cvar_attr = pd.concat( [
        pd.concat([pv01_delta_row1_df, pv01_delta_tail_pnl], axis = 1), 
        pd.concat([pv01_delta_row2_df, pv01_delta_tail_pnl_shr], axis = 1)  
      ], axis = 0).reset_index(drop=True)      
      
      self._pv01_delta_cvar_attr = pv01_delta_cvar_attr 
       
    @property   
    def pos_cvar(
        self 
       ) -> pd.DataFrame :
      
      if self._pos_cvar is None:
        raise PortfolioError("CVaR is not calculated, execute method calc_cvar()")
      else: 
        return self._pos_cvar
      
    @property   
    def pos_grp_cvar(
        self 
       ) -> pd.DataFrame :
      
      if self._pos_grp_cvar is None:
        raise PortfolioError("CVaR is not calculated, execute method calc_cvar()")
      else: 
        return self._pos_grp_cvar      
      
    @property   
    def prtfl_cvar(
        self 
       ) -> pd.DataFrame :
      
      if self._prtfl_cvar is None:
        raise PortfolioError("CVaR is not calculated, execute method calc_cvar()")
      else: 
        return self._prtfl_cvar
      
    @property   
    def cvar_scenarios(
        self 
       ) -> pd.DataFrame :
      
      if self._cvar_scenarios is None:
        raise PortfolioError("CVaR is not calculated, execute method calc_cvar()")
      else: 
        return self._cvar_scenarios      
      
    @property   
    def pv01_delta_cvar_attr(
        self 
       ) -> pd.DataFrame :
      
      if self._pv01_delta_cvar_attr is None:
        raise PortfolioError("CVaR is not calculated, execute method calc_cvar()")
      else: 
        return self._pv01_delta_cvar_attr