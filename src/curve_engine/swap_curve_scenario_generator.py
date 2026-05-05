from __future__ import annotations
from typing import Dict
from pathlib import Path
from datetime import datetime 
import numpy as np
import pandas as pd 
from curve_engine.swap_rate_pca_estimator import SwapRatePCAEstimator
from curve_engine.swap_curve_solver import SwapCurveSolver
from curve_engine.swap_curve_builder import SwapCurveBuilder
from curve_engine.swap_rate_mc_simulator import SwapRateMCSimulator
 
class SwapCurveScenarioGenerator:
    """
    Scenario generator for interest rate curves.

    This class combines Monte Carlo simulation of swap rates with curve
    bootstrapping to produce a set of fully calibrated yield curves under
    simulated market conditions.

    Each scenario consists of:
    - simulated swap rates
    - calibrated discount curve
    - derived curve analytics

    Parameters
    ----------
    curve_date : datetime
        Valuation date for curve construction.

    swap_rate_sim : SwapRateMCSimulator
        Monte Carlo simulated scenarios of (par) swap rate curve.
        
    swap_rate_pca : SwapRatePCAEstimator
        Principal Component Analysis of (par) swap rate curve. 

    Attributes
    ----------
    hist_rates : pd.DataFrame
        Historical rate levels.

    sim_rates : pd.DataFrame
        Simulated rate levels.

    mc_scenario_curves : Dict[int, SwapCurveBuilder]
        Dictionary mapping scenario ID to calibrated curve objects.

    pca_loadings : pd.DataFrame
        PCA factor loadings for risk analysis.

    Notes
    -----
    - Each scenario is independently bootstrapped into a curve.
    - Enables full revaluation of portfolios under simulated market states.
    """
       
    def __init__(
        self, 
        curve_date : datetime, 
        swap_rate_sim : SwapRateMCSimulator, 
        swap_rate_pca : SwapRatePCAEstimator, 
        ) -> None: 

        mc_sim = swap_rate_sim
        pca = swap_rate_pca
        
        self.curve_date = curve_date           
        self.mc_n_scenarios = mc_sim.n_scenarios
        self.mc_t_scale_factor = mc_sim.t_scale_factor
        self.hist_rates = mc_sim.rates                 
        self.hist_shifts = mc_sim.shifts
        self.hist_shifts.loc[:,'2Y':] = self.hist_shifts.loc[:,'2Y':] * 100
        self.hist_cshifts = mc_sim.cshifts 
        self.hist_cshifts.loc[:,'2Y':] = self.hist_cshifts.loc[:,'2Y':] * 100
        self.hist_cov_mat = mc_sim.cov_mat
        self.hist_cov_mat.iloc[:, 1:] = self.hist_cov_mat.iloc[:, 1:] * 100 * 100
        self.hist_corr_mat = mc_sim.corr_mat         

        self.sim_rates = mc_sim.rate_scenarios        
        self.sim_shifts = mc_sim.shift_scenarios                                
        self.sim_shifts.loc[:,'2Y':] = self.sim_shifts.loc[:,'2Y':] * 100
        
        self.mc_scenarios = {
             int(row[1].iloc[0]) :      # each row as tuple. 2nd element of tuple is row values as Series. 
             row[1].iloc[1:].to_dict()  # key is set to scenario value, rates and tenors are collected into a dict.
             for row in mc_sim.rate_scenarios.iterrows() }        
        

        self.pca_eigvals = pca.pca_eigvals
        self.pca_eigvecs = pca.pca_eigvecs
        self.pca_loadings = pca.pca_loadings
        self.pca_loadings.loc[:,'PC1':] = self.pca_loadings.loc[:,'PC1':] * 100                
        self.sim_pc_multipliers = self._calc_sim_pc_multipliers()
        
    def _calc_sim_pc_multipliers(
        self
        ) -> None: 
        
        sim_shifts_mat = self.sim_shifts.loc[:,'2Y':].values
        pca_eigvecs_mat = self.pca_eigvecs.loc[:,'PC1':].values  
        sim_pc_multipliers_mat = sim_shifts_mat @ pca_eigvecs_mat
        sim_pc_multipliers = (
            pd.DataFrame( data = sim_pc_multipliers_mat, 
                          index = self.sim_shifts['Scenario'].values, 
                          columns = self.pca_eigvecs.loc[:,'PC1':].columns )
              .reset_index()
              .rename(columns = {'index' : 'Scenario'})
        )
        return sim_pc_multipliers
                        
    def _mc_scenario_curve_builder(
        self, 
        curve_id : str, 
        scenario_id : int, 
        scenario_rates : Dict[str, float]        
        
        ) -> SwapCurveBuilder:
    
        scenario_solver_id = f"{curve_id} {scenario_id}" 
        scenario_curve_id  = f"{curve_id} {scenario_id}"
        
        scenario_solver = SwapCurveSolver(
            solver_id = scenario_solver_id, 
            curve_date = self.curve_date, 
            instrument_rates = scenario_rates 
            ) 
      
        scenario_curve = SwapCurveBuilder.from_solver(
            curve_id = scenario_curve_id, 
            curve_solver = scenario_solver
            )
        
        return scenario_curve    
    
    def build_mc_scenario_curves(
        self, 
        curve_id : str
        ) -> None: 
        
        self.mc_base_curve_id = curve_id 
        self.mc_scenario_curves = { 
             scenario : 
             self._mc_scenario_curve_builder( 
                  curve_id = curve_id, 
                  scenario_id = scenario, 
                  scenario_rates = scenario_rates)
           
              for scenario, scenario_rates in self.mc_scenarios.items() }     
                
    def _build_stress_scenario_curves(
        self
        ) -> None: 
        pass