from __future__ import annotations
from typing import Dict
from datetime import datetime 
from pathlib import Path
import os 
import pandas as pd 
from curve_engine.swap_curve_solver import SwapCurveSolver
from curve_engine.swap_curve_builder import SwapCurveBuilder
from curve_engine.swap_rate_pca_estimator import SwapRatePCAEstimator
from curve_engine.swap_rate_mc_simulator import SwapRateMCSimulator 
from curve_engine.swap_curve_scenario_generator import SwapCurveScenarioGenerator
from position_engine.liability_position import LiabilityPosition
from position_engine.liability_portfolio import LiabilityPortfolio
from hedge_engine.liability_hedge_constructor import LiabilityHedgeConstructor

def create_calc_batch(
    analysis_date : datetime, 
    analysis_date_swap_rates : Dict[str, float], 
    liabilities_cashflow_schedule : Dict[str, float], 
    mc_n_scenarios : float, 
    mc_t_scale_factor : float, 
    mc_rng_seed : int = 1, 
    swap_ts_csv_path: Path | None = None  
    ) -> Dict[str, pd.DataFrame]:
    
    # Note:
    # Both SwapRateMCSimulator and SwapRatePCAEstimator require historical
    # swap rate time series data in a specific format.
    #
    # If no explicit file path is provided to create_calc_batch function,
    # a default CSV file will be used instead.
    #
    # The default dataset is located in the "curve_data" folder of the project
    # (see "eur_swap_ts.csv" for the expected format).

    if swap_ts_csv_path is None: 
    
      import os 
      from pathlib import Path

      SWAP_TIME_SERIES_CSV_PATH = ""    
      
      for parent in Path(__file__).parents:
         if (parent / "src").exists():
             SWAP_TIME_SERIES_CSV_PATH = parent / "curve_data" / "eur_swap_ts.csv"
         break       
          
      csv_path = SWAP_TIME_SERIES_CSV_PATH 
    
    else: 
      
      csv_path = swap_ts_csv_path
         
    solver = SwapCurveSolver(
        solver_id = 'eurswap', 
        curve_date = analysis_date, 
        instrument_rates = analysis_date_swap_rates )
    
    curve = SwapCurveBuilder.from_solver(
        curve_id = "eurswap",
        curve_solver = solver)
    
    # Generate PCA for the curve
    swap_rate_pca = SwapRatePCAEstimator.from_csv( 
      swap_ts_csv_path = csv_path 
    ) 
    
    # Generate MC scenarios for the swap rates curve - based on historical data 
    swap_rate_sim = SwapRateMCSimulator.from_csv(
      n_scenarios = mc_n_scenarios, 
      t_scale_factor = mc_t_scale_factor, 
      rng_seed = mc_rng_seed, 
      swap_ts_csv_path = csv_path 
    )
    
    # Generate the Jacobians for par-to-forward and par-to-PC transformations
    curve.calc_par_fwd_jacobian()    
    curve.calc_par_pc_jacobian(pca = swap_rate_pca)

    # Generate scenarios for the eurswap curve, calibrated to simulated swap rates 
    scenarios = SwapCurveScenarioGenerator(  
    curve_date = analysis_date, 
    swap_rate_sim = swap_rate_sim, 
    swap_rate_pca = swap_rate_pca
    )
    
    # Calibrate curve in each scenario     
    print("Starting MC simulation of curve scenarios:", datetime.now())
    scenarios.build_mc_scenario_curves( curve_id = 'eurswap' )
    print("...ended at:", datetime.now())
    
    # Create the Liabilities position 
    lby_pos = LiabilityPosition(
      pos_id = 'liabilities 30Y', 
      pos_grp = 'liabilities', 
      effective_dt = analysis_date, 
      currency = 'eur', 
      position_curve_builder = curve,       
      payment_schedule = liabilities_cashflow_schedule )

    # Construct the hedges for liability position 
    lby_hedge = LiabilityHedgeConstructor(
      liability_position = lby_pos, 
      liability_curve = curve, 
      liquid_irs_tenors = ['2Y','5Y','10Y','30Y'], 
      swap_rate_cov_mat = scenarios.hist_cov_mat, 
      hedge_dt = analysis_date )
    
    # Create the base, unhedged liability portfolio (consistin of the liability position only)
    lby_prtfl = LiabilityPortfolio( 
      positions = [lby_pos], 
      prtfl_id = 'liability portfolio (unhedged)', 
      effective_dt = analysis_date )
    
    # Create the fully hedged liability portfolio 
    lby_prtfl_full_hedge = (
      lby_hedge.create_hedged_portfolio(
      liability_position = lby_hedge.liability_position, 
      hedge_positions = lby_hedge.benchmark_hedge_positions, 
      effective_dt = analysis_date, 
      prtfl_id = 'liability portfolio (benchmark hedge)' ) ) 
    
    # Create the proxy hedged liability portfolio
    lby_prtfl_proxy_hedge = (
      lby_hedge.create_hedged_portfolio(
      liability_position = lby_hedge.liability_position, 
      hedge_positions = lby_hedge.liquid_proxy_hedge_positions, 
      effective_dt = analysis_date, 
      prtfl_id = 'liability portfolio (proxy hedge)' ) ) 
    
    # Simulate PnL for all portfolios 
    print("Starting MC simulation of PnL for Liability portfolios:", datetime.now()) 
    lby_prtfl.simulate_pnl(mc_curve_scenarios = scenarios)
    lby_prtfl_full_hedge.simulate_pnl(mc_curve_scenarios = scenarios)
    lby_prtfl_proxy_hedge.simulate_pnl(mc_curve_scenarios = scenarios)
    print("...ended at:", datetime.now())
          
    # Calculate CVaR and related risk metrics for all portfolios 
    lby_prtfl.calc_cvar(0.95)
    lby_prtfl_full_hedge.calc_cvar(0.95)
    lby_prtfl_proxy_hedge.calc_cvar(0.95)

    # Collect relevant data as dfs in a dict structure
    lby_pos_cashflows = lby_pos.cashflows().copy() 
    lby_pos_spot_delta = lby_pos.spot_delta().copy() 
    lby_pos_fwd_delta = lby_pos.fwd_delta().copy() 
    lby_pos_spot_gamma = lby_pos.spot_gamma().copy() 
    lby_pos_cashflow_zcb_positions = lby_pos.cashflow_zcb_positions.copy()
    
    # Create df with description of MC simulation 
    sim_parameters = pd.DataFrame( 
      { 'Analysis Date' : scenarios.curve_date, 
        'No. Scenarios' : scenarios.mc_n_scenarios, 
        'Time Scaling (Days)' : scenarios.mc_t_scale_factor
      }, index = [0]
    )
    
    return { 
            "curve" : 
              { 
                "nodes" : curve.nodes, 
                "disc_factors" : curve.disc_factors, 
                "spot_zero_rates" : curve.spot_zero_rates, 
                "spot_par_rates" : curve.spot_par_rates, 
                "fwd_par_rates" : curve.fwd_par_rates, 
                "pca_eigvals" : curve.pca_eigvals, 
                "pca_eigvecs" : curve.pca_eigvecs, 
                "pca_loadings" : curve.pca_loadings
               }, 
            "scenarios" :
              { 
                "hist_rates" : scenarios.hist_rates,
                "hist_shifts" : scenarios.hist_shifts, 
                "hist_cov_mat" : scenarios.hist_cov_mat, 
                "hist_corr_mat" : scenarios.hist_corr_mat, 
                "sim_rates" : scenarios.sim_rates, 
                "sim_shifts" : scenarios.sim_shifts, 
                "pca_eigvals" : scenarios.pca_eigvals,                                  
                "pca_eigvecs" : scenarios.pca_eigvecs,
                "pca_loadings" : scenarios.pca_loadings,
                "sim_pc_multipliers" : scenarios.sim_pc_multipliers, 
                "sim_parameters" : sim_parameters                                
              }, 
            "lby_pos" :
              { 
                "cashflows" : lby_pos_cashflows, 
                "spot_delta" : lby_pos_spot_delta, 
                "fwd_delta" : lby_pos_fwd_delta, 
                "spot_gamma" : lby_pos_spot_gamma, 
                "cashflow_zcb_positions" : lby_pos_cashflow_zcb_positions
              }, 
            "lby_hedge" :
              { 
                "full_hedge_instruments" : lby_hedge.full_hedge_instruments, 
                "liquid_hedge_instruments" : lby_hedge.liquid_hedge_instruments 
              },               
            "lby_prtfl" : 
              {
                "positions" : lby_prtfl.positions, 
                "cashflows" : lby_prtfl.cashflows, 
                "npv" : lby_prtfl.pv, 
                "spot_delta" : lby_prtfl.spot_delta, 
                "fwd_delta" : lby_prtfl.fwd_delta, 
                "spot_gamma" : lby_prtfl.spot_gamma, 
                "sim_pos_pnl" : lby_prtfl.sim_pos_pnl, 
                "sim_pos_grp_pnl" : lby_prtfl.sim_pos_grp_pnl, 
                "sim_prtfl_pnl" : lby_prtfl.sim_prtfl_pnl, 
                "prtfl_cvar" : lby_prtfl.prtfl_cvar, 
                "pos_grp_cvar" : lby_prtfl.pos_grp_cvar, 
                "pos_cvar" : lby_prtfl.pos_cvar, 
                "cvar_scenarios" : lby_prtfl.cvar_scenarios, 
                "pv01_delta_cvar_attr" : lby_prtfl.pv01_delta_cvar_attr
              }, 
            "lby_prtfl_full_hedge" : 
              {
                "positions" : lby_prtfl_full_hedge.positions, 
                "cashflows" : lby_prtfl_full_hedge.cashflows, 
                "npv" : lby_prtfl_full_hedge.pv, 
                "spot_delta" : lby_prtfl_full_hedge.spot_delta, 
                "fwd_delta" : lby_prtfl_full_hedge.fwd_delta, 
                "spot_gamma" : lby_prtfl_full_hedge.spot_gamma, 
                "sim_pos_pnl" : lby_prtfl_full_hedge.sim_pos_pnl, 
                "sim_pos_grp_pnl" : lby_prtfl_full_hedge.sim_pos_grp_pnl, 
                "sim_prtfl_pnl" : lby_prtfl_full_hedge.sim_prtfl_pnl, 
                "prtfl_cvar" : lby_prtfl_full_hedge.prtfl_cvar, 
                "pos_grp_cvar" : lby_prtfl_full_hedge.pos_grp_cvar, 
                "pos_cvar" : lby_prtfl_full_hedge.pos_cvar, 
                "cvar_scenarios" : lby_prtfl_full_hedge.cvar_scenarios, 
                "pv01_delta_cvar_attr" : lby_prtfl_full_hedge.pv01_delta_cvar_attr
              },                 
            "lby_prtfl_proxy_hedge" : 
              {
                "positions" : lby_prtfl_proxy_hedge.positions, 
                "cashflows" : lby_prtfl_proxy_hedge.cashflows, 
                "npv" : lby_prtfl_proxy_hedge.pv, 
                "spot_delta" : lby_prtfl_proxy_hedge.spot_delta, 
                "fwd_delta" : lby_prtfl_proxy_hedge.fwd_delta, 
                "spot_gamma" : lby_prtfl_proxy_hedge.spot_gamma, 
                "sim_pos_pnl" : lby_prtfl_proxy_hedge.sim_pos_pnl, 
                "sim_pos_grp_pnl" : lby_prtfl_proxy_hedge.sim_pos_grp_pnl, 
                "sim_prtfl_pnl" : lby_prtfl_proxy_hedge.sim_prtfl_pnl, 
                "prtfl_cvar" : lby_prtfl_proxy_hedge.prtfl_cvar, 
                "pos_grp_cvar" : lby_prtfl_proxy_hedge.pos_grp_cvar, 
                "pos_cvar" : lby_prtfl_proxy_hedge.pos_cvar, 
                "cvar_scenarios" : lby_prtfl_proxy_hedge.cvar_scenarios, 
                "pv01_delta_cvar_attr" : lby_prtfl_proxy_hedge.pv01_delta_cvar_attr
              }
            }                            
                                
def create_parquet(
    batch_df_dict : Dict[str, pd.DataFrame]   
    ) -> None :

    PROJECT_ROOT = ""
    STREAMLIT_APP_DATA_PATH = ""
    
    for parent in Path(__file__).parents:
      if (parent / "src").exists():
         PROJECT_ROOT = parent   
         STREAMLIT_APP_DATA_PATH = parent / "streamlit_app_data"
         break        
    
    os.chdir(STREAMLIT_APP_DATA_PATH)
    os.makedirs("parquet_files", exist_ok = True)    
        
    data_dict = batch_df_dict
        
    # curve data 
    for key, df in data_dict["curve"].items():
        df.to_parquet(f"parquet_files/curve_{key}.parquet")

    # mc scenarios data 
    for key, df in data_dict["scenarios"].items():
        df.to_parquet(f"parquet_files/scenarios_{key}.parquet")

    # liability position data 
    for key, df in data_dict["lby_pos"].items():
        df.to_parquet(f"parquet_files/lby_pos_{key}.parquet")

    # liability hedge data 
    for key, df in data_dict["lby_hedge"].items():
        df.to_parquet(f"parquet_files/lby_hedge_{key}.parquet")

    # portfolios data 
    portfolio_keys = [
        "lby_prtfl",
        "lby_prtfl_full_hedge",
        "lby_prtfl_proxy_hedge"
    ]

    for name in portfolio_keys:
        for key, df in data_dict[name].items():
            df.to_parquet(f"parquet_files/{name}_{key}.parquet")