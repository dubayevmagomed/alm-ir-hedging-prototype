from __future__ import annotations
from typing import Dict
from pathlib import Path
from datetime import datetime 
from dataclasses import dataclass
from rateslib import IRS as rlIRS, Curve as rlCurve, Solver as rlSolver, dcf as rldcf, Frequency as rlFrequency, calendars as rlCalendars

class SolverError(Exception):
  pass
  
@dataclass 
class SolverConfig:
    """Data container for configuration of curve construction"""       
    currency : str
    calendar : str = 'tgt'
    interpolation : str = "spline"
    day_count_convention : str = 'act360'
    leg_convention : str = '30e360'
    leg_frequency : int = 12
    float_leg_convention : str = 'act360'
    float_leg_frequency : int = 6
      
class SwapCurveSolver:
    """
    Bootstraps an interest rate curve from market swap rates using RatesLib.

    This class constructs and calibrates a discount curve by fitting a set of
    interest rate swap (IRS) instruments to observed market rates. The calibration
    is performed using the RatesLib solver framework, ensuring internal consistency
    between instrument pricing and curve discount factors.

    Parameters
    ----------
    solver_id : str
        Identifier for the curve (e.g. 'EURSWAP').

    curve_date : datetime
        Valuation date for the curve.

    instrument_rates : Dict[str, float]
        Mapping of tenor to market par swap rate (in percent or decimal depending on convention).
        Example: {'2Y': 0.0295, '5Y': 0.0310, ...}

    Attributes
    ----------
    instruments : List[IRS]
        List of calibration instruments (RatesLib IRS objects).

    curve_nodes : pd.DataFrame
        Curve node structure containing tenors and corresponding discount factors.

    curve : rateslib Curve
        Calibrated discount curve object.

    solver : rateslib Solver
        Solver instance used for calibration and risk calculations.

    Methods
    -------
    build_instruments()
        Constructs IRS instruments used for curve calibration.

    build_curve_nodes()
        Initializes curve nodes prior to calibration.

    solve_curve()
        Runs the solver to calibrate discount factors to market rates.

    Notes
    -----
    - The curve is calibrated to exactly reproduce market swap rates.
    - The solver object is required for computing sensitivities (PV01, gamma, etc.).
    - The resulting curve can be reused across multiple positions for consistent pricing.
    """
    def __init__( 
       self, 
       solver_id : str,             
       curve_date : datetime, 
       instrument_rates : Dict[str, float] | None = None
      ) -> None:
      
     self.solver_id = solver_id
     self.curve_date = curve_date    
     self.config = SolverConfig("eur")
        
    # the tenors supplied in instrument_rates argument define nodes of the curve   
    # a instrument is created for each node. Curve instruments are used to solve for disc factors at each node
     self.instrument_rates = self._check_instrument_rates(instrument_rates) 
        
    # for each node/tenor, create a RatesLib instument for calibration  
     self.instruments = self._create_instruments()
    
    # create the Curve and Solver objects from RatesLib, and bootsrap/solve disc factors
     self.curve = self._create_curve()
     self.solver = self._solve_curve() 
     self.is_solved = True
     
    def _tenor_to_years( 
      self, 
      tenor : str
      ) -> float: 
      
      if tenor.endswith("Y"):
        return float(tenor[:-1])
      if tenor.endswith("M"):
        return float(tenor[:-1]) / 12
      if tenor.endswith("W"):
        return float(tenor[:-1]) / 52
      if tenor.endswith("D"):
        return float(tenor[:-1]) / 365
      raise ValueError(f"Unknown tenor format: {tenor}")
    
    def _check_instrument_rates(
      self, 
      instrument_rates : Dict[str, float] | None = None                  
      ) -> Dict[str, float]:
    
      if not instrument_rates:
          raise SolverError('curve requires instrument rates for each node')
      
      # ensure chronological order in the dict, 1y :, 2y :, ...    
      _instrument_rates_sorted = dict( 
                                   sorted( instrument_rates.items()
                                         , key = lambda tenor_str: self._tenor_to_years( tenor_str[0] )
                                         ) 
                                     )
      
      return _instrument_rates_sorted
  
    def _create_instruments( 
      self 
      ) -> Dict[str, rlIRS]:
      
      _instruments = {}            
      
      # the tenor is mapped/translated to maturity of corresponding inst., see termination= argument below
      for tenor in self.instrument_rates.keys():  
        try:
            _instruments[tenor] = rlIRS(
                effective = self.curve_date
              , calendar  = rlCalendars.get(self.config.calendar) 
              , currency  = self.config.currency
              , curves    = self.solver_id
              , termination = tenor 
              , convention  = self.config.leg_convention 
              , leg2_convention = self.config.float_leg_convention          
              , frequency = rlFrequency.Months(self.config.leg_frequency, None)
              , leg2_frequency = rlFrequency.Months(self.config.float_leg_frequency, None)
        )
        except Exception as e:
          raise SolverError(f"error creating instrument for tenor node {tenor}: {e}")      
     
      return _instruments
  
    def _create_curve( 
      self 
      ) -> rlCurve:
      
      # for each tenor: corresponding disc. factor/node date is set to final pmt dt of the underlying IRS i.e. tenor node -> tenor date
      tenor_dates = { tenor : max( max(instrument.leg1.schedule.pschedule) 
                                 , max(instrument.leg2.schedule.pschedule) 
                                 )  
                                 for tenor, instrument in self.instruments.items() 
                    }
      
      self.curve_nodes = { tenor : ( tenor_date 
                              , rldcf( convention = self.config.day_count_convention, 
                                            start = self.curve_date, 
                                              end = tenor_date
                                     ), 
                              )
                       for tenor, tenor_date in tenor_dates.items()                     
                    }
      
      # set the initial disc. factor value to 1      
      _nodes = { self.curve_date : 1.0, 
                  **{ tenor_dates[tenor] : 1.0 for tenor in self.instruments.keys() }
               }          
      
      return rlCurve(
               id = self.solver_id, 
               nodes = _nodes, 
               calendar = self.config.calendar, 
               convention = self.config.day_count_convention, 
               interpolation = self.config.interpolation
            )  
        
    def _solve_curve(
          self
       ) -> None:

     tenors = list(self.instrument_rates.keys())
     return rlSolver(
            curves = [self.curve],
            instruments = [self.instruments[tenor] for tenor in tenors],
            instrument_labels = [tenor for tenor in tenors],
            s = [self.instrument_rates[tenor] for tenor in tenors],
            id = self.solver_id
        )
