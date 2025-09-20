from duckpipe.calculator.AirportDistanceCalculator import AirportDistanceCalculator
from duckpipe.calculator.BusStopDistanceCalculator import BusStopDistanceCalculator
from duckpipe.calculator.Clustering import Clustering
from duckpipe.calculator.CoastlineDistanceCalculator import CoastlineDistanceCalculator
from duckpipe.calculator.CoordinateCalculator import CoordinateCalculator
from duckpipe.calculator.LanduseCalculator import LanduseCalculator
from duckpipe.calculator.MainRoadDistanceCalculator import MainRoadDistanceCalculator
from duckpipe.calculator.MainRoadLLWCalculator import MainRoadLLWCalculator
from duckpipe.calculator.MDLDistanceCalculator import MDLDistanceCalculator
from duckpipe.calculator.PortDistanceCalculator import PortDistanceCalculator
from duckpipe.calculator.RailstationDistanceCalculator import RailstationDistanceCalculator
from duckpipe.calculator.RelativeElevationCalculator import RelativeElevationCalculator
from duckpipe.calculator.RiverDistanceCalculator import RiverDistanceCalculator
from duckpipe.calculator.RoadDistanceCalculator import RoadDistanceCalculator
from duckpipe.calculator.RoadLLWCalculator import RoadLLWCalculator
from duckpipe.calculator.Worker import Worker
from duckpipe.calculator._IntersectingOACalculator import IntersectingOACalculator
from duckpipe.calculator.HouseYearCalculator import HouseYearCalculator


class CalculatorMixin(
    AirportDistanceCalculator, 
    BusStopDistanceCalculator,
    Clustering,
    CoastlineDistanceCalculator,
    CoordinateCalculator,
    LanduseCalculator,
    MainRoadDistanceCalculator,
    MainRoadLLWCalculator, 
    MDLDistanceCalculator,
    PortDistanceCalculator,
    RailstationDistanceCalculator,
    RelativeElevationCalculator,
    RiverDistanceCalculator,
    RoadDistanceCalculator,
    RoadLLWCalculator,
    Worker,
    IntersectingOACalculator,
    HouseYearCalculator
):
    pass

