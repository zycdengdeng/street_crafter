from .waymo_processor import WaymoPointCloudProcessor
from .pandaset_processor import PandasetPointCloudProcessor
from .base_processor import BasePointCloudProcessor
from street_gaussian.config import cfg

PointCloudProcessorType = {
    "Waymo": WaymoPointCloudProcessor,
    "Pandaset": PandasetPointCloudProcessor
}


def getPointCloudProcessor() -> BasePointCloudProcessor:
    return PointCloudProcessorType[cfg.data.type]()
