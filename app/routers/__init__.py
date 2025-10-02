__all__ = [
    "root_router",
    "customer_v1_router"
]

from .v1 import customer_router as customer_v1_router
from .root import root_router
