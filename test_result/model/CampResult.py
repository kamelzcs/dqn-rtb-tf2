from pydantic import BaseModel


class CampResult(BaseModel):
    imp: float
    click: float
    cost: float
    wr: float
    ecpc: float
    ecpi: float


