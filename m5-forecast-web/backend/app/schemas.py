from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

Level = Literal['L1', 'L2', 'L3', 'L9', 'L12']
Model = Literal['TFT', 'LSTM', 'ARIMA', 'PROPHET']
Calibration = Literal['RAW', 'STORE_CAL', 'GLOBAL_CAL', 'SERIES_CAL']


class ForecastRequest(BaseModel):
    model_config = ConfigDict(extra='forbid')
    model: Model = 'TFT'
    level: Level = 'L1'
    series_id: str = Field(default='TOTAL', min_length=1, max_length=100)
    origin_day: int = 1913
    horizon: int = Field(default=28, ge=28, le=28)
    strategy: Literal['bottom_up', 'direct'] = 'bottom_up'
    calibration: Calibration = 'RAW'


class CompareRequest(ForecastRequest):
    models: list[Model] = Field(default=['TFT', 'LSTM', 'ARIMA', 'PROPHET'], min_length=1, max_length=4)


class ForecastResponse(BaseModel):
    model: str
    execution_mode: Literal['artifact_replay']
    strategy: str
    calibration: str
    level: Level
    series_id: str
    origin_day: int
    horizon: int
    dates: list[str]
    historical_dates: list[str]
    historical_sales: list[float]
    predictions: list[float]
    actuals: list[float] | None
    metrics: dict[str, float | None]
    metadata: dict
