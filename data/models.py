from datetime import datetime
from typing import Optional, List
from pydantic import BaseModel, Field


class ProductData(BaseModel):
    """Product information model"""
    product_id: str = Field(..., description="Unique product identifier")
    name: str = Field(..., description="Product name")
    category: str = Field(..., description="Product category")
    brand: str = Field(..., description="Product brand")
    price: float = Field(..., description="Current product price")
    created_at: datetime = Field(default_factory=datetime.now)


class SalesData(BaseModel):
    """Sales transaction model"""
    product_id: str = Field(..., description="Product identifier")
    date: datetime = Field(..., description="Sales date")
    quantity_sold: int = Field(..., description="Number of units sold")
    revenue: float = Field(..., description="Total revenue from sales")
    customer_count: int = Field(default=1, description="Number of unique customers")
    discount_applied: float = Field(default=0.0, description="Discount percentage applied")
    channel: str = Field(default="online", description="Sales channel (online, store, mobile)")


class MarketData(BaseModel):
    """External market factors model"""
    date: datetime = Field(..., description="Date of market data")
    market_index: float = Field(..., description="General market performance index")
    competitor_price_avg: Optional[float] = Field(None, description="Average competitor price")
    search_volume: Optional[int] = Field(None, description="Search volume for product category")
    social_mentions: Optional[int] = Field(None, description="Social media mentions")
    economic_indicator: Optional[float] = Field(None, description="Economic health indicator")


class SeasonalData(BaseModel):
    """Seasonal and temporal features model"""
    date: datetime = Field(..., description="Date")
    is_weekend: bool = Field(..., description="Whether date is weekend")
    is_holiday: bool = Field(..., description="Whether date is a holiday")
    month: int = Field(..., description="Month (1-12)")
    quarter: int = Field(..., description="Quarter (1-4)")
    day_of_year: int = Field(..., description="Day of year (1-365/366)")
    week_of_year: int = Field(..., description="Week of year (1-52/53)")
    season: str = Field(..., description="Season (spring, summer, fall, winter)")


class PredictionRequest(BaseModel):
    """Request model for trend predictions"""
    product_id: str = Field(..., description="Product to predict trends for")
    days_ahead: int = Field(default=30, description="Number of days to predict ahead")
    include_confidence: bool = Field(default=True, description="Include confidence intervals")
    model_type: Optional[str] = Field(default="ensemble", description="Model type to use")


class PredictionResponse(BaseModel):
    """Response model for trend predictions"""
    product_id: str = Field(..., description="Product identifier")
    predictions: List[dict] = Field(..., description="List of daily predictions")
    confidence_intervals: Optional[List[dict]] = Field(None, description="Confidence intervals")
    trend_direction: str = Field(..., description="Overall trend direction (up, down, stable)")
    trend_strength: float = Field(..., description="Trend strength score (0-1)")
    model_accuracy: float = Field(..., description="Model accuracy score")
    generated_at: datetime = Field(default_factory=datetime.now)


class ModelMetrics(BaseModel):
    """Model performance metrics"""
    model_name: str = Field(..., description="Name of the model")
    mae: float = Field(..., description="Mean Absolute Error")
    mse: float = Field(..., description="Mean Squared Error")
    rmse: float = Field(..., description="Root Mean Squared Error")
    mape: float = Field(..., description="Mean Absolute Percentage Error")
    r2_score: float = Field(..., description="R-squared score")
    training_date: datetime = Field(default_factory=datetime.now)


class TrendAlert(BaseModel):
    """Alert model for significant trend changes"""
    product_id: str = Field(..., description="Product identifier")
    alert_type: str = Field(..., description="Type of alert (spike, drop, anomaly)")
    severity: str = Field(..., description="Severity level (low, medium, high)")
    message: str = Field(..., description="Alert message")
    confidence: float = Field(..., description="Confidence score (0-1)")
    detected_at: datetime = Field(default_factory=datetime.now)