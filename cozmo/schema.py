"""Internal plan schema (v0) until Cozmo publishes theirs.

Every metric field is an Interval so photo-tier honesty is structural, not optional.
Stitch/adjacency are filled by the door-graph when more than one photo folder is run.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

SCHEMA_VERSION = "cozmo.plan.v0"


class Interval(BaseModel):
    value: float
    ci_low: float
    ci_high: float
    method: str
    unit: str = "m"

    @classmethod
    def measured(cls, value: float, rel: float, method: str, unit: str = "m") -> Interval:
        span = abs(value) * rel
        return cls(
            value=value,
            ci_low=value - span,
            ci_high=value + span,
            method=method,
            unit=unit,
        )


class Point2(BaseModel):
    x: float
    y: float


class Wall(BaseModel):
    id: str
    start: Point2
    end: Point2
    length: Interval


class Opening(BaseModel):
    id: str
    kind: Literal["door", "window"]
    wall_id: str
    width: Interval
    t_start: float = Field(ge=0.0, le=1.0)
    t_end: float = Field(ge=0.0, le=1.0)
    evidence: str = ""


class RoomPlan(BaseModel):
    id: str
    source_folder: str
    polygon: list[Point2]
    walls: list[Wall]
    openings: list[Opening]
    ceiling_height: Interval
    floor_area: Interval
    notes: list[str] = Field(default_factory=list)
    backend: str = ""


class Adjacency(BaseModel):
    room_a: str
    room_b: str
    opening_a: str
    opening_b: str


class PropertyPlan(BaseModel):
    schema_version: str = SCHEMA_VERSION
    tier: Literal["photos", "video", "lidar"] = "photos"
    stitch: Literal["unstitched", "door_graph", "walk_graph"] = "unstitched"
    rooms: list[RoomPlan]
    adjacency: list[Adjacency] = Field(default_factory=list)
    disclosures: list[str] = Field(default_factory=list)
    extra: dict[str, Any] = Field(default_factory=dict)
