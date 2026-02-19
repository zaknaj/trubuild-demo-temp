"""
Pydantic models for BOQ (Bill of Quantities) data structures.

These models strictly conform to schema.json and enforce validation
with extra="forbid" to prevent schema drift.

The schema hierarchy is:
    BOQ (project-level)
    └── Division (major scope/trade)
        └── Grouping (subcategory)
            └── SubGrouping (work category)
                └── LineItem (line item)
"""

from __future__ import annotations

from typing import Optional
from pydantic import BaseModel, Field, ConfigDict


class LineItem(BaseModel):
    """
    Represents a single measurable BOQ line item.

    A line item is the most granular unit in the BOQ hierarchy, representing
    a specific work element, material, or service with associated
    quantity and pricing information.
    """

    model_config = ConfigDict(extra="forbid")

    item_id: Optional[str] = Field(
        default=None,
        description="A unique identifier for the item (system-generated)"
    )

    item_description: Optional[str] = Field(
        default=None,
        description="A detailed description of the work, material, or service for this item"
    )

    item_quantity: Optional[float] = Field(
        default=None,
        description="The measured quantity of the item"
    )

    item_unit: Optional[str] = Field(
        default=None,
        description="The unit of measurement (e.g., m, m², m³, pcs, lot)"
    )

    item_rate: Optional[float] = Field(
        default=None,
        description="The cost per unit of measurement"
    )

    item_rate_raw: Optional[str] = Field(
        default=None,
        description="Raw rate value from Excel (e.g., 'Included', 'Excluded', 'N/A')"
    )

    item_amount: Optional[float] = Field(
        default=None,
        description="The total amount for this line item (quantity × rate), extracted from the BOQ"
    )

    item_amount_raw: Optional[str] = Field(
        default=None,
        description="Raw amount value from Excel (e.g., 'Included', 'Excluded', 'N/A')"
    )


class SubGrouping(BaseModel):
    """
    Represents a sub-grouping within a grouping.

    A sub-grouping categorizes line items within a grouping, such as
    'Site clearance; including all ancillary works as necessary' or 
    'Anti-Termite Treatment'. Sub-groupings do not have a total amount.
    """

    model_config = ConfigDict(extra="forbid")

    sub_grouping_name: Optional[str] = Field(
        default=None,
        description="The name of the sub-grouping (e.g., 'Site clearance', 'Anti-Termite Treatment')"
    )

    line_items: list[LineItem] = Field(
        default_factory=list,
        description="Each item represents a single measurable BOQ line item"
    )


class Grouping(BaseModel):
    """
    Represents a grouping within a division.

    A grouping is a subcategory within a division, usually representing
    a logical breakdown such as work type, location, or construction element.
    Groupings contain sub-groupings which in turn contain line items.
    """

    model_config = ConfigDict(extra="forbid")

    grouping_name: Optional[str] = Field(
        default=None,
        description="The descriptive name of the group (e.g., Site Preparation, Earthworks)"
    )

    grouping_total_amount: Optional[str] = Field(
        default=None,
        description="The total cost of all line items in this grouping"
    )

    sub_groupings: list[SubGrouping] = Field(
        default_factory=list,
        description="Sub-groupings within this grouping that categorize line items"
    )


class Division(BaseModel):
    """
    Represents a major scope or trade division.

    A division represents a major scope or trade, commonly aligned with
    BOQ sections or standards (e.g., SITE WORK, CONCRETE WORK, MASONRY).
    """

    model_config = ConfigDict(extra="forbid")

    division_name: Optional[str] = Field(
        default=None,
        description="The name of the division or trade (e.g., SITE WORK, CONCRETE WORK, MASONRY)"
    )

    division_total_amount: Optional[str] = Field(
        default=None,
        description="The total cost of all groups and items under this division"
    )

    grouping: list[Grouping] = Field(
        default_factory=list,
        description="Each group is a subcategory within a division"
    )


class BOQ(BaseModel):
    """
    Root model representing a complete Bill of Quantities document.

    This is the top-level container holding project metadata and all
    divisions. The structure strictly conforms to schema.json.

    Attributes:
        project_name: The official name or title of the construction project
        contractor: The name of the contractor or bidder
        project_total_amount: The total contract value of the project (as string)
        divisions: List of divisions containing groupings and line items
    """

    model_config = ConfigDict(extra="forbid")

    project_name: Optional[str] = Field(
        default=None,
        description="The official name or title of the construction project as stated in the BOQ document"
    )

    contractor: Optional[str] = Field(
        default=None,
        description="The name of the contractor, bidder, or company responsible for pricing or executing the BOQ"
    )

    project_total_amount: Optional[str] = Field(
        default=None,
        description="The total contract value of the project"
    )

    divisions: list[Division] = Field(
        default_factory=list,
        description="Each division represents a major scope or trade"
    )

    def to_json(self, indent: int = 2) -> str:
        """
        Serialize the BOQ to a JSON string.

        Args:
            indent: Number of spaces for indentation (default: 2)

        Returns:
            JSON string representation of the BOQ
        """
        return self.model_dump_json(indent=indent)

    def to_dict(self) -> dict:
        """
        Convert the BOQ to a dictionary.

        Returns:
            Dictionary representation of the BOQ
        """
        return self.model_dump()
