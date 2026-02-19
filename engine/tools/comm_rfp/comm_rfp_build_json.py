"""
State-based BOQ assembly module.

This module handles the assembly of extracted PDF content into a
structured BOQ object that conforms to schema.json.

The assembly process uses a state machine approach:
1. Initialize with empty BOQ structure
2. Process lines sequentially
3. Maintain current state (division, grouping)
4. Transition state on division/grouping detection
5. Add line items to current grouping

Key concepts:
- "UNSPECIFIED" division/grouping used when structure is missing
- IDs are auto-generated if not present in the document
- Totals stored as strings per schema

Usage:
    from build_json import BOQBuilder
    from parse_pdf import extract_lines_from_pdf
    
    lines = extract_lines_from_pdf("boq.pdf")
    builder = BOQBuilder()
    boq = builder.build(lines)
    print(boq.to_json())
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional

from tools.comm_rfp.comm_rfp_models import BOQ, Division, Grouping, SubGrouping, LineItem
from tools.comm_rfp.comm_rfp_parse_pdf import ExtractedLine
# spec_loader removed — not needed in monorepo


# Constants for fallback values
UNSPECIFIED_DIVISION = "UNSPECIFIED"
UNSPECIFIED_GROUPING = "UNSPECIFIED"
UNSPECIFIED_SUB_GROUPING = "GENERAL"


class ParserState(Enum):
    """Parser state machine states."""
    INITIAL = auto()
    IN_DIVISION = auto()
    IN_GROUPING = auto()
    IN_ITEMS = auto()


@dataclass
class ParseContext:
    """
    Maintains the current parsing context/state.

    This dataclass tracks the current position in the BOQ hierarchy
    during sequential line processing.

    Attributes:
        state: Current parser state
        current_division: Currently active division (or None)
        current_grouping: Currently active grouping (or None)
        current_sub_grouping: Currently active sub-grouping (or None)
        item_counter: Counter for generating item IDs
    """
    state: ParserState = ParserState.INITIAL
    current_division: Optional[Division] = None
    current_grouping: Grouping | None = None
    current_sub_grouping: SubGrouping | None = None
    item_counter: int = 0

    def generate_item_id(self) -> str:
        """Generate a unique item ID."""
        self.item_counter += 1
        return f"ITM-{self.item_counter:04d}"


@dataclass
class ParsedItemData:
    """
    Intermediate representation of a parsed item line.

    Attributes:
        item_id: Extracted or generated item ID
        item_description: Item description
        item_quantity: Parsed quantity value
        item_unit: Unit of measurement
        item_rate: Unit rate/price
        raw_line: Original line text for debugging
    """
    item_id: str | None = None
    item_description: str | None = None
    item_quantity: float | None = None
    item_unit: str | None = None
    item_rate: float | None = None
    raw_line: str = ""


class BOQBuilder:
    """
    Builder for constructing BOQ objects from extracted PDF lines.

    The builder uses a state machine approach to process lines
    sequentially, detecting divisions, groupings, and items based
    on regex patterns defined in the extraction spec.

    Attributes:
        spec: Extraction specification containing rules
        boq: The BOQ being constructed
        context: Current parsing context/state
    """

    def __init__(self, spec: ExtractionSpec | None = None):
        """
        Initialize the BOQ builder.

        Args:
            spec: Extraction specification, or None for default rules
        """
        self.spec = spec or get_default_spec()
        self.boq: BOQ | None = None
        self.context: ParseContext | None = None

    def build(self, lines: list[ExtractedLine]) -> BOQ:
        """
        Build a BOQ from extracted PDF lines.

        This is the main entry point for BOQ construction. It processes
        lines sequentially, maintaining state and building the hierarchy.

        Args:
            lines: List of extracted lines from PDF

        Returns:
            Constructed BOQ object conforming to schema.json
        """
        # Initialize fresh state
        self.boq = BOQ()
        self.context = ParseContext()

        # First pass: extract project metadata from all lines
        self._extract_metadata(lines)

        # Second pass: build division/grouping/item hierarchy
        for line in lines:
            if line.is_empty:
                continue
            self._process_line(line)

        # Ensure we have at least an unspecified division
        self._ensure_structure()

        # Calculate totals (stored as strings per schema)
        self._calculate_totals()

        return self.boq

    def _extract_metadata(self, lines: list[ExtractedLine]) -> None:
        """
        Extract project-level metadata from lines.

        Searches all lines for project name and contractor
        information using metadata rules.

        Args:
            lines: All extracted lines
        """
        for rule in self.spec.metadata_rules:
            for line in lines:
                value = rule.extract(line.stripped)
                if value:
                    if rule.rule_type == RuleType.PROJECT_NAME:
                        self.boq.project_name = value.strip()
                    elif rule.rule_type == RuleType.CONTRACTOR:
                        self.boq.contractor = value.strip()
                    break  # Found a match for this rule type

    def _process_line(self, line: ExtractedLine) -> None:
        """
        Process a single line and update state accordingly.

        The processing order is:
        1. Check if line starts a new division
        2. Check if line starts a new grouping
        3. Check if line is an item
        4. Otherwise, ignore (header/footer/notes)

        Args:
            line: The line to process
        """
        text = line.stripped

        # Try division detection
        if self._try_division(text):
            return

        # Try grouping detection
        if self._try_grouping(text):
            return

        # Try item detection
        if self._try_item(text):
            return

        # Line doesn't match any pattern - could be continuation or noise
        # TODO: Handle multi-line descriptions
        # TODO: Detect and extract subtotals

    def _try_division(self, text: str) -> bool:
        """
        Attempt to parse line as a division header.

        Args:
            text: Line text to check

        Returns:
            True if line was recognized as division header
        """
        for rule in self.spec.division_rules:
            match = rule.match(text)
            if match:
                # Extract division name
                try:
                    division_name = match.group(2) if match.lastindex >= 2 else match.group(1)
                except IndexError:
                    division_name = match.group(0)

                # Create new division
                division = Division(
                    division_name=division_name.strip(),
                    grouping=[]
                )

                self.boq.divisions.append(division)
                self.context.current_division = division
                self.context.current_grouping = None
                self.context.current_sub_grouping = None
                self.context.state = ParserState.IN_DIVISION

                return True

        return False

    def _try_grouping(self, text: str) -> bool:
        """
        Attempt to parse line as a grouping header.

        Args:
            text: Line text to check

        Returns:
            True if line was recognized as grouping header
        """
        for rule in self.spec.group_rules:
            match = rule.match(text)
            if match:
                # Extract grouping name
                try:
                    grouping_name = match.group(2) if match.lastindex >= 2 else match.group(1)
                except IndexError:
                    grouping_name = match.group(0)

                # Ensure we have a division
                self._ensure_division()

                # Create new grouping
                grouping = Grouping(
                    grouping_name=grouping_name.strip(),
                    sub_groupings=[]
                )

                self.context.current_division.grouping.append(grouping)
                self.context.current_grouping = grouping
                self.context.current_sub_grouping = None
                self.context.state = ParserState.IN_GROUPING

                return True

        return False

    def _try_item(self, text: str) -> bool:
        """
        Attempt to parse line as a BOQ line item.

        Args:
            text: Line text to check

        Returns:
            True if line was recognized as an item
        """
        for rule in self.spec.item_rules:
            match = rule.match(text)
            if match:
                # Parse item data from match groups
                item_data = self._parse_item_match(match, text)

                # Ensure we have division, grouping, and sub-grouping
                self._ensure_division()
                self._ensure_grouping()
                self._ensure_sub_grouping()

                # Create item with generated ID if not present
                item = LineItem(
                    item_id=item_data.item_id or self.context.generate_item_id(),
                    item_description=item_data.item_description,
                    item_quantity=item_data.item_quantity,
                    item_unit=item_data.item_unit,
                    item_rate=item_data.item_rate
                )

                self.context.current_sub_grouping.line_items.append(item)
                self.context.state = ParserState.IN_ITEMS

                return True

        return False

    def _parse_item_match(self, match: re.Match, raw_line: str) -> ParsedItemData:
        """
        Parse item data from a regex match.

        Args:
            match: Regex match object
            raw_line: Original line text

        Returns:
            ParsedItemData with extracted values
        """
        groups = match.groups()
        data = ParsedItemData(raw_line=raw_line)

        # Standard pattern: code/id, description, quantity, unit, rate
        if len(groups) >= 5:
            data.item_id = groups[0]
            data.item_description = groups[1].strip() if groups[1] else None
            data.item_quantity = self._parse_number(groups[2])
            data.item_unit = groups[3]
            data.item_rate = self._parse_number(groups[4])
        elif len(groups) >= 4:
            # id, description, quantity, unit
            data.item_id = groups[0]
            data.item_description = groups[1].strip() if groups[1] else None
            data.item_quantity = self._parse_number(groups[2])
            data.item_unit = groups[3]
        elif len(groups) >= 3:
            # id, description, unit (lump sum)
            data.item_id = groups[0]
            data.item_description = groups[1].strip() if groups[1] else None
            data.item_unit = groups[2]
            data.item_quantity = 1.0  # Lump sum = quantity of 1
        elif len(groups) >= 2:
            data.item_id = groups[0]
            data.item_description = groups[1].strip() if groups[1] else None
        elif len(groups) >= 1:
            data.item_description = groups[0].strip() if groups[0] else None

        return data

    def _parse_number(self, value: str | None) -> float | None:
        """
        Parse a number string, handling common formats.

        Args:
            value: String representation of number

        Returns:
            Parsed float value, or None if unparseable
        """
        if not value:
            return None

        try:
            # Remove thousands separators (commas)
            cleaned = value.replace(",", "")
            return float(cleaned)
        except (ValueError, AttributeError):
            return None

    def _ensure_division(self) -> None:
        """Ensure a current division exists, creating UNSPECIFIED if needed."""
        if self.context.current_division is None:
            division = Division(
                division_name=UNSPECIFIED_DIVISION,
                grouping=[]
            )
            self.boq.divisions.append(division)
            self.context.current_division = division

    def _ensure_grouping(self) -> None:
        """Ensure a current grouping exists, creating UNSPECIFIED if needed."""
        if self.context.current_grouping is None:
            grouping = Grouping(
                grouping_name=UNSPECIFIED_GROUPING,
                sub_groupings=[]
            )
            self.context.current_division.grouping.append(grouping)
            self.context.current_grouping = grouping

    def _ensure_sub_grouping(self) -> None:
        """Ensure a current sub-grouping exists, creating GENERAL if needed."""
        if self.context.current_grouping is None:
            self._ensure_grouping()
        if self.context.current_sub_grouping is None:
            sub_grouping = SubGrouping(
                sub_grouping_name=UNSPECIFIED_SUB_GROUPING,
                line_items=[]
            )
            self.context.current_grouping.sub_groupings.append(sub_grouping)
            self.context.current_sub_grouping = sub_grouping

    def _ensure_structure(self) -> None:
        """Ensure the BOQ has at least one division with one grouping."""
        if not self.boq.divisions:
            self._ensure_division()
            self._ensure_grouping()
            self._ensure_sub_grouping()

    def _calculate_totals(self) -> None:
        """
        Calculate division and project totals.

        Totals are only calculated if item_rate and item_quantity are available
        for items. Totals are stored as strings per schema.

        Note:
            This is a simple calculation. Real BOQs may have complex
            pricing structures (discounts, markups, taxes) that require
            custom calculation logic.
        """
        # TODO: Add support for explicit total extraction from PDF
        # TODO: Add validation that extracted totals match calculated totals

        project_total = 0.0
        has_any_total = False

        for division in self.boq.divisions:
            division_total = 0.0
            division_has_total = False

            for grouping in division.grouping:
                for sub_grouping in grouping.sub_groupings:
                    for item in sub_grouping.line_items:
                        if item.item_quantity is not None and item.item_rate is not None:
                            item_total = item.item_quantity * item.item_rate
                            division_total += item_total
                            division_has_total = True

            if division_has_total:
                # Store as string per schema
                division.division_total_amount = f"{division_total:.2f}"
                project_total += division_total
                has_any_total = True

        if has_any_total:
            # Store as string per schema
            self.boq.project_total_amount = f"{project_total:.2f}"


def build_boq_from_lines(
    lines: list[ExtractedLine],
    spec: ExtractionSpec | None = None
) -> BOQ:
    """
    Convenience function to build a BOQ from extracted lines.

    Args:
        lines: Extracted PDF lines
        spec: Optional extraction specification

    Returns:
        Constructed BOQ object
    """
    builder = BOQBuilder(spec)
    return builder.build(lines)


# TODO: Add support for multi-line item descriptions
# TODO: Add support for nested groupings (sub-groups)
# TODO: Add confidence scoring for extracted values
# TODO: Add validation against expected BOQ structure
# TODO: Add support for alternate BOQ formats (tabular vs. hierarchical)
# TODO: Multi-PDF aggregation support
