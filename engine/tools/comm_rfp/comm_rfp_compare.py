"""
Vendor BOQ Comparison Module.

Compares multiple vendor submissions against a template BOQ,
calculating totals and identifying lowest/highest bidders.

Usage:
    from tools.comm_rfp.comm_rfp_compare import compare_vendor_boqs, ComparisonReport
    
    report = compare_vendor_boqs(
        template_path="template.xlsx",
        vendor_files={
            "Vendor A": "vendor_a.xlsx",
            "Vendor B": "vendor_b.xlsx",
        }
    )
    
    # Or compare already-extracted JSON files
    report = compare_vendor_jsons(
        template_json="template.json",
        vendor_jsons={
            "Vendor A": "vendor_a.json",
            "Vendor B": "vendor_b.json",
        }
    )
"""

import json
from dataclasses import dataclass, field
from typing import Optional, Iterator
from pathlib import Path


def _iter_grouping_items(grouping: dict) -> Iterator[dict]:
    """Yield line items from a grouping, supporting sub-groupings or legacy line_items."""
    if grouping.get("sub_groupings"):
        for sub in grouping.get("sub_groupings", []):
            for item in sub.get("line_items", []):
                yield item
    else:
        for item in grouping.get("line_items", []):
            yield item


def _iter_grouping_items_with_subgrouping(grouping: dict) -> Iterator[tuple[str, int, dict]]:
    """Yield (sub_grouping_name, index, item) for stable matching across sheets."""
    if grouping.get("sub_groupings"):
        for sub in grouping.get("sub_groupings", []):
            sub_name = sub.get("sub_grouping_name") or "GENERAL"
            for idx, item in enumerate(sub.get("line_items", [])):
                yield sub_name, idx, item
    else:
        for idx, item in enumerate(grouping.get("line_items", [])):
            yield "GENERAL", idx, item


def _upper(value: Optional[str]) -> str:
    """Safely uppercase optional strings."""
    return (value or "").upper()


@dataclass
class VendorLineItem:
    """A line item with vendor-specific pricing."""
    item_id: str
    item_description: str
    item_quantity: Optional[float]
    item_unit: Optional[str]
    item_rate: Optional[float]
    item_amount: Optional[float]  # quantity * rate
    
    @classmethod
    def from_dict(cls, data: dict) -> "VendorLineItem":
        qty = data.get("item_quantity")
        rate = data.get("item_rate")
        amount = data.get("item_amount")
        if amount is None and qty is not None and rate is not None:
            amount = qty * rate
        
        return cls(
            item_id=data.get("item_id", ""),
            item_description=data.get("item_description", ""),
            item_quantity=qty,
            item_unit=data.get("item_unit"),
            item_rate=rate,
            item_amount=amount,
        )


@dataclass
class LineItemComparison:
    """Comparison of a single line item across vendors."""
    item_id: str
    item_description: str
    item_quantity: Optional[float]
    item_unit: Optional[str]
    sub_grouping_name: Optional[str] = None
    
    # Vendor rates: {vendor_name: rate}
    vendor_rates: dict[str, Optional[float]] = field(default_factory=dict)
    # Vendor amounts: {vendor_name: amount}
    vendor_amounts: dict[str, Optional[float]] = field(default_factory=dict)
    
    # PTE benchmark (optional, not a vendor)
    pte_rate: Optional[float] = None
    pte_amount: Optional[float] = None
    
    # Flags
    lowest_bidder: Optional[str] = None
    highest_bidder: Optional[str] = None
    lowest_rate: Optional[float] = None
    highest_rate: Optional[float] = None
    rate_variance: Optional[float] = None  # % difference between lowest and highest
    
    def calculate_flags(self):
        """Calculate lowest/highest bidder flags."""
        valid_rates = {k: v for k, v in self.vendor_rates.items() if v is not None and v > 0}
        
        if not valid_rates:
            return
        
        self.lowest_rate = min(valid_rates.values())
        self.highest_rate = max(valid_rates.values())
        
        for vendor, rate in valid_rates.items():
            if rate == self.lowest_rate:
                self.lowest_bidder = vendor
            if rate == self.highest_rate:
                self.highest_bidder = vendor
        
        if self.lowest_rate is not None and self.lowest_rate > 0.01:
            raw = ((self.highest_rate - self.lowest_rate) / self.lowest_rate) * 100
            # Cap at 500% so one bad item (wrong unit, typo) doesn't blow the summary
            self.rate_variance = min(raw, 500.0)
    
    def to_dict(self) -> dict:
        d = {
            "item_id": self.item_id,
            "item_description": self.item_description,
            "item_quantity": self.item_quantity,
            "item_unit": self.item_unit,
            "sub_grouping_name": self.sub_grouping_name,
            "vendor_rates": self.vendor_rates,
            "vendor_amounts": self.vendor_amounts,
            "lowest_bidder": self.lowest_bidder,
            "highest_bidder": self.highest_bidder,
            "lowest_rate": self.lowest_rate,
            "highest_rate": self.highest_rate,
            "rate_variance_percent": round(self.rate_variance, 2) if self.rate_variance else None,
        }
        if self.pte_rate is not None or self.pte_amount is not None:
            d["pte_rate"] = self.pte_rate
            d["pte_amount"] = self.pte_amount
        return d


@dataclass
class GroupingComparison:
    """Comparison of a grouping across vendors."""
    grouping_name: str
    line_items: list[LineItemComparison] = field(default_factory=list)
    
    # Totals per vendor
    vendor_totals: dict[str, float] = field(default_factory=dict)
    pte_total: Optional[float] = None
    lowest_bidder: Optional[str] = None
    highest_bidder: Optional[str] = None
    
    def calculate_totals(self, vendors: list[str]):
        """Calculate grouping totals per vendor."""
        for vendor in vendors:
            total = sum(
                item.vendor_amounts.get(vendor, 0) or 0
                for item in self.line_items
            )
            self.vendor_totals[vendor] = total

        # PTE grouping total
        pte_sum = sum(item.pte_amount or 0 for item in self.line_items)
        if pte_sum > 0:
            self.pte_total = pte_sum
        
        valid_totals = {k: v for k, v in self.vendor_totals.items() if v > 0}
        if valid_totals:
            self.lowest_bidder = min(valid_totals, key=valid_totals.get)
            self.highest_bidder = max(valid_totals, key=valid_totals.get)
    
    def to_dict(self) -> dict:
        d = {
            "grouping_name": self.grouping_name,
            "vendor_totals": self.vendor_totals,
            "lowest_bidder": self.lowest_bidder,
            "highest_bidder": self.highest_bidder,
            "line_items": [item.to_dict() for item in self.line_items],
        }
        if self.pte_total is not None:
            d["pte_total"] = self.pte_total
        return d


@dataclass
class DivisionComparison:
    """Comparison of a division across vendors."""
    division_name: str
    groupings: list[GroupingComparison] = field(default_factory=list)
    
    # Extracted totals from BOQ (the "Total - Carried to Summary" amounts)
    vendor_totals: dict[str, float] = field(default_factory=dict)
    # Calculated totals (sum of line items) for discrepancy detection
    vendor_calculated_totals: dict[str, float] = field(default_factory=dict)
    lowest_bidder: Optional[str] = None
    highest_bidder: Optional[str] = None
    
    def calculate_totals(self, vendors: list[str], extracted_totals: dict[str, Optional[float]] = None):
        """Set division totals per vendor from extracted values, with calculated fallback."""
        extracted_totals = extracted_totals or {}
        
        for vendor in vendors:
            # Always calculate from line items (for discrepancy detection)
            calc_total = sum(
                grp.vendor_totals.get(vendor, 0) or 0
                for grp in self.groupings
            )
            self.vendor_calculated_totals[vendor] = calc_total
            
            # Prefer the extracted total (from "Carried to Summary" / summary sheet)
            ext = extracted_totals.get(vendor)
            if ext is not None and ext > 0:
                self.vendor_totals[vendor] = ext
            else:
                self.vendor_totals[vendor] = calc_total
        
        valid_totals = {k: v for k, v in self.vendor_totals.items() if v > 0}
        if valid_totals:
            self.lowest_bidder = min(valid_totals, key=valid_totals.get)
            self.highest_bidder = max(valid_totals, key=valid_totals.get)
    
    @property
    def pte_total(self) -> Optional[float]:
        t = sum(g.pte_total or 0 for g in self.groupings)
        return t if t > 0 else None

    def to_dict(self) -> dict:
        d = {
            "division_name": self.division_name,
            "vendor_totals": self.vendor_totals,
            "vendor_calculated_totals": self.vendor_calculated_totals,
            "lowest_bidder": self.lowest_bidder,
            "highest_bidder": self.highest_bidder,
            "groupings": [grp.to_dict() for grp in self.groupings],
        }
        pte = self.pte_total
        if pte is not None:
            d["pte_total"] = pte
        return d


@dataclass
class ComparisonWarning:
    """A validation warning from cross-vendor analysis."""
    level: str                    # "error", "warning", "info"
    category: str                 # "outlier_rate", "missing_item", "total_mismatch"
    division: str
    grouping: Optional[str] = None
    item_id: Optional[str] = None
    item_description: Optional[str] = None
    vendor: Optional[str] = None
    message: str = ""

    def to_dict(self) -> dict:
        d = {"level": self.level, "category": self.category,
             "division": self.division, "message": self.message}
        if self.grouping:
            d["grouping"] = self.grouping
        if self.item_id:
            d["item_id"] = self.item_id
        if self.item_description:
            d["item_description"] = self.item_description
        if self.vendor:
            d["vendor"] = self.vendor
        return d


@dataclass
class ComparisonReport:
    """Full comparison report across all vendors."""
    project_name: str
    template_file: str
    vendors: list[str] = field(default_factory=list)
    divisions: list[DivisionComparison] = field(default_factory=list)
    
    # Extracted project totals (from BOQ summary sheets / "Carried to Summary")
    vendor_totals: dict[str, float] = field(default_factory=dict)
    # Calculated project totals (sum of line items)
    vendor_calculated_totals: dict[str, float] = field(default_factory=dict)
    lowest_bidder: Optional[str] = None
    highest_bidder: Optional[str] = None
    
    # Summary stats
    total_line_items: int = 0
    items_with_variance: int = 0
    avg_variance_percent: Optional[float] = None
    
    # Cross-vendor validation warnings
    warnings: list[ComparisonWarning] = field(default_factory=list)
    
    def calculate_totals(self, extracted_project_totals: dict[str, Optional[float]] = None):
        """Set project totals from extracted values, with calculated fallback."""
        extracted_project_totals = extracted_project_totals or {}
        
        for vendor in self.vendors:
            # Sum the division-level extracted totals
            div_sum = sum(
                div.vendor_totals.get(vendor, 0) or 0
                for div in self.divisions
            )
            # Sum the division-level calculated totals
            calc_sum = sum(
                div.vendor_calculated_totals.get(vendor, 0) or 0
                for div in self.divisions
            )
            self.vendor_calculated_totals[vendor] = calc_sum
            
            # Prefer extracted project total, then division sum, then calculated
            ext = extracted_project_totals.get(vendor)
            if ext is not None and ext > 0:
                self.vendor_totals[vendor] = ext
            elif div_sum > 0:
                self.vendor_totals[vendor] = div_sum
            else:
                self.vendor_totals[vendor] = calc_sum
        
        valid_totals = {k: v for k, v in self.vendor_totals.items() if v > 0}
        if valid_totals:
            self.lowest_bidder = min(valid_totals, key=valid_totals.get)
            self.highest_bidder = max(valid_totals, key=valid_totals.get)
        
        # Summary stats: use median of per-item variance so a few extreme items
        # (wrong unit, typo) don't blow "Avg. Variance" (e.g. 215% → typical 50–80%).
        variances = []
        for div in self.divisions:
            for grp in div.groupings:
                for item in grp.line_items:
                    self.total_line_items += 1
                    if item.rate_variance is not None and item.rate_variance > 0:
                        self.items_with_variance += 1
                        variances.append(item.rate_variance)
        
        if variances:
            sorted_v = sorted(variances)
            n = len(sorted_v)
            self.avg_variance_percent = (
                sorted_v[n // 2] if n % 2 else (sorted_v[n // 2 - 1] + sorted_v[n // 2]) / 2
            )
    
    @property
    def pte_total(self) -> Optional[float]:
        t = sum(d.pte_total or 0 for d in self.divisions)
        return t if t > 0 else None

    def to_dict(self) -> dict:
        summary: dict = {
            "vendor_totals": self.vendor_totals,
            "vendor_calculated_totals": self.vendor_calculated_totals,
            "lowest_bidder": self.lowest_bidder,
            "highest_bidder": self.highest_bidder,
            "total_line_items": self.total_line_items,
            "items_with_variance": self.items_with_variance,
            "avg_variance_percent": round(self.avg_variance_percent, 2) if self.avg_variance_percent else None,
        }
        pte = self.pte_total
        if pte is not None:
            summary["pte_total"] = pte

        result = {
            "project_name": self.project_name,
            "template_file": self.template_file,
            "vendors": self.vendors,
            "has_pte": pte is not None,
            "summary": summary,
            "divisions": [div.to_dict() for div in self.divisions],
        }
        if self.warnings:
            result["warnings"] = [w.to_dict() for w in self.warnings]
        return result
    
    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)
    
    def save(self, path: str):
        """Save report to JSON file."""
        with open(path, 'w') as f:
            f.write(self.to_json())


def load_boq_json(path: str) -> dict:
    """Load a BOQ JSON file."""
    with open(path) as f:
        return json.load(f)


def compare_vendor_jsons(
    template_json: str,
    vendor_jsons: dict[str, str],
    project_name: Optional[str] = None,
) -> ComparisonReport:
    """
    Compare multiple vendor BOQ JSON files against a template.
    
    Args:
        template_json: Path to template BOQ JSON
        vendor_jsons: Dict of {vendor_name: json_path}
        project_name: Project name (extracted from template if not provided)
    
    Returns:
        ComparisonReport with all comparisons
    """
    # Load template
    template = load_boq_json(template_json)
    
    # Load vendor BOQs
    vendor_boqs = {}
    for vendor_name, json_path in vendor_jsons.items():
        vendor_boqs[vendor_name] = load_boq_json(json_path)
    
    return _compare_boqs(
        template=template,
        vendor_boqs=vendor_boqs,
        template_file=template_json,
        project_name=project_name,
    )


def _validate_cross_vendor(report: ComparisonReport) -> None:
    """Run cross-vendor validation and populate report.warnings.

    Checks:
    1. Outlier rates — one vendor's rate is >5x the median of all others
    2. Missing items — vendor has null rate/amount when all others have values
    3. Total mismatch — vendor extracted total differs >10% from calculated total
    """
    vendors = report.vendors
    if len(vendors) < 2:
        return

    for div in report.divisions:
        # ── Total mismatch per division ──
        for vendor in vendors:
            ext = div.vendor_totals.get(vendor, 0)
            calc = div.vendor_calculated_totals.get(vendor, 0)
            if ext > 0 and calc > 0:
                diff_pct = abs(ext - calc) / ext * 100
                if diff_pct > 10:
                    report.warnings.append(ComparisonWarning(
                        level="warning",
                        category="total_mismatch",
                        division=div.division_name,
                        vendor=vendor,
                        message=(
                            f"{vendor}: division total mismatch in {div.division_name} — "
                            f"extracted={ext:,.0f}, calculated={calc:,.0f} ({diff_pct:.0f}% diff). "
                            f"Possible truncated extraction."
                        ),
                    ))

        for grp in div.groupings:
            for item in grp.line_items:
                # Collect non-null rates
                valid_rates = {
                    v: r for v, r in item.vendor_rates.items()
                    if r is not None and r > 0
                }
                null_vendors = [
                    v for v in vendors
                    if item.vendor_rates.get(v) is None
                    and item.vendor_amounts.get(v) is None
                ]

                # ── Missing items ──
                if len(valid_rates) >= 2 and null_vendors:
                    for v in null_vendors:
                        report.warnings.append(ComparisonWarning(
                            level="warning",
                            category="missing_item",
                            division=div.division_name,
                            grouping=grp.grouping_name,
                            item_id=item.item_id,
                            item_description=item.item_description,
                            vendor=v,
                            message=(
                                f"{v}: missing rate/amount for "
                                f"{item.item_id or ''} {(item.item_description or '')[:60]} "
                                f"in {div.division_name}/{grp.grouping_name} "
                                f"(other vendors have values)"
                            ),
                        ))

                # ── Outlier rates ──
                if len(valid_rates) >= 3:
                    sorted_rates = sorted(valid_rates.values())
                    median_rate = sorted_rates[len(sorted_rates) // 2]
                    if median_rate > 0:
                        for v, r in valid_rates.items():
                            ratio = r / median_rate
                            if ratio > 5.0 or ratio < 0.2:
                                report.warnings.append(ComparisonWarning(
                                    level="error" if ratio > 10 or ratio < 0.1 else "warning",
                                    category="outlier_rate",
                                    division=div.division_name,
                                    grouping=grp.grouping_name,
                                    item_id=item.item_id,
                                    item_description=item.item_description,
                                    vendor=v,
                                    message=(
                                        f"{v}: rate {r:,.2f} is {ratio:.1f}x median ({median_rate:,.2f}) "
                                        f"for {item.item_id or ''} {(item.item_description or '')[:50]} "
                                        f"— possible extraction error"
                                    ),
                                ))


def _compare_boqs(
    template: dict,
    vendor_boqs: dict[str, dict],
    template_file: str,
    project_name: Optional[str] = None,
    pte_boq: Optional[dict] = None,
) -> ComparisonReport:
    """
    Internal comparison function.
    
    Args:
        template: Template BOQ dict
        vendor_boqs: Dict of {vendor_name: boq_dict}
        template_file: Template file path for report
        project_name: Project name
    
    Returns:
        ComparisonReport
    """
    vendors = list(vendor_boqs.keys())
    
    report = ComparisonReport(
        project_name=project_name or template.get("project_name", "Unknown Project"),
        template_file=template_file,
        vendors=vendors,
    )
    
    # Build PTE lookup (optional benchmark — NOT a vendor)
    pte_lookup: dict = {}
    if pte_boq:
        for div in pte_boq.get("divisions", []):
            div_key = _upper(div.get("division_name", ""))
            for grp in div.get("grouping", []):
                grp_key = _upper(grp.get("grouping_name", ""))
                for sub_name, idx, item in _iter_grouping_items_with_subgrouping(grp):
                    sub_key = _upper(sub_name)
                    item_key = _upper(item.get("item_id", ""))
                    key = (div_key, grp_key, sub_key, item_key, idx)
                    pte_lookup[key] = VendorLineItem.from_dict(item)
    
    # Build lookup for vendor items AND extracted division totals
    vendor_lookups = {}
    # {vendor: {UPPER(div_name): extracted_total_float}}
    vendor_division_totals: dict[str, dict[str, Optional[float]]] = {}
    # {vendor: extracted project total}
    vendor_project_totals: dict[str, Optional[float]] = {}
    
    for vendor_name, boq in vendor_boqs.items():
        lookup = {}
        div_totals = {}
        
        # Extract project-level total
        proj_total_str = boq.get("project_total_amount")
        if proj_total_str:
            try:
                vendor_project_totals[vendor_name] = float(str(proj_total_str).replace(",", ""))
            except (ValueError, TypeError):
                vendor_project_totals[vendor_name] = None
        
        for div in boq.get("divisions", []):
            div_name = div.get("division_name", "")
            div_key = _upper(div_name)
            
            # Extract the division total from the BOQ data (1-to-1, no calculation)
            div_total_str = div.get("division_total_amount")
            if div_total_str:
                try:
                    div_totals[div_key] = float(str(div_total_str).replace(",", ""))
                except (ValueError, TypeError):
                    div_totals[div_key] = None
            
            for grp in div.get("grouping", []):
                grp_name = grp.get("grouping_name", "")
                grp_key = _upper(grp_name)
                for sub_name, idx, item in _iter_grouping_items_with_subgrouping(grp):
                    item_id = item.get("item_id", "")
                    sub_key = _upper(sub_name)
                    item_key = _upper(item_id)
                    key = (div_key, grp_key, sub_key, item_key, idx)
                    lookup[key] = VendorLineItem.from_dict(item)
        
        vendor_lookups[vendor_name] = lookup
        vendor_division_totals[vendor_name] = div_totals
    
    # Compare each division/grouping/item from template
    for div in template.get("divisions", []):
        div_name = div.get("division_name", "")
        div_key = _upper(div_name)
        div_comparison = DivisionComparison(division_name=div_name)
        
        for grp in div.get("grouping", []):
            grp_name = grp.get("grouping_name", "")
            grp_key = _upper(grp_name)
            grp_comparison = GroupingComparison(grouping_name=grp_name)
            
            for sub_name, idx, item in _iter_grouping_items_with_subgrouping(grp):
                item_id = item.get("item_id", "")
                sub_key = _upper(sub_name)
                item_key = _upper(item_id)
                key = (div_key, grp_key, sub_key, item_key, idx)
                
                item_comparison = LineItemComparison(
                    item_id=item_id,
                    item_description=item.get("item_description", ""),
                    item_quantity=item.get("item_quantity"),
                    item_unit=item.get("item_unit"),
                    sub_grouping_name=sub_name,
                )
                
                # Get rates from each vendor
                for vendor_name in vendors:
                    vendor_item = vendor_lookups[vendor_name].get(key)
                    if vendor_item:
                        item_comparison.vendor_rates[vendor_name] = vendor_item.item_rate
                        item_comparison.vendor_amounts[vendor_name] = vendor_item.item_amount
                    else:
                        # Try fuzzy match by item_id within same division/grouping/sub-grouping
                        fuzzy_key = None
                        for k in vendor_lookups[vendor_name]:
                            if (
                                k[0] == div_key
                                and k[1] == grp_key
                                and k[2] == sub_key
                                and k[3] == item_key
                            ):
                                fuzzy_key = k
                                break
                        
                        # Fallback: match within division/grouping ignoring sub-grouping
                        if fuzzy_key is None:
                            for k in vendor_lookups[vendor_name]:
                                if (
                                    k[0] == div_key
                                    and k[1] == grp_key
                                    and k[3] == item_key
                                ):
                                    fuzzy_key = k
                                    break
                        
                        # Final fallback: match within division only (legacy behavior)
                        if fuzzy_key is None:
                            for k in vendor_lookups[vendor_name]:
                                if k[0] == div_key and k[3] == item_key:
                                    fuzzy_key = k
                                    break
                        
                        if fuzzy_key:
                            vendor_item = vendor_lookups[vendor_name][fuzzy_key]
                            item_comparison.vendor_rates[vendor_name] = vendor_item.item_rate
                            item_comparison.vendor_amounts[vendor_name] = vendor_item.item_amount
                        else:
                            item_comparison.vendor_rates[vendor_name] = None
                            item_comparison.vendor_amounts[vendor_name] = None
                
                # Match PTE benchmark (same matching strategy as vendors)
                if pte_lookup:
                    pte_item = pte_lookup.get(key)
                    if pte_item is None:
                        for k in pte_lookup:
                            if k[0] == div_key and k[1] == grp_key and k[3] == item_key:
                                pte_item = pte_lookup[k]
                                break
                    if pte_item is None:
                        for k in pte_lookup:
                            if k[0] == div_key and k[3] == item_key:
                                pte_item = pte_lookup[k]
                                break
                    if pte_item:
                        item_comparison.pte_rate = pte_item.item_rate
                        item_comparison.pte_amount = pte_item.item_amount

                item_comparison.calculate_flags()
                grp_comparison.line_items.append(item_comparison)
            
            grp_comparison.calculate_totals(vendors)
            div_comparison.groupings.append(grp_comparison)
        
        # Pass extracted division totals for each vendor
        extracted_div_totals = {}
        for vendor_name in vendors:
            extracted_div_totals[vendor_name] = vendor_division_totals.get(vendor_name, {}).get(div_key)
        
        div_comparison.calculate_totals(vendors, extracted_totals=extracted_div_totals)
        report.divisions.append(div_comparison)
    
    report.calculate_totals(extracted_project_totals=vendor_project_totals)
    _validate_cross_vendor(report)
    return report


# CLI for testing
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 4:
        print("Usage: python -m src.compare_vendors template.json vendor1.json vendor2.json ...")
        print("       python -m src.compare_vendors template.json 'Vendor A:vendor_a.json' 'Vendor B:vendor_b.json'")
        sys.exit(1)
    
    template_path = sys.argv[1]
    
    # Parse vendor arguments
    vendor_jsons = {}
    for arg in sys.argv[2:]:
        if ":" in arg:
            name, path = arg.split(":", 1)
            vendor_jsons[name] = path
        else:
            # Use filename as vendor name
            name = Path(arg).stem
            vendor_jsons[name] = arg
    
    print(f"Template: {template_path}")
    print(f"Vendors: {list(vendor_jsons.keys())}")
    print()
    
    report = compare_vendor_jsons(template_path, vendor_jsons)
    
    print(f"Project: {report.project_name}")
    print(f"Total items: {report.total_line_items}")
    print()
    print("=== VENDOR TOTALS ===")
    for vendor, total in report.vendor_totals.items():
        flag = ""
        if vendor == report.lowest_bidder:
            flag = " [LOWEST]"
        elif vendor == report.highest_bidder:
            flag = " [HIGHEST]"
        print(f"  {vendor}: {total:,.2f}{flag}")
    
    print()
    print("=== DIVISION BREAKDOWN ===")
    for div in report.divisions:
        print(f"\n{div.division_name}:")
        for vendor, total in div.vendor_totals.items():
            flag = ""
            if vendor == div.lowest_bidder:
                flag = " ✓"
            elif vendor == div.highest_bidder:
                flag = " ✗"
            print(f"  {vendor}: {total:,.2f}{flag}")
    
    # Save report
    output_path = "comparison_report.json"
    report.save(output_path)
    print(f"\nReport saved to: {output_path}")
