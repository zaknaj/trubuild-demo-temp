# Commercial RFP Job Queue Contract


## Supported Jobs

- `comm_rfp_extract`
- `comm_rfp_compare`


## Enqueue Payload

Required:
- `package_id` (string)
- `company_id` (string)
- `analysis_type` (`extract` or `compare`)

Optional:
- `user_id` (string)
- `user_name` (string)
- `package_name` (string)
- `country_code` (string, default `USA`)
- `remote_ip` (string)
- `compute_reanalysis` (boolean, default `true`)

Example:

```json
{
  "package_id": "pkg_123",
  "company_id": "co_abc",
  "analysis_type": "compare",
  "user_id": "usr_1",
  "user_name": "Sam",
  "package_name": "Airport Package",
  "country_code": "USA",
  "compute_reanalysis": true
}
```

## Job Status States

Possible states:
- `pending`
- `in progress`
- `completed`
- `error`

On completion, read the `result` artifact.

## Output Shape: `extract`

```json
{
  "analysisType": "extract",
  "source_file": "template.xlsx",
  "source_key": "co_abc/pkg_123/comm_rfp/boq/template.xlsx",
  "boq": {
    "project_name": "string|null",
    "contractor": "string|null",
    "project_total_amount": "string|null",
    "divisions": [
      {
        "division_name": "string",
        "division_total_amount": "string|null",
        "grouping": [
          {
            "grouping_name": "string",
            "sub_groupings": [
              {
                "sub_grouping_name": "string",
                "line_items": [
                  {
                    "item_id": "string",
                    "item_description": "string",
                    "item_quantity": 0,
                    "item_unit": "string|null",
                    "item_rate": 0,
                    "item_amount": 0
                  }
                ]
              }
            ]
          }
        ]
      }
    ]
  }
}
```

## Output Shape: `compare`

```json
{
  "analysisType": "compare",
  "template_file": "template.xlsx",
  "template_key": "co_abc/pkg_123/comm_rfp/boq/template.xlsx",
  "pte_file": "pte.xlsx|null",
  "pte_key": "co_abc/pkg_123/comm_rfp/boq/pte.xlsx|null",
  "vendors": {
    "VendorA": {
      "file": "vendor_a.xlsx",
      "key": "co_abc/pkg_123/comm_rfp/tender/VendorA/vendor_a.xlsx"
    },
    "VendorB": {
      "file": "vendor_b.xlsx",
      "key": "co_abc/pkg_123/comm_rfp/tender/VendorB/vendor_b.xlsx"
    }
  },
  "report": {
    "project_name": "string",
    "template_file": "string",
    "vendors": ["VendorA", "VendorB"],
    "total_line_items": 0,
    "summary": {},
    "divisions": []
  }
}
```

Notes:
- `pte_file` and `pte_key` are `null` when no PTE is available.
- `vendors` object keys are contractor folder names.

## MinIO Upload Agreements

All object keys must use:
- `{company_id}/{package_id}/...`

Expected folders:
- `comm_rfp/boq/`
  - must contain at least one Excel (`.xlsx` or `.xls`)
  - template is selected from here
  - optional PTE may also be placed here
- `comm_rfp/rfp/`
  - optional fallback location for PTE Excel
- `comm_rfp/tender/<contractor>/`
  - one folder per contractor/vendor
  - each contractor folder must contain at least one Excel file

Selection rules:
- Template: chosen from `comm_rfp/boq/` (prefers names containing `template`, then `boq`).
- PTE: first filename containing `pte` (search `comm_rfp/boq/` first, then `comm_rfp/rfp/`).
- Vendor file: first sorted Excel found in each `comm_rfp/tender/<contractor>/`.

## App Checklist

- Upload files to MinIO with the required layout.
- Enqueue with `analysis_type = extract` or `compare`.
- Poll until status is `completed` or `error`.
- Parse the completed `result` artifact using the output shapes above.
