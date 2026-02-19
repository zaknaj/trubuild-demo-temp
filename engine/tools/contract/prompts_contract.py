import json
from utils.core.log import get_logger

prompts_NEC = [
    # """,
    # 3. Extracts the start, end date and duration
    """
From the execution of the contract, extract the commencement date (under keyword 'Start date') and end date (under keyword 'End date') in the format: date abbreviated month year (no slashes). Before answering, go through the whole document to find the final start and end dates. Calculate the duration between the commencement date and the end date in terms of months and output it under the keyword 'Duration'. Cite the document name, section, and page under which you got this information from under the keyword 'Source'. No additional information is required.
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "type": "object",
  "properties": {
    "dates": {
      "type": "object",
      "properties": {
        "startDate": {
          "type": "string",
          "description": "The commencement date of the contract in the format: date abbreviated month year.",
          "default": "Start date not found"
        },
        "endDate": {
          "type": "string",
          "description": "The end date of the contract in the format: date abbreviated month year.",
          "default": "End date not found"
        },
        "duration": {
          "type": "integer",
          "description": "The duration between the commencement date and the end date in terms of months.",
          "default": 0
        },
        "source": {
          "type": "string",
          "description": "The citation for the document name, section, and page where the dates were extracted.",
          "default": "Source information not found"
        }
      },
      "required": ["startDate", "endDate", "duration", "source"]
    }
  },
  "required": ["dates"]
}

""",
    # 4. Extracts and summarizes the scope
    """
Extract and summarize the scope of work (otherwise known as works or works information) of the contract. Title of your output should be "Scope of Work". The first sentence should be an overall summary of the scope of work. Then, the next sentence in a new line should read, "The scope of work includes but is not limited to:" and list the scopes in bullet point format. Bold important words/phrases that need to be highlighted. Cite the document name, section, and page under which you got this information under the keyword 'Source'.
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "type": "object",
  "properties": {
    "scopeOfWork": {
      "type": "object",
      "properties": {
        "summary": {
          "type": "string",
          "description": "An overall summary of the scope of work.",
          "default": "Scope summary not available."
        },
        "details": {
          "type": "string",
          "description": "Sentence should start with: The scope of work includes but is not limited to:",
          "default": "The scope of work includes but is not limited to:"
        },
        "scopes": {
          "type": "array",
          "items": {
            "type": "string",
            "default": "Specific scope item not provided."
          },
          "description": "List of specific scopes included in the work, in bullet point format.",
          "default": []
        },
        "source": {
          "type": "string",
          "description": "The citation for the document name, section, and page where the scope information was extracted.",
          "default": "Source information not available."
        }
      },
      "required": ["summary", "details", "scopes", "source"]
    }
  },
  "required": ["scopeOfWork"]
}
However, if the provided context is not a contract or does not have relevant scope as intended here, please use the default statements to inform that this is not a full contract with scopes.
""",
    # 5. Compensation events
    """
{
  "compensationEventsList": {
    "compensationEvents": {
      "summary": "string",
      // Summary of compensation events including triggers and procedures for addressing them in a few sentences.

      "keyPoints": [
        "string"
      ],
      // Key factors including trigger conditions and expected responses.

      "source": "string"
      // Reference for the source of compensation events information.
    },
    "changeManagement": {
      "summary": "string",
      // Describes the protocols for managing changes in the project scope or contract terms in a few sentences.

      "keyPoints": [
        "string"
      ],
      // Essential aspects such as timelines for submissions and responses in a few sentences.

      "source": "string"
      // Source reference for change management information.
    },
    "claims": {
      "summary": "string",
      // Overview of the process for submitting claims relating to compensation or contractual adjustments in a few sentences.

      "keyPoints": [
        "string"
      ],
      // Highlights procedural steps and any critical deadlines involved in a few sentences.

      "source": "string"
      // Document source for claims information.
    },
    "details": {
      "summary": "string",
      // Summary capturing essential details such as important figures, dates, and financial implications in a few sentences.

      "keyPoints": [
        "string"
      ],
      // Concise list of financial and chronological details critical to the contract's compensation clauses in a few sentences.

      "source": "string"
      // Source reference for the detailed figures and dates.
    },
    "source": "string"
    // General citation for the entire compensation events list section.
  }
}

However, if the provided context is not a contract or does not have relevant information asked above, please use the default statements to inform that this is not a full contract with the required information.

""",
    # 6. Delay damages
    """
Review the contract and its supporting documents and identify the penalties covered regarding delay damages. Title of your output should be "Delay Damages". You can find relevant information in section X7 and the Z clauses. Bold key information that needs to be highlighted, such as dates, percentages, amount, figures, among others. Cite the document name, section, and page under which you got this information from under the keyword 'Source'. If there are none, state that there are no Delay Damages penalties covered in the contract.

{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "type": "object",
  "properties": {
    "delayDamages": {
      "type": "object",
      "properties": {
        "description": {
          "type": "string",
          "description": "Details of the penalties for delay damages including key information highlighted. If there are no penalties, state this explicitly.",
          "default": "No Delay Damages penalties covered in the contract."
        },
        "keyInformation": {
          "type": "array",
          "items": {
            "type": "string"
          },
          "description": "List of key details such as dates, percentages, amounts, figures that need to be highlighted.",
          "default": []
        },
        "source": {
          "type": "string",
          "description": "The citation for the document name, section, and page where the delay damages information was extracted.",
          "default": "Source information not available"
        }
      },
      "required": ["description", "keyInformation", "source"]
    }
  },
  "required": ["delayDamages"]
}
However, if the provided context is not a contract or does not have relevant information asked above, please use the default statements to inform that this is not a full contract with the relevant information.

""",
    # 7. Termination clauses
    """
Summarize termination clauses (sanctions/retribution/penance):

If termination penalties are mentioned in the contract, identify and summarize them. Bold key information such as dates, percentages, amounts, and figures.
If no termination penalties are present, state: "No termination penalties are mentioned in this contract."
Include any other relevant information regarding termination found in the contract.
If there are no clauses about termination, state: "No clauses regarding termination were found in this contract."

{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "type": "object",
  "properties": {
    "terminationClauses": {
      "type": "object",
      "properties": {
        "penalties": {
          "type": "string",
          "description": "Summary of termination penalties including key details such as dates, percentages, amounts, and figures. If no penalties are mentioned, it states clearly no penalties are present.",
          "default": "No termination penalties are mentioned in this contract."
        },
        "otherInformation": {
          "type": "string",
          "description": "Any other relevant information regarding termination found in the contract.",
          "default": "No other relevant information regarding termination."
        },
        "source": {
          "type": "string",
          "description": "The citation for the document name, section, and page where the termination information was extracted in the format Source: [Document Name], Section [Section Number], Page [Page Number].
",
          "default": "No clauses regarding termination were found in this contract."
        }
      },
      "required": ["penalties", "otherInformation", "source"]
    }
  },
  "required": ["terminationClauses"]
}
However, if the provided context is not a contract or does not have relevant information asked above, please use the default statements to inform that this is not a full contract with the relevant information.
""",
    # 8. Quality and KPI
    """
Summarize the quality requirements and key performance indicators (KPIs) specified in the contract. Include all relevant metrics and standards. Also check the appendices of the document. Bold key information that need to be highlighted, such as dates, percentages, amount, figures, among others. Cite the document name, section, and page under which you got this information from under the keyword 'Source'.

{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "type": "object",
  "properties": {
    "qualityandKPI": {
      "type": "object",
      "properties": {
        "qualityRequirements": {
          "type": "string",
          "description": "Summary of the quality requirements specified in the contract, including metrics and standards in a few sentences.",
          "default": "No specific quality requirements mentioned in this contract."
        },
        "keyPerformanceIndicators": {
          "type": "array",
          "items": {
            "type": "string"
          },
          "description": "List of KPIs with details such as target values, dates, percentages, and figures that need to be highlighted.",
          "default": []
        },
        "source": {
          "type": "string",
          "description": "The citation for the document name, section, and page where the quality and KPI information was extracted.",
          "default": "No KPI or quality requirements information found in this contract."
        }
      },
      "required": ["qualityRequirements", "keyPerformanceIndicators", "source"]
    }
  },
  "required": ["qualityandKPI"]
}
However, if the provided context is not a contract or does not have relevant information asked above, please use the default statements to inform that this is not a full contract with the relevant information.
""",
    # 9. Dispute Resolution
    """
Please analyze the contract and accompanying documents to extract a brief summary of the provisions related to dispute resolution costs.
Particularly focus on any financial obligations or penalties associated with resolving disputes under this contract.
Your summary should be brief but informative, including key figures and values.
Organize your summary under the title 'Dispute Resolution Costs'. Bold any significant figures such as dates, percentages, and monetary amounts.
Conclude by citing the document name, section, and page number under the keyword 'Source'.

{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "type": "object",
  "properties": {
    "disputeResolutionCosts": {
      "type": "object",
      "properties": {
        "summary": {
          "type": "string",
          "description": "A brief but informative summary of the financial obligations or penalties associated with resolving disputes, including significant figures such as dates, percentages, and monetary amounts.",
          "default": "No specific penalties or costs detailed for dispute resolutions; general framework provided."
        },
        "source": {
          "type": "string",
          "description": "The citation for the document name, section, and page where the dispute resolution costs information was extracted.",
          "default": "Source information not available."
        }
      },
      "required": ["summary", "source"]
    }
  },
  "required": ["disputeResolutionCosts"]
}

However, if the provided context is not a contract or does not have relevant information asked above, please use the default statements to inform that this is not a full contract with the relevant information.

""",
    # 10. Modified standard clauses detection
    """
This is an NEC contract. All non-standard clauses are supposed to be stated in Option Z.
Check whether the standard clauses in the other part of the contract have been modified.
If so, numerically list the modified clauses, without the content of the clauses verbatim.

{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "type": "object",
  "properties": {
    "modifiedStandardClauses": {
      "type": "object",
      "properties": {
        "listOfModifiedClauses": {
          "type": "array",
          "items": {
            "type": "string"
          },
          "description": "Numerically listed modified standard clauses, identified by their clause numbers or identifiers, without the content of the clauses verbatim.",
          "default": []
        },
        "source": {
          "type": "string",
          "description": "The citation for the document name, section, and page where the information about modified standard clauses was extracted.",
          "default": "No modifications to standard clauses were detected."
        }
      },
      "required": ["listOfModifiedClauses", "source"]
    }
  },
  "required": ["modifiedStandardClauses"]
}
However, if the provided context is not a contract or does not have relevant information asked above, please use the default statements to inform that this is not a full contract with the relevant information.

""",
    # 11. Analyze option Z
    """
Review Option Z of the contract, which contains non-standard clauses. Identify and numerically list only those clauses within Option Z that are deemed high-risk. High-risk clauses should be determined based on the following 15 key words:  Payment, Delay, Damages, Quality, Milestone, Completion, Delivery, Performance, Incentive, Budget, Corruption, HSE, Key Date, Site Information, Disallowed Cost, Defined Cost, Works Information, Scope, Prices, Compensation, Claims, Insurance, Risk. List these clauses without including their verbatim content. Organize your findings under the title 'High Risk Clauses.

{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "type": "object",
  "properties": {
    "highRiskClauses": {
      "type": "object",
      "properties": {
        "clauses": {
          "type": "array",
          "items": {
            "type": "object",
            "properties": {
              "clauseIdentifier": {
                "type": "string",
                "description": "The identifier or number of the high-risk clause within Option Z."
              },
              "keywords": {
                "type": "array",
                "items": {
                  "type": "string"
                },
                "description": "List of keywords indicating why the clause is considered high-risk."
              }
            },
            "required": ["clauseIdentifier", "keywords"]
          },
          "description": "A numerically ordered list of high-risk clauses identified within Option Z.",
          "default": []
        },
        "title": {
          "type": "string",
          "description": "The title for the section where findings are organized.",
          "default": "High Risk Clauses"
        }
      },
      "required": ["clauses", "title"]
    }
  },
  "required": ["highRiskClauses"]
}

However, if the provided context is not a contract or does not have relevant information asked above, please use the default statements to inform that this is not a full contract with the relevant information.
""",
    # 13. Non conformance penalty
    """
Carefully review the contract and its supporting documents to extract a summary of the penalties for non-conformance.
Emphasize the conditions under which these penalties apply, and the criteria for assessing non-conformance.
Your summary should be brief but informative, including key figures, values, and dates.
Organize your findings under the title 'Non-conformance Penalties'. Bold any significant figures such as dates, percentages, and monetary amounts.
If the contract does not specify non-conformance penalties, note this absence and outline the general non-conformance guidelines instead. Finally, cite the document name, section, and page number under the keyword 'Source'.

{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "type": "object",
  "properties": {
    "nonconformancePenalties": {
      "type": "object",
      "properties": {
        "summary": {
          "type": "string",
          "description": "A brief but informative summary of the penalties for non-conformance, including conditions, criteria, and key figures such as dates, percentages, and monetary amounts.",
          "default": "No specific non-conformance penalties specified; general guidelines provided instead."
        },
        "keyFigures": {
          "type": "array",
          "items": {
            "type": "string"
          },
          "description": "List of significant figures such as dates, percentages, monetary amounts, and other pertinent metrics related to non-conformance.",
          "default": []
        },
        "source": {
          "type": "string",
          "description": "The citation for the document name, section, and page where the non-conformance penalties information was extracted.",
          "default": "Source information not available."
        }
      },
      "required": ["summary", "keyFigures", "source"]
    }
  },
  "required": ["nonconformancePenalties"]
}

However, if the provided context is not a contract or does not have relevant information asked above, please use the default statements to inform that this is not a full contract with the relevant information.
""",
    # 14. Contradiction analysis
    """
List the contradictory clauses found within this contract in the below JSON schema.
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "type": "object",
  "properties": {
    "contradictions": {
      "type": "object",
      "patternProperties": {
        "Clause_ID": {
          "type": "array",
          "items": {
            "type": "string",
            "description": "Clause number that contradicts the key clause"
          },
          "description": "List of clauses that contradict the key clause",
          "default": [],
          "uniqueItems": true
        }
      },
      "description": "Mapping of clause numbers to lists of clause numbers they significantly contradict with. Only include contradictions that are significant, meaning complete opposites. Don't include vague, ambiguous, and other types of problematic clauses. Only contradictory ones that are likely to have a major impact on the client and contractor.",
      "default": {}
    }
  },
  "required": ["contradictions"]
}

However, if the provided context is not a contract or does not have relevant information asked above, please use the default statements to inform that this is not a full contract with the relevant information.

""",
    # 15. Ambiguity analysis
    """

  {
  "$schema": "http://json-schema.org/draft-07/schema#",
  "type": "object",
  "properties": {
    "ambiguousClauses": {
      "type": "object",
      "properties": {
        "ambiguousClausesList": {
          "type": "array",
          "items": {
            "type": "string",
            "description": "The clause number of a highly ambiguous clause within the contract."
          },
          "description": "A list of the most significant ambiguous clauses identified within the contract, presented in a numbered format.",
          "default": []
        }
      },
      "required": ["ambiguousClausesList"]
    }
  },
  "required": ["ambiguousClauses"]
}
However, if the provided context is not a contract or does not have relevant information asked above, please use the default statements to inform that this is not a full contract with the relevant information.

""",
    # 16 duplicate clauses
    """

Go through the whole document and identify duplicate clauses found. Output your answer in numbered format and list only the pairs of duplicate clause numbers.
The title of your output should be "Duplicate Clauses". If no duplicates are found, state: "No duplicate clauses found."

{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "type": "object",
  "properties": {
    "duplicateClauses": {
      "type": "array",
      "items": {
        "type": "array",
        "items": {
          "type": "string",
          "description": "Clause number"
        },
        "minItems": 2,
        "maxItems": 2,
        "description": "A pair of duplicate clauses"
      },
      "description": "List of pairs of duplicate clauses. Each pair should be unique and not repeated. If no duplicates are found, the array is empty.",
      "default": [],
      "uniqueItems": true
    }
  },
  "required": ["duplicateClauses"]
}
However, if the provided context is not a contract or does not have relevant information asked above, please use the default statements to inform that this is not a full contract with the relevant information.

""",
    # 17. Payment terms 1
    """
Please provide a detailed summary of the following aspects in the contract: valuation, assessment, retention, components.
Title your summary 'Payment Terms.' Ensure that all pertinent details and specifics are included.
At the end, reference the document name, section, and page number using the keyword 'Source.'
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "type": "object",
  "properties": {
    "paymentTerms1": {
      "type": "object",
      "properties": {
        "valuation": {
          "type": "string",
          "description": "Describes the process and criteria for determining the financial value of work completed under the contract in a maximum of two sentences.",
          "default": "Valuation details not available."
        },
        "assessment": {
          "type": "string",
          "description": "Details the procedures for reviewing, approving, and certifying payment amounts due under the contract in a maximum of two sentences.",
          "default": "Assessment procedures not specified."
        },
        "retention": {
          "type": "string",
          "description": "Outlines the conditions under which a portion of the payment is withheld as security against defects or incomplete work in a maximum of two sentences.",
          "default": "Retention details not provided."
        },
        "components": {
          "type": "string",
          "description": "Specifies any particular parts of the work or materials that have unique pricing or valuation conditions in a maximum of two sentences.",
          "default": "No specific components outlined."
        },
        "source": {
          "type": "string",
          "description": "The citation for the document name, section, and page where the payment terms information was extracted in a maximum of two sentences.",
          "default": "Source information not available."
        }
      },
      "required": ["valuation", "assessment", "retention", "components", "source"]
    }
  },
  "required": ["paymentTerms1"]
}
However, if the provided context is not a contract or does not have relevant information asked above, please use the default statements to inform that this is not a full contract with the relevant information.

""",
    # 18. Payment terms 2
    """
Provide a comprehensive summary of these aspects of the contract: adjustments, costs, terms, and the specified currency for defined costs.
Title your summary 'Payment Terms.' Ensure that all pertinent details and specifics are included.
At the end, reference the document name, section, and page number using the keyword 'Source.'
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "type": "object",
  "properties": {
    "paymentTerms2": {
      "type": "object",
      "properties": {
        "adjustments": {
          "type": "string",
          "description": "Details of any adjustments that can be made to payments, including conditions and methods. Make it concise; in a maximum of 2-3 sentences.",
          "default": "Adjustment details not available."
        },
        "costs": {
          "type": "string",
          "description": "Brief summary of the costs covered under the contract, including how costs are calculated and what expenses are included or excluded. Make it concise; about 2-3 sentences and nothing more.",
          "default": "Cost details not provided."
        },
        "terms": {
          "type": "string",
          "description": "Breifly summarize the specific terms related to payment schedules, late payment penalties, and other relevant payment condition. Make it concise; about 2-3 sentences and nothing more.",
          "default": "Payment terms not specified."
        },
        "specifiedCurrency": {
          "type": "string",
          "description": "The currency used for defining and settling all costs under the contract.",
          "default": "Currency details not specified."
        },
        "source": {
          "type": "string",
          "description": "The citation for the document name, section, and page where the payment terms information was extracted.",
          "default": "Source information not available."
        }
      },
      "required": ["adjustments", "costs", "terms", "specifiedCurrency", "source"]
    }
  },
  "required": ["paymentTerms2"]
}
However, if the provided context is not a contract or does not have relevant information asked above, please use the default statements to inform that this is not a full contract with the relevant information.

""",
]

prompts_FIDIC = [
    # """,
    # 3. Extracts the start, end date and duration
    """
From the execution of the contract, extract the commencement date (under keyword 'Start date') and end date (under keyword 'End date') in the format: date abbreviated month year (no slashes). Before answering, go through the whole document to find the final start and end dates. Calculate the duration between the commencement date and the end date in terms of months and output it under the keyword 'Duration'. Cite the document name, section, and page under which you got this information from under the keyword 'Source'. No additional information is required.
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "type": "object",
  "properties": {
    "dates": {
      "type": "object",
      "properties": {
        "startDate": {
          "type": "string",
          "description": "The commencement date of the contract in the format: date abbreviated month year.",
          "default": "Start date not found"
        },
        "endDate": {
          "type": "string",
          "description": "The end date of the contract in the format: date abbreviated month year.",
          "default": "End date not found"
        },
        "duration": {
          "type": "integer",
          "description": "The duration between the commencement date and the end date in terms of months.",
          "default": 0
        },
        "source": {
          "type": "string",
          "description": "The citation for the document name, section, and page where the dates were extracted.",
          "default": "Source information not found"
        }
      },
      "required": ["startDate", "endDate", "duration", "source"]
    }
  },
  "required": ["dates"]
}
""",
    # 4. Extracts and summarizes the scope
    """
Extract and summarize the scope of work (otherwise known as works or works information) of the contract. Title of your output should be "Scope of Work". The first sentence should be an overall summary of the scope of work. Then, the next sentence in a new line should read, "The scope of work includes but is not limited to:" and list the scopes in bullet point format. Bold important words/phrases that need to be highlighted. Cite the document name, section, and page under which you got this information under the keyword 'Source'.
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "type": "object",
  "properties": {
    "scopeOfWork": {
      "type": "object",
      "properties": {
        "summary": {
          "type": "string",
          "description": "An overall summary of the scope of work.",
          "default": "Scope summary not available."
        },
        "details": {
          "type": "string",
          "description": "Sentence should start with: The scope of work includes but is not limited to:",
          "default": "The scope of work includes but is not limited to:"
        },
        "scopes": {
          "type": "array",
          "items": {
            "type": "string",
            "default": "Specific scope item not provided."
          },
          "description": "List of specific scopes included in the work, in bullet point format.",
          "default": []
        },
        "source": {
          "type": "string",
          "description": "The citation for the document name, section, and page where the scope information was extracted.",
          "default": "Source information not available."
        }
      },
      "required": ["summary", "details", "scopes", "source"]
    }
  },
  "required": ["scopeOfWork"]
}
""",
    # 5. Compensation events
    """
{
  "compensationEventsList": {
    "compensationEvents": {
      "summary": "string",
      // Summary of compensation events including triggers and procedures for addressing them in a few sentences.

      "keyPoints": [
        "string"
      ],
      // Key factors including trigger conditions and expected responses.

      "source": "string"
      // Reference for the source of compensation events information.
    },
    "changeManagement": {
      "summary": "string",
      // Describes the protocols for managing changes in the project scope or contract terms in a few sentences.

      "keyPoints": [
        "string"
      ],
      // Essential aspects such as timelines for submissions and responses in a few sentences.

      "source": "string"
      // Source reference for change management information.
    },
    "claims": {
      "summary": "string",
      // Overview of the process for submitting claims relating to compensation or contractual adjustments in a few sentences.

      "keyPoints": [
        "string"
      ],
      // Highlights procedural steps and any critical deadlines involved in a few sentences.

      "source": "string"
      // Document source for claims information.
    },
    "details": {
      "summary": "string",
      // Summary capturing essential details such as important figures, dates, and financial implications in a few sentences.

      "keyPoints": [
        "string"
      ],
      // Concise list of financial and chronological details critical to the contract's compensation clauses in a few sentences.

      "source": "string"
      // Source reference for the detailed figures and dates.
    },
    "source": "string"
    // General citation for the entire compensation events list section.
  }
}

However, if the provided context is not a contract or does not have relevant information asked above, please use the default statements to inform that this is not a full contract with the required information.

""",
    # 6. Delay damages
    """
Review the contract and its supporting documents and identify the penalties covered regarding delay damages. Title of your output should be "Delay Damages". You can find relevant information in section X7 and the Z clauses. Bold key information that needs to be highlighted, such as dates, percentages, amount, figures, among others. Cite the document name, section, and page under which you got this information from under the keyword 'Source'. If there are none, state that there are no Delay Damages penalties covered in the contract.

{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "type": "object",
  "properties": {
    "delayDamages": {
      "type": "object",
      "properties": {
        "description": {
          "type": "string",
          "description": "Details of the penalties for delay damages including key information highlighted. If there are no penalties, state this explicitly.",
          "default": "No Delay Damages penalties covered in the contract."
        },
        "keyInformation": {
          "type": "array",
          "items": {
            "type": "string"
          },
          "description": "List of key details such as dates, percentages, amounts, figures that need to be highlighted.",
          "default": []
        },
        "source": {
          "type": "string",
          "description": "The citation for the document name, section, and page where the delay damages information was extracted.",
          "default": "Source information not available"
        }
      },
      "required": ["description", "keyInformation", "source"]
    }
  },
  "required": ["delayDamages"]
}
""",
    # 7. Termination clauses
    """
Summarize termination clauses (sanctions/retribution/penance):

If termination penalties are mentioned in the contract, identify and summarize them. Bold key information such as dates, percentages, amounts, and figures.
If no termination penalties are present, state: "No termination penalties are mentioned in this contract."
Include any other relevant information regarding termination found in the contract.
Cite the source for the information using the format: "Source: [Document Name], Section [Section Number], Page [Page Number]".
If there are no clauses about termination, state: "No clauses regarding termination were found in this contract."

{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "type": "object",
  "properties": {
    "terminationClauses": {
      "type": "object",
      "properties": {
        "penalties": {
          "type": "string",
          "description": "Summary of termination penalties including key details such as dates, percentages, amounts, and figures. If no penalties are mentioned, it states clearly no penalties are present.",
          "default": "No termination penalties are mentioned in this contract."
        },
        "otherInformation": {
          "type": "string",
          "description": "Any other relevant information regarding termination found in the contract.",
          "default": "No other relevant information regarding termination."
        },
        "source": {
          "type": "string",
          "description": "The citation for the document name, section, and page where the termination information was extracted.",
          "default": "No clauses regarding termination were found in this contract."
        }
      },
      "required": ["penalties", "otherInformation", "source"]
    }
  },
  "required": ["terminationClauses"]
}

""",
    # 8. Quality and KPI
    """
Summarize the quality requirements and key performance indicators (KPIs) specified in the contract. Include all relevant metrics and standards. Also check the appendices of the document. Bold key information that need to be highlighted, such as dates, percentages, amount, figures, among others. Cite the document name, section, and page under which you got this information from under the keyword 'Source'.

{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "type": "object",
  "properties": {
    "qualityandKPI": {
      "type": "object",
      "properties": {
        "qualityRequirements": {
          "type": "string",
          "description": "Summary of the quality requirements specified in the contract, including metrics and standards in a few sentences.",
          "default": "No specific quality requirements mentioned in this contract."
        },
        "keyPerformanceIndicators": {
          "type": "array",
          "items": {
            "type": "string"
          },
          "description": "List of KPIs with details such as target values, dates, percentages, and figures that need to be highlighted.",
          "default": []
        },
        "source": {
          "type": "string",
          "description": "The citation for the document name, section, and page where the quality and KPI information was extracted.",
          "default": "No KPI or quality requirements information found in this contract."
        }
      },
      "required": ["qualityRequirements", "keyPerformanceIndicators", "source"]
    }
  },
  "required": ["qualityandKPI"]
}

""",
    # 9. Dispute Resolution
    """
Please analyze the contract and accompanying documents to extract a brief summary of the provisions related to dispute resolution costs.
Particularly focus on any financial obligations or penalties associated with resolving disputes under this contract.
Your summary should be brief but informative, including key figures and values.
Organize your summary under the title 'Dispute Resolution Costs'. Bold any significant figures such as dates, percentages, and monetary amounts.
Conclude by citing the document name, section, and page number under the keyword 'Source'.

{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "type": "object",
  "properties": {
    "disputeResolutionCosts": {
      "type": "object",
      "properties": {
        "summary": {
          "type": "string",
          "description": "A brief but informative summary of the financial obligations or penalties associated with resolving disputes, including significant figures such as dates, percentages, and monetary amounts.",
          "default": "No specific penalties or costs detailed for dispute resolutions; general framework provided."
        },
        "source": {
          "type": "string",
          "description": "The citation for the document name, section, and page where the dispute resolution costs information was extracted.",
          "default": "Source information not available."
        }
      },
      "required": ["summary", "source"]
    }
  },
  "required": ["disputeResolutionCosts"]
}


""",
    # 10. Modified standard clauses detection
    """
This is an NEC contract. All non-standard clauses are supposed to be stated in Option Z.
Check whether the standard clauses in the other part of the contract have been modified.
If so, numerically list the modified clauses, without the content of the clauses verbatim.

{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "type": "object",
  "properties": {
    "modifiedStandardClauses": {
      "type": "object",
      "properties": {
        "listOfModifiedClauses": {
          "type": "array",
          "items": {
            "type": "string"
          },
          "description": "Numerically listed modified standard clauses, identified by their clause numbers or identifiers, without the content of the clauses verbatim.",
          "default": []
        },
        "source": {
          "type": "string",
          "description": "The citation for the document name, section, and page where the information about modified standard clauses was extracted.",
          "default": "No modifications to standard clauses were detected."
        }
      },
      "required": ["listOfModifiedClauses", "source"]
    }
  },
  "required": ["modifiedStandardClauses"]
}

""",
    # 11. Analyze option Z
    """
Review Option Z of the contract, which contains non-standard clauses. Identify and numerically list only those clauses within Option Z that are deemed high-risk. High-risk clauses should be determined based on the following 15 key words:  Payment, Delay, Damages, Quality, Milestone, Completion, Delivery, Performance, Incentive, Budget, Corruption, HSE, Key Date, Site Information, Disallowed Cost, Defined Cost, Works Information, Scope, Prices, Compensation, Claims, Insurance, Risk. List these clauses without including their verbatim content. Organize your findings under the title 'High Risk Clauses.

{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "type": "object",
  "properties": {
    "highRiskClauses": {
      "type": "object",
      "properties": {
        "clauses": {
          "type": "array",
          "items": {
            "type": "object",
            "properties": {
              "clauseIdentifier": {
                "type": "string",
                "description": "The identifier or number of the high-risk clause within Option Z."
              },
              "keywords": {
                "type": "array",
                "items": {
                  "type": "string"
                },
                "description": "List of keywords indicating why the clause is considered high-risk."
              }
            },
            "required": ["clauseIdentifier", "keywords"]
          },
          "description": "A numerically ordered list of high-risk clauses identified within Option Z.",
          "default": []
        },
        "title": {
          "type": "string",
          "description": "The title for the section where findings are organized.",
          "default": "High Risk Clauses"
        }
      },
      "required": ["clauses", "title"]
    }
  },
  "required": ["highRiskClauses"]
}
""",
    # 13. Non conformance penalty
    """
Carefully review the contract and its supporting documents to extract a summary of the penalties for non-conformance.
Emphasize the conditions under which these penalties apply, and the criteria for assessing non-conformance.
Your summary should be brief but informative, including key figures, values, and dates.
Organize your findings under the title 'Non-conformance Penalties'. Bold any significant figures such as dates, percentages, and monetary amounts.
If the contract does not specify non-conformance penalties, note this absence and outline the general non-conformance guidelines instead. Finally, cite the document name, section, and page number under the keyword 'Source'.

{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "type": "object",
  "properties": {
    "nonconformancePenalties": {
      "type": "object",
      "properties": {
        "summary": {
          "type": "string",
          "description": "A brief but informative summary of the penalties for non-conformance, including conditions, criteria, and key figures such as dates, percentages, and monetary amounts.",
          "default": "No specific non-conformance penalties specified; general guidelines provided instead."
        },
        "keyFigures": {
          "type": "array",
          "items": {
            "type": "string"
          },
          "description": "List of significant figures such as dates, percentages, monetary amounts, and other pertinent metrics related to non-conformance.",
          "default": []
        },
        "source": {
          "type": "string",
          "description": "The citation for the document name, section, and page where the non-conformance penalties information was extracted.",
          "default": "Source information not available."
        }
      },
      "required": ["summary", "keyFigures", "source"]
    }
  },
  "required": ["nonconformancePenalties"]
}
""",
    # 14. Contradiction analysis
    """
List the contradictory clauses found within this contract in the below JSON schema.
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "type": "object",
  "properties": {
    "contradictions": {
      "type": "object",
      "patternProperties": {
        "Clause_ID": {
          "type": "array",
          "items": {
            "type": "string",
            "description": "Clause number that contradicts the key clause"
          },
          "description": "List of clauses that contradict the key clause",
          "default": [],
          "uniqueItems": true
        }
      },
      "description": "Mapping of clause numbers to lists of clause numbers they significantly contradict with. Only include contradictions that are significant, meaning complete opposites. Don't include vague, ambiguous, and other types of problematic clauses. Only contradictory ones that are likely to have a major impact on the client and contractor.",
      "default": {}
    }
  },
  "required": ["contradictions"]
}

However, if the provided context is not a contract or does not have relevant information asked above, please use the default statements to inform that this is not a full contract with the relevant information.

""",
    # 15. Ambiguity analysis
    """

  {
  "$schema": "http://json-schema.org/draft-07/schema#",
  "type": "object",
  "properties": {
    "ambiguousClauses": {
      "type": "object",
      "properties": {
        "ambiguousClausesList": {
          "type": "array",
          "items": {
            "type": "string",
            "description": "The clause number of a highly ambiguous clause within the contract."
          },
          "description": "A list of the most significant ambiguous clauses identified within the contract, presented in a numbered format.",
          "default": []
        }
      },
      "required": ["ambiguousClausesList"]
    }
  },
  "required": ["ambiguousClauses"]
}

""",
    # 16 duplicate clauses
    """

Go through the whole document and identify duplicate clauses found. Output your answer in numbered format and list only the pairs of duplicate clause numbers.
The title of your output should be "Duplicate Clauses". If no duplicates are found, state: "No duplicate clauses found."

{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "type": "object",
  "properties": {
    "duplicateClauses": {
      "type": "array",
      "items": {
        "type": "array",
        "items": {
          "type": "string",
          "description": "Clause number"
        },
        "minItems": 2,
        "maxItems": 2,
        "description": "A pair of duplicate clauses"
      },
      "description": "List of pairs of duplicate clauses. Each pair should be unique and not repeated. If no duplicates are found, the array is empty.",
      "default": [],
      "uniqueItems": true
    }
  },
  "required": ["duplicateClauses"]
}

""",
    # 17. Payment terms 1
    """
Please provide a detailed summary of the following aspects in the contract: valuation, assessment, retention, components.
Title your summary 'Payment Terms.' Ensure that all pertinent details and specifics are included.
At the end, reference the document name, section, and page number using the keyword 'Source.'
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "type": "object",
  "properties": {
    "paymentTerms1": {
      "type": "object",
      "properties": {
        "valuation": {
          "type": "string",
          "description": "Describes the process and criteria for determining the financial value of work completed under the contract in a maximum of two sentences.",
          "default": "Valuation details not available."
        },
        "assessment": {
          "type": "string",
          "description": "Details the procedures for reviewing, approving, and certifying payment amounts due under the contract in a maximum of two sentences.",
          "default": "Assessment procedures not specified."
        },
        "retention": {
          "type": "string",
          "description": "Outlines the conditions under which a portion of the payment is withheld as security against defects or incomplete work in a maximum of two sentences.",
          "default": "Retention details not provided."
        },
        "components": {
          "type": "string",
          "description": "Specifies any particular parts of the work or materials that have unique pricing or valuation conditions in a maximum of two sentences.",
          "default": "No specific components outlined."
        },
        "source": {
          "type": "string",
          "description": "The citation for the document name, section, and page where the payment terms information was extracted in a maximum of two sentences.",
          "default": "Source information not available."
        }
      },
      "required": ["valuation", "assessment", "retention", "components", "source"]
    }
  },
  "required": ["paymentTerms1"]
}
""",
    # 18. Payment terms 2
    """
Provide a comprehensive summary of these aspects of the contract: adjustments, costs, terms, and the specified currency for defined costs.
Title your summary 'Payment Terms.' Ensure that all pertinent details and specifics are included.
At the end, reference the document name, section, and page number using the keyword 'Source.'
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "type": "object",
  "properties": {
    "paymentTerms2": {
      "type": "object",
      "properties": {
        "adjustments": {
          "type": "string",
          "description": "Details of any adjustments that can be made to payments, including conditions and methods. Make it concise; in a maximum of 2-3 sentences.",
          "default": "Adjustment details not available."
        },
        "costs": {
          "type": "string",
          "description": "Brief summary of the costs covered under the contract, including how costs are calculated and what expenses are included or excluded. Make it concise; about 2-3 sentences and nothing more.",
          "default": "Cost details not provided."
        },
        "terms": {
          "type": "string",
          "description": "Breifly summarize the specific terms related to payment schedules, late payment penalties, and other relevant payment condition. Make it concise; about 2-3 sentences and nothing more.",
          "default": "Payment terms not specified."
        },
        "specifiedCurrency": {
          "type": "string",
          "description": "The currency used for defining and settling all costs under the contract.",
          "default": "Currency details not specified."
        },
        "source": {
          "type": "string",
          "description": "The citation for the document name, section, and page where the payment terms information was extracted.",
          "default": "Source information not available."
        }
      },
      "required": ["adjustments", "costs", "terms", "specifiedCurrency", "source"]
    }
  },
  "required": ["paymentTerms2"]
}
""",
]


# TODO add loggers here
def high_risk_clauses_review(json_data):
    logger = get_logger()
    "Prompts for contract review - high risk"
    prompts = []
    if "highRiskClauses" in json_data:
        high_risk_clauses = json_data["highRiskClauses"]
        if "clauses" in high_risk_clauses:
            clauses = high_risk_clauses["clauses"]
            for clause in clauses:
                if "clauseIdentifier" in clause and "keywords" in clause:
                    clause_id = clause["clauseIdentifier"]
                    keywords = ", ".join(clause["keywords"])
                    prompt = f"""Briefly summarize clause {clause_id} in 2 lines. It is deemed to be a high risk clause of the following keywords: {keywords}. Briefly state the impact this high risk clause brings to the client (impact). State this clause's impact level to the client with one word only: high, medium, low (impactLevel). The following risk trackers exist: operation, time, legal, financial, quality. Determine the type of Risk (s) (RiskType). Using these trackers, briefly explain the type of risk this clause brings to the client (risk).  Give very specific answers and avoid generalized, vague or common sense answers. Keep the answers short and conscise. Avoid vague, generalized, common-sense answers.
                    Output your answer in the following JSON struct:  Mind you that the clauses should be a dictionary within a list.
"clauses":{{"id": "", "summary":"a very brief summary of clause {clause_id}"}}[],
    "impact": "very briefly explain the impact of this high risk clause",
    "impactLevel": "",
    "riskType": [],
    "risk": "Very briefly state the risk this high risk clause brings to the client"
    }},
"""
                    prompts.append(prompt)
                else:
                    logger.warning("Missing 'clauseIdentifier' or 'keywords' in some clauses.")

    return prompts


def contradictory_clauses_review(json_data):
    "Prompts for contract review - contradictory clauses"
    logger = get_logger()
    prompts = []
    if "contradictions" not in json_data:
        logger.warning("No 'contradictions' found in analysis result.")
        return []

    for clause, contradicted_clauses in json_data["contradictions"].items():
        for contradicted_clause in contradicted_clauses:
            prompt = f"""Very briefly summarize clause {clause} and clause {contradicted_clause}. There exists a contradiction between clause {clause} and clause {contradicted_clause}.
Very briefly state why they are contradictory and the impact it brings to the client (impact).
State this clause's impact level to the client with one word only: high, medium, low (impactLevel).
The following risk trackers exist: Operation, time, legal, financial, quality.
State the type of risk(s) this clause brings to the client (riskType).
Determine the type of risk the contradiction brings to the client with a very brief explanation (risk).
Give very specific answers and avoid generalized, vague, or common-sense answers. Keep the answers short and concise.
Your output should be in the following JSON struct:
"clauses": [{{
    "id": "{clause}",
    "summary": ""
}}, {{
    "id": "{contradicted_clause}",
    "summary": ""
}}],
"impact": "",
"impactLevel": "",
"riskType": [],
"risk": ""
}}"""
            prompts.append(prompt)

    return prompts


def ambiguous_clauses_review(json_data):
    "Prompts for contract review - ambiguous clauses"
    logger = get_logger()
    prompts = []
    if "ambiguousClauses" not in json_data:
        logger.warning("Missing 'ambiguousClauses' in json_data.")
        return []

    ambiguous_section = json_data["ambiguousClauses"]
    ambiguous_list = ambiguous_section.get("ambiguousClausesList", [])
    if not isinstance(ambiguous_list, list):
        logger.warning("'ambiguousClausesList' is not a list.")
        return []

    for clause in ambiguous_list:
        if not clause:
            logger.warning("Empty clause found in 'ambiguousClausesList'; skipping.")
            continue

        prompt = f"""Very briefly summarize clause {clause} in two lines. It has been deemed to be ambiguous. Very briefly describe why it was flagged as such and the impact it brings to the client(impact). State this clause's impact level to the client with one word only: high, medium, low (impactLevel). The following risk trackers exist: Operation, time, legal, financial, quality. State the type of risk(s) this clause brings to the client (riskType). Very briefly explain the risk of what this ambiguity brings to the client (risk). Keep your output conscise and avoid vague, generalized, common-sense answers.
                    Output your answer in the following JSON struct:
         "clauses":{{"id": "", "summary":""}}[],
    "impact": "",
    "impactLevel": "",
    "riskType":[],
    "risk": ""
    }},
"""
        prompts.append(prompt)

    return prompts


def duplicate_clauses_review(json_data):
    "Prompts for contract review - duplicate clauses"
    logger = get_logger()
    prompts = []
    if "duplicateClauses" in json_data:
        duplicate_groups = json_data["duplicateClauses"]
        if not isinstance(duplicate_groups, list):
            logger.warning("'duplicateClauses' is not a list.")
            return []
        for group in duplicate_groups:
            clause_1, clause_2 = group

            prompt = f"""Very briefly summarize clause {clause_1} and clause {clause_2}. Very briefly state the problem of this duplication and the impact it brings to the client (impact). State this clause's impact level to the client with one word only: high, medium, low (impactLevel). The following risk trackers exist: Operation, time, legal, financial, quality. State the type of risk(s) this clause brings to the client (riskType). Very briefly explain the type of risk this duplication brings to the client (risk). Keep your output conscise and avoid vague, generalized, common-sense answers.
                    Output your answer in the following JSON struct:
                    "clauses":{{"id": "", "summary":""}}[],
    "impact": "",
    "impactLevel": "",
    "riskType":[],
    "risk": ""
    }},
                    """
            prompts.append(prompt)
    return prompts


def modified_clauses_review(json_data):
    "Prompts for contract review - modified standard clauses"
    logger = get_logger()
    prompts = []
    if "modifiedStandardClauses" not in json_data:
        logger.warning("Missing 'modifiedStandardClauses' in json_data.")
        return []
    modified_clauses = json_data["modifiedStandardClauses"].get("listOfModifiedClauses", [])

    for clause in modified_clauses:
        prompt = f"""Very briefly summarize clause {clause} in two lines. It has been deemed to be modified from the standard. Very briefly describe why it was flagged as such and the impact it brings to the client (impact). State this clause's impact level to the client with one word only: high, medium, low (impactLevel). The following risk trackers exist: Operation, time, legal, financial, quality. State the type of risk(s) this clause brings to the client (riskType). Briefly explain the risk of what this ambiguity brings to the client with a very brief explanation (risk). Keep your output conscise and avoid vague, generalized, common-sense answers.
                    Output your answer in the following JSON struct:
         "clauses":{{"id": "", "summary":""}}[],
    "impact": "",
    "impactLevel": "",
    "riskType":[],
    "risk": ""
    }}
"""
        prompts.append(prompt)

    return prompts


def consolidate_responses(responses, clause_type):
    "Consolidates all responses for contract review"
    consolidated_responses = {}

    for response in responses:
        if not response.get("clauses") or len(response["clauses"]) == 0:
            continue
        main_clause_id = response["clauses"][0]["id"]
        main_clause_summary = response["clauses"][0]["summary"]

        if main_clause_id not in consolidated_responses:
            consolidated_responses[main_clause_id] = {
                "mainClause": {
                    "id": main_clause_id,
                    "summary": main_clause_summary,
                    "source": {},
                },
                "impact": {},
                "impactLevel": {},
                "riskType": {},
                "risk": {},
                "type": clause_type,
            }

        if len(response["clauses"]) > 1:
            consolidated_responses[main_clause_id]["clauses"] = []
            for clause in response["clauses"][1:]:
                clause_id = clause["id"]
                clause_llm_summary = clause["summary"]
                consolidated_responses[main_clause_id]["clauses"].append(
                    {"id": clause_id, "summary": clause_llm_summary, "source": {}}
                )

                consolidated_responses[main_clause_id]["impact"][clause_id] = response[
                    "impact"
                ]
                consolidated_responses[main_clause_id]["impactLevel"][clause_id] = (
                    response["impactLevel"]
                )
                consolidated_responses[main_clause_id]["riskType"][clause_id] = (
                    response["riskType"]
                )
                consolidated_responses[main_clause_id]["risk"][clause_id] = response[
                    "risk"
                ]

        if len(response["clauses"]) == 1:
            clause_id = main_clause_id
            consolidated_responses[main_clause_id]["impact"][clause_id] = response[
                "impact"
            ]
            consolidated_responses[main_clause_id]["impactLevel"][clause_id] = response[
                "impactLevel"
            ]
            consolidated_responses[main_clause_id]["riskType"][clause_id] = response[
                "riskType"
            ]
            consolidated_responses[main_clause_id]["risk"][clause_id] = response["risk"]

    return list(consolidated_responses.values())


def retrieve_clause_details(clause_id):
    "Retrieves the exact word and clause number for clauses with detected issues"
    prompt = f"""
    Locate and extract only the first two sentences of clause {clause_id} verbatim.
    Format your response as JSON in the following structure:
    {{
        "clauseNumber": "{clause_id}",
        "clauseText": "first two sentences of the clause verbatim",
    }}
    """
    return prompt


def validate_source_response(response):
    "Validates the response for contract review for the function that looks for source (highlight)"
    required_keys = ["clauseNumber", "clauseText"]
    try:
        response_data = json.loads(response)
        for key in required_keys:
            if key not in response_data:
                print(f"Missing key: {key} in response data")

                return False, f"Missing key: {key}"
        response_data["page"] = "Not Found"
        response_data["sourceDocumentName"] = "Not Found"
        if not isinstance(response_data["clauseNumber"], str):
            return False, "clauseNumber must be a string."
        if not isinstance(response_data["clauseText"], str):
            return False, "clauseText must be a string."

        return True, response_data
    except json.JSONDecodeError:
        return False, "Response is not valid JSON."

ANALYZER_SYS  = """
        You are a contract analyst specialist in the construction industry. Your task is to thoroughly review all provided information and respond to queries accurately. Use British English for all responses. You are already programmed to output in JSON format. Structure your answers such that it is consistent for backend parsing. Use markdown notation. Focus only on key information and avoid being verbose. Ensure to **bold** crucial details such as dates, figures, numbers, and percentages. Maintain clarity and precision in your analysis.
        """
REVIEW_SYS  = (
            "You are a contract analyst specialist in the construction industry. "
            "Your task is to thoroughly review all provided information and respond accurately. "
            "Use British English. Output JSON only - no markdown."
        )
