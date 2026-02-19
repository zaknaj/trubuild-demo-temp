import type {
  BOQData,
  BOQDivision,
  BOQSection,
  BOQLineItem,
  ContractorBid,
  CommercialEvaluationData,
  ArithmeticError,
  NormalizationSettings,
  CustomOverrides,
} from "./types"

// Realistic construction division templates based on CSI MasterFormat
const DIVISION_TEMPLATES = [
  {
    code: "02",
    name: "Sitework",
    sections: [
      { code: "02.100", name: "Site Preparation", items: ["Clearing and grubbing", "Demolition of existing structures", "Tree removal", "Site grading", "Excavation", "Backfill and compaction", "Erosion control measures", "Temporary fencing", "Site survey and layout", "Topsoil stripping"] },
      { code: "02.200", name: "Earthwork", items: ["Bulk excavation", "Trench excavation", "Rock excavation", "Fill material supply", "Compacted fill", "Gravel bedding", "Sand bedding", "Soil stabilization", "Dewatering", "Shoring and bracing"] },
      { code: "02.300", name: "Site Utilities", items: ["Storm drainage piping", "Sanitary sewer piping", "Water main installation", "Fire hydrant installation", "Manholes and catch basins", "Utility trenching", "Pipe bedding material", "Utility markers", "Valve boxes", "Thrust blocks"] },
      { code: "02.400", name: "Roads and Paving", items: ["Subgrade preparation", "Aggregate base course", "Asphalt paving", "Concrete paving", "Curbs and gutters", "Sidewalks", "Parking lot striping", "Speed bumps", "Concrete barriers", "Pavement markings"] },
      { code: "02.500", name: "Landscaping", items: ["Topsoil placement", "Seeding and sodding", "Tree planting", "Shrub planting", "Irrigation system", "Landscape edging", "Mulch application", "Retaining walls", "Decorative stone", "Outdoor lighting"] },
    ],
  },
  {
    code: "03",
    name: "Concrete",
    sections: [
      { code: "03.100", name: "Concrete Formwork", items: ["Foundation formwork", "Wall formwork", "Column formwork", "Beam formwork", "Slab formwork", "Stair formwork", "Form release agent", "Form ties and accessories", "Reshoring", "Form stripping"] },
      { code: "03.200", name: "Concrete Reinforcement", items: ["Rebar Grade 60 #4", "Rebar Grade 60 #5", "Rebar Grade 60 #6", "Rebar Grade 60 #8", "Welded wire mesh", "Fiber reinforcement", "Rebar chairs and spacers", "Mechanical splices", "Epoxy coated rebar", "Rebar fabrication"] },
      { code: "03.300", name: "Cast-in-Place Concrete", items: ["Foundation concrete 4000 PSI", "Slab on grade concrete", "Elevated slab concrete", "Column concrete 5000 PSI", "Wall concrete", "Beam concrete", "Concrete pumping", "Concrete finishing", "Concrete curing compound", "Expansion joints"] },
      { code: "03.400", name: "Precast Concrete", items: ["Precast columns", "Precast beams", "Precast double tees", "Precast wall panels", "Precast stairs", "Hollow core planks", "Precast connections", "Grouting precast joints", "Precast erection", "Precast sealants"] },
      { code: "03.500", name: "Concrete Repairs", items: ["Concrete patching", "Crack injection", "Surface grinding", "Concrete resurfacing", "Waterproofing coating", "Anti-carbonation coating", "Structural strengthening", "Epoxy anchors", "Concrete sealer", "Joint repair"] },
    ],
  },
  {
    code: "04",
    name: "Masonry",
    sections: [
      { code: "04.100", name: "Unit Masonry", items: ["CMU 8x8x16 standard", "CMU 8x8x16 lightweight", "CMU 12x8x16 standard", "Face brick standard", "Face brick special shapes", "Fire brick", "Glass block", "Mortar Type S", "Mortar Type N", "Grout for CMU"] },
      { code: "04.200", name: "Masonry Reinforcement", items: ["Horizontal joint reinforcement", "Vertical rebar in CMU", "Bond beams", "Lintel blocks", "Masonry anchors", "Wall ties", "Control joint material", "Expansion joint filler", "Flashing", "Weep holes"] },
      { code: "04.300", name: "Stone Masonry", items: ["Granite veneer", "Limestone panels", "Marble cladding", "Slate flooring", "Stone anchors", "Stone sealant", "Stone pointing", "Cut stone copings", "Stone sills", "Stone thresholds"] },
      { code: "04.400", name: "Masonry Restoration", items: ["Tuckpointing", "Brick replacement", "Stone replacement", "Masonry cleaning", "Graffiti removal", "Water repellent coating", "Crack stitching", "Lintel replacement", "Parapet repair", "Chimney restoration"] },
    ],
  },
  {
    code: "05",
    name: "Metals",
    sections: [
      { code: "05.100", name: "Structural Steel", items: ["Wide flange beams", "Steel columns", "Steel angles", "Steel channels", "Steel plates", "Base plates", "High strength bolts", "Anchor bolts", "Shear studs", "Steel erection"] },
      { code: "05.200", name: "Steel Joists and Deck", items: ["Open web steel joists", "Joist girders", "Metal roof deck", "Metal floor deck", "Composite deck", "Deck accessories", "Pour stops", "Deck closures", "Deck welding", "Deck attachment"] },
      { code: "05.300", name: "Metal Fabrications", items: ["Steel stairs", "Steel handrails", "Steel ladders", "Steel gratings", "Steel lintels", "Loose lintels", "Steel bollards", "Steel posts", "Miscellaneous angles", "Steel embeds"] },
      { code: "05.400", name: "Metal Finishes", items: ["Hot-dip galvanizing", "Shop primer painting", "Fireproofing spray", "Intumescent coating", "Powder coating", "Anodizing", "Field touch-up paint", "Rust inhibitor", "Metal polish", "Protective wrap"] },
      { code: "05.500", name: "Ornamental Metals", items: ["Aluminum railings", "Stainless steel railings", "Decorative grilles", "Metal screens", "Aluminum louvers", "Metal canopies", "Flagpoles", "Metal signage frames", "Decorative brackets", "Metal trim"] },
    ],
  },
  {
    code: "07",
    name: "Thermal and Moisture Protection",
    sections: [
      { code: "07.100", name: "Waterproofing", items: ["Below grade waterproofing membrane", "Foundation dampproofing", "Plaza waterproofing", "Waterstops", "Injection grouting", "Drainage board", "Protection board", "Bentonite panels", "Crystalline waterproofing", "Liquid membrane"] },
      { code: "07.200", name: "Insulation", items: ["Batt insulation R-13", "Batt insulation R-19", "Rigid insulation board", "Spray foam insulation", "Blown-in insulation", "Pipe insulation", "Duct insulation", "Acoustic insulation", "Fire safing insulation", "Thermal barrier"] },
      { code: "07.300", name: "Roofing", items: ["TPO roofing membrane", "EPDM roofing membrane", "Built-up roofing", "Modified bitumen roofing", "Standing seam metal roof", "Roof insulation tapered", "Roof insulation flat", "Roof drains", "Roof scuppers", "Roof walkway pads"] },
      { code: "07.400", name: "Sheet Metal", items: ["Metal copings", "Metal fascia", "Metal gravel stops", "Roof expansion joints", "Metal flashings", "Through-wall flashing", "Counterflashing", "Metal gutters", "Metal downspouts", "Splash blocks"] },
      { code: "07.500", name: "Sealants", items: ["Silicone sealant", "Polyurethane sealant", "Butyl sealant", "Backer rod", "Joint primer", "Expansion joint covers", "Fire stop sealant", "Acoustic sealant", "Traffic-grade sealant", "Caulk removal"] },
    ],
  },
  {
    code: "09",
    name: "Finishes",
    sections: [
      { code: "09.100", name: "Metal Framing", items: ["Metal studs 3-5/8\"", "Metal studs 6\"", "Metal track", "Metal furring channels", "Resilient channels", "Metal angles", "Metal clips", "Drywall screws", "Deflection track", "Shaft wall studs"] },
      { code: "09.200", name: "Gypsum Board", items: ["Drywall 5/8\" Type X", "Drywall 1/2\" regular", "Moisture resistant drywall", "Abuse resistant drywall", "Shaft liner board", "Drywall finishing Level 4", "Drywall finishing Level 5", "Corner bead", "Joint compound", "Drywall tape"] },
      { code: "09.300", name: "Tile", items: ["Ceramic floor tile", "Ceramic wall tile", "Porcelain tile", "Glass mosaic tile", "Tile backer board", "Thin-set morite", "Tile grout", "Tile sealer", "Waterproof membrane", "Tile trim pieces"] },
      { code: "09.400", name: "Flooring", items: ["Carpet tile", "Broadloom carpet", "Vinyl composition tile (VCT)", "Luxury vinyl tile (LVT)", "Sheet vinyl flooring", "Rubber flooring", "Epoxy flooring", "Floor preparation", "Floor leveling compound", "Floor transitions"] },
      { code: "09.500", name: "Ceilings", items: ["Acoustic ceiling tiles 2x2", "Acoustic ceiling tiles 2x4", "Suspended ceiling grid", "Drywall ceiling", "Wood ceiling panels", "Metal ceiling panels", "Linear metal ceiling", "Ceiling access panels", "Ceiling diffusers", "Ceiling hangers"] },
      { code: "09.600", name: "Painting", items: ["Primer interior", "Primer exterior", "Latex paint interior walls", "Latex paint ceilings", "Semi-gloss paint doors/trim", "Exterior latex paint", "Epoxy paint", "Stain and sealer", "Clear coat finish", "Specialty coatings"] },
    ],
  },
]

// Helper to generate unique IDs
function generateId(): string {
  return Math.random().toString(36).substring(2, 11)
}

// Helper to get random number in range
function randomInRange(min: number, max: number): number {
  return Math.random() * (max - min) + min
}

// Helper to pick random items from array
function pickRandom<T>(arr: T[], count: number): T[] {
  const shuffled = [...arr].sort(() => Math.random() - 0.5)
  return shuffled.slice(0, count)
}

// Generate realistic base prices for different item types
function generateBasePrice(itemDescription: string): number {
  const lowerDesc = itemDescription.toLowerCase()
  
  // High value items
  if (lowerDesc.includes("steel") || lowerDesc.includes("precast") || lowerDesc.includes("structural")) {
    return randomInRange(5000, 50000)
  }
  // Medium-high value
  if (lowerDesc.includes("concrete") || lowerDesc.includes("membrane") || lowerDesc.includes("roofing")) {
    return randomInRange(2000, 20000)
  }
  // Medium value
  if (lowerDesc.includes("tile") || lowerDesc.includes("flooring") || lowerDesc.includes("insulation")) {
    return randomInRange(1000, 10000)
  }
  // Lower value items
  if (lowerDesc.includes("paint") || lowerDesc.includes("sealant") || lowerDesc.includes("tape")) {
    return randomInRange(200, 2000)
  }
  // Default range
  return randomInRange(500, 8000)
}

// Generate quantity based on unit type
function generateQuantity(unit: string): number {
  switch (unit) {
    case "m2":
    case "sf":
      return Math.round(randomInRange(100, 5000))
    case "m3":
    case "cy":
      return Math.round(randomInRange(10, 500))
    case "lm":
    case "lf":
      return Math.round(randomInRange(50, 1000))
    case "kg":
    case "ton":
      return Math.round(randomInRange(100, 2000))
    case "pcs":
    case "ea":
      return Math.round(randomInRange(5, 200))
    case "ls":
      return 1
    default:
      return Math.round(randomInRange(10, 500))
  }
}

// Pick appropriate unit for item
function pickUnit(itemDescription: string): string {
  const lowerDesc = itemDescription.toLowerCase()
  
  if (lowerDesc.includes("paving") || lowerDesc.includes("flooring") || lowerDesc.includes("tile") || 
      lowerDesc.includes("roofing") || lowerDesc.includes("membrane") || lowerDesc.includes("ceiling") ||
      lowerDesc.includes("drywall") || lowerDesc.includes("insulation board")) {
    return "m2"
  }
  if (lowerDesc.includes("concrete") || lowerDesc.includes("excavation") || lowerDesc.includes("fill") ||
      lowerDesc.includes("grout") || lowerDesc.includes("backfill")) {
    return "m3"
  }
  if (lowerDesc.includes("pipe") || lowerDesc.includes("rebar") || lowerDesc.includes("channel") ||
      lowerDesc.includes("track") || lowerDesc.includes("rail") || lowerDesc.includes("trim")) {
    return "lm"
  }
  if (lowerDesc.includes("steel") && (lowerDesc.includes("beam") || lowerDesc.includes("column"))) {
    return "ton"
  }
  if (lowerDesc.includes("paint") || lowerDesc.includes("sealant") || lowerDesc.includes("compound") ||
      lowerDesc.includes("primer") || lowerDesc.includes("coating")) {
    return "ls"
  }
  
  return "pcs"
}

/**
 * Generates mock BOQ data for a commercial evaluation
 * @param contractors Array of contractors with id and name
 * @returns CommercialEvaluationData with BOQ structure and contractor bids
 */
export function generateMockBOQData(
  contractors: Array<{ id: string; name: string }>
): CommercialEvaluationData {
  // Pick 4-6 random divisions
  const numDivisions = Math.floor(randomInRange(4, 7))
  const selectedDivisions = pickRandom(DIVISION_TEMPLATES, numDivisions)
  
  // Store base prices for each line item to ensure contractor prices are close
  const basePrices: Record<string, number> = {}
  
  // Build BOQ structure
  const divisions: BOQDivision[] = selectedDivisions.map((divTemplate) => {
    // Pick 5-8 sections from each division
    const numSections = Math.floor(randomInRange(5, Math.min(9, divTemplate.sections.length + 1)))
    const selectedSections = pickRandom(divTemplate.sections, numSections)
    
    const sections: BOQSection[] = selectedSections.map((secTemplate) => {
      // Pick 10-20 line items from each section
      const numItems = Math.floor(randomInRange(10, Math.min(21, secTemplate.items.length + 1)))
      const selectedItems = pickRandom(secTemplate.items, numItems)
      
      const lineItems: BOQLineItem[] = selectedItems.map((itemDesc, idx) => {
        const itemId = generateId()
        const unit = pickUnit(itemDesc)
        const basePrice = generateBasePrice(itemDesc)
        basePrices[itemId] = basePrice
        
        return {
          id: itemId,
          code: `${secTemplate.code}.${String(idx + 1).padStart(3, "0")}`,
          description: itemDesc,
          quantity: generateQuantity(unit),
          unit,
        }
      })
      
      return {
        id: generateId(),
        code: secTemplate.code,
        name: secTemplate.name,
        lineItems,
      }
    })
    
    return {
      id: generateId(),
      code: divTemplate.code,
      name: divTemplate.name,
      sections,
    }
  })
  
  const boq: BOQData = { divisions }
  
  // Collect all line item IDs
  const allLineItemIds: string[] = []
  for (const div of divisions) {
    for (const sec of div.sections) {
      for (const item of sec.lineItems) {
        allLineItemIds.push(item.id)
      }
    }
  }
  
  // Generate contractor bids
  const contractorBids: ContractorBid[] = contractors.map((contractor) => {
    const prices: Record<string, number | null> = {}
    const includedItems: string[] = []
    const arithmeticErrors: Record<string, ArithmeticError> = {}
    let totalAmount = 0
    
    for (const itemId of allLineItemIds) {
      const rand = Math.random()
      
      // ~5-8% chance of "Included" item (price included in another item)
      if (rand < 0.06) {
        prices[itemId] = null
        includedItems.push(itemId)
      }
      // ~10-12% chance of missing/unpriced item
      else if (rand < 0.18) {
        prices[itemId] = null
      }
      // ~3-5% chance of arithmetic error
      else if (rand < 0.22) {
        const basePrice = basePrices[itemId]
        const variance = randomInRange(-0.25, 0.25)
        const calculatedPrice = Math.round(basePrice * (1 + variance) * 100) / 100
        // Introduce an error: submitted differs from calculated by 5-15%
        const errorFactor = randomInRange(0.85, 1.15)
        const submittedPrice = Math.round(calculatedPrice * errorFactor * 100) / 100
        
        prices[itemId] = submittedPrice
        arithmeticErrors[itemId] = {
          submitted: submittedPrice,
          calculated: calculatedPrice,
        }
        totalAmount += submittedPrice
      }
      // Normal priced item
      else {
        const basePrice = basePrices[itemId]
        const variance = randomInRange(-0.25, 0.25)
        const price = Math.round(basePrice * (1 + variance) * 100) / 100
        prices[itemId] = price
        totalAmount += price
      }
    }
    
    return {
      contractorId: contractor.id,
      contractorName: contractor.name,
      prices,
      includedItems,
      arithmeticErrors,
      totalAmount: Math.round(totalAmount * 100) / 100,
    }
  })
  
  // Sort contractors by total amount (lowest first)
  contractorBids.sort((a, b) => a.totalAmount - b.totalAmount)
  
  return {
    boq,
    contractors: contractorBids,
  }
}

/** Default normalization settings */
export const DEFAULT_NORMALIZATION_SETTINGS: NormalizationSettings = {
  normalizeUnpriced: true,
  normalizeArithmeticErrors: true,
  algorithm: "median",
}

/**
 * Computes normalized prices based on settings and custom overrides
 * - Respects "Included" items (never normalizes them)
 * - Optionally normalizes unpriced items
 * - Optionally normalizes arithmetic errors
 * - Supports median or lowest price algorithm
 * - Applies custom overrides last (they take precedence)
 */
export function normalizeContractorBids(
  boq: BOQData,
  contractors: ContractorBid[],
  settings: NormalizationSettings = DEFAULT_NORMALIZATION_SETTINGS,
  customOverrides: CustomOverrides = {}
): ContractorBid[] {
  // Collect all line item IDs
  const allLineItemIds: string[] = []
  for (const div of boq.divisions) {
    for (const sec of div.sections) {
      for (const item of sec.lineItems) {
        allLineItemIds.push(item.id)
      }
    }
  }
  
  // For each line item, compute fill value based on algorithm
  const fillValues: Record<string, number> = {}
  for (const itemId of allLineItemIds) {
    // Get valid prices (non-null, not included, not arithmetic error unless we're normalizing those)
    const validPrices = contractors
      .filter((c) => {
        // Exclude included items from calculation
        if ((c.includedItems ?? []).includes(itemId)) return false
        // Only include arithmetic error prices if we're not normalizing them
        if ((c.arithmeticErrors ?? {})[itemId] && settings.normalizeArithmeticErrors) return false
        return true
      })
      .map((c) => (c.prices ?? {})[itemId])
      .filter((p): p is number => p !== null)
      .sort((a, b) => a - b)
    
    if (validPrices.length > 0) {
      if (settings.algorithm === "lowest") {
        fillValues[itemId] = validPrices[0]
      } else {
        // Median
        const mid = Math.floor(validPrices.length / 2)
        fillValues[itemId] = validPrices.length % 2 === 0
          ? (validPrices[mid - 1] + validPrices[mid]) / 2
          : validPrices[mid]
      }
    } else {
      fillValues[itemId] = 0
    }
  }
  
  // Create normalized bids
  const normalizedBids: ContractorBid[] = contractors.map((contractor) => {
    const normalizedPrices: Record<string, number | null> = {}
    let totalAmount = 0
    const includedItems = contractor.includedItems ?? []
    const arithmeticErrors = contractor.arithmeticErrors ?? {}
    const prices = contractor.prices ?? {}
    
    for (const itemId of allLineItemIds) {
      const overrideKey = `${contractor.contractorId}-${itemId}`
      const isIncluded = includedItems.includes(itemId)
      const hasArithmeticError = !!arithmeticErrors[itemId]
      const isUnpriced = prices[itemId] === null && !isIncluded
      
      let price: number | null
      
      // Check for custom override first (takes precedence)
      if (customOverrides[overrideKey] !== undefined) {
        price = customOverrides[overrideKey]
      }
      // Included items stay as null (shown as "Included")
      else if (isIncluded) {
        price = null
      }
      // Normalize unpriced items if setting is enabled
      else if (isUnpriced && settings.normalizeUnpriced) {
        price = fillValues[itemId]
      }
      // Normalize arithmetic errors if setting is enabled
      else if (hasArithmeticError && settings.normalizeArithmeticErrors) {
        price = fillValues[itemId]
      }
      // Keep original price
      else {
        price = prices[itemId]
      }
      
      normalizedPrices[itemId] = price
      totalAmount += price ?? 0
    }
    
    return {
      ...contractor,
      prices: normalizedPrices,
      totalAmount: Math.round(totalAmount * 100) / 100,
    }
  })
  
  // Re-sort by normalized total
  normalizedBids.sort((a, b) => a.totalAmount - b.totalAmount)
  
  return normalizedBids
}
