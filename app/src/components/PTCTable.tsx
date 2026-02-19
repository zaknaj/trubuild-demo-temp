import { useState, useMemo } from "react"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { Tabs, TabsList, TabsTrigger, TabsContent } from "@/components/ui/tabs"
import {
  Table,
  TableHeader,
  TableBody,
  TableRow,
  TableHead,
  TableCell,
} from "@/components/ui/table"
import { Input } from "@/components/ui/input"
import { cn } from "@/lib/utils"
import type {
  ContractorPTCs,
  PTCItem,
  PTCCategory,
  PTCStatus,
} from "@/lib/types"
import { Download, Upload, Plus } from "lucide-react"

interface PTCTableProps {
  contractors: ContractorPTCs[]
  onSave: (updatedPTCs: ContractorPTCs[]) => void
  isSaving?: boolean
}

const CATEGORIES: { id: PTCCategory | "general"; label: string }[] = [
  { id: "general", label: "General" },
  { id: "exclusions", label: "Exclusions" },
  { id: "deviations", label: "Deviations" },
  { id: "pricing_anomalies", label: "Pricing Anomalies" },
  { id: "arithmetic_checks", label: "Arithmetic Checks" },
]

export function PTCTable({ contractors, onSave, isSaving }: PTCTableProps) {
  const [localPTCs, setLocalPTCs] = useState<ContractorPTCs[]>(contractors)
  const [selectedContractorId, setSelectedContractorId] = useState<string>(
    contractors[0]?.contractorId ?? ""
  )
  const [selectedCategory, setSelectedCategory] = useState<
    PTCCategory | "general"
  >("general")
  const [editingCell, setEditingCell] = useState<{
    ptcId: string
    field: keyof PTCItem
  } | null>(null)


  // Check if there are unsaved changes
  const isDirty = useMemo(() => {
    return JSON.stringify(localPTCs) !== JSON.stringify(contractors)
  }, [localPTCs, contractors])

  // Get pending count for a contractor
  const getPendingCount = (contractorId: string) => {
    const contractor = localPTCs.find((c) => c.contractorId === contractorId)
    if (!contractor) return 0
    return contractor.ptcs.filter((p) => p.status === "pending").length
  }

  // Get selected contractor
  const selectedContractor = localPTCs.find(
    (c) => c.contractorId === selectedContractorId
  )

  // Get filtered PTCs for current category
  const filteredPTCs = useMemo(() => {
    if (!selectedContractor) return []
    if (selectedCategory === "general") {
      return selectedContractor.ptcs
    }
    return selectedContractor.ptcs.filter(
      (p) => p.category === selectedCategory
    )
  }, [selectedContractor, selectedCategory])

  // Update a PTC field
  const updatePTC = (
    contractorId: string,
    ptcId: string,
    field: keyof PTCItem,
    value: string
  ) => {
    setLocalPTCs((prev) =>
      prev.map((contractor) => {
        if (contractor.contractorId !== contractorId) return contractor
        return {
          ...contractor,
          ptcs: contractor.ptcs.map((ptc) => {
            if (ptc.id !== ptcId) return ptc
            return { ...ptc, [field]: value }
          }),
        }
      })
    )
  }

  // Toggle status
  const toggleStatus = (contractorId: string, ptcId: string) => {
    setLocalPTCs((prev) =>
      prev.map((contractor) => {
        if (contractor.contractorId !== contractorId) return contractor
        return {
          ...contractor,
          ptcs: contractor.ptcs.map((ptc) => {
            if (ptc.id !== ptcId) return ptc
            return {
              ...ptc,
              status: ptc.status === "pending" ? "closed" : "pending",
            }
          }),
        }
      })
    )
  }

  // Add new PTC
  const addNewPTC = () => {
    if (!selectedContractor) return

    const newPTC: PTCItem = {
      id: crypto.randomUUID(),
      referenceSection: "",
      queryDescription: "",
      vendorResponse: "",
      status: "pending",
      category:
        selectedCategory === "general" ? "exclusions" : selectedCategory,
    }

    setLocalPTCs((prev) =>
      prev.map((contractor) => {
        if (contractor.contractorId !== selectedContractorId) return contractor
        return {
          ...contractor,
          ptcs: [...contractor.ptcs, newPTC],
        }
      })
    )

    // Start editing the new row
    setEditingCell({ ptcId: newPTC.id, field: "referenceSection" })
  }

  // Handle save
  const handleSave = () => {
    onSave(localPTCs)
  }

  // Get category count for tabs
  const getCategoryCount = (category: PTCCategory | "general") => {
    if (!selectedContractor) return 0
    if (category === "general") {
      return selectedContractor.ptcs.filter((p) => p.status === "pending")
        .length
    }
    return selectedContractor.ptcs.filter(
      (p) => p.category === category && p.status === "pending"
    ).length
  }

  if (contractors.length === 0) {
    return (
      <div className="flex items-center justify-center h-64 text-muted-foreground">
        No contractors available
      </div>
    )
  }

  return (
    <div className="flex flex-col gap-4">
      {/* Contractor Selector */}
      <div className="flex items-center gap-2">
        <span className="text-xs font-medium text-muted-foreground">
          Contractor
        </span>
        <div className="flex gap-1 p-1 bg-muted rounded-lg w-fit">
          {contractors.map((contractor) => {
            const pendingCount = getPendingCount(contractor.contractorId)
            const isSelected = selectedContractorId === contractor.contractorId

            return (
              <button
                key={contractor.contractorId}
                onClick={() => setSelectedContractorId(contractor.contractorId)}
                className={cn(
                  "px-3 py-1.5 text-sm font-medium rounded-md transition-colors",
                  isSelected
                    ? "bg-background text-foreground shadow-sm"
                    : "text-muted-foreground hover:text-foreground"
                )}
              >
                {contractor.contractorName}
                {pendingCount > 0 && (
                  <span className="ml-1.5 text-orange-600">({pendingCount})</span>
                )}
              </button>
            )
          })}
        </div>
      </div>

      {/* Action Bar */}
      <div className="flex items-center justify-between">
        <div className="flex gap-2">
          <Button variant="outline" size="sm" disabled>
            <Download className="size-4" data-icon="inline-start" />
            Export
          </Button>
          <Button variant="outline" size="sm" disabled>
            <Upload className="size-4" data-icon="inline-start" />
            Upload Responses
          </Button>
        </div>
        {isDirty && (
          <Button onClick={handleSave} size="sm" disabled={isSaving}>
            {isSaving ? "Saving..." : "Save Changes"}
          </Button>
        )}
      </div>

      {/* Category Tabs */}
      <Tabs
        value={selectedCategory}
        onValueChange={(v) => setSelectedCategory(v as PTCCategory | "general")}
      >
        <TabsList variant="line">
          {CATEGORIES.map((cat) => {
            const count = getCategoryCount(cat.id)
            return (
              <TabsTrigger key={cat.id} value={cat.id}>
                {cat.label}
                {count > 0 && (
                  <span className="ml-1 text-orange-600">({count})</span>
                )}
              </TabsTrigger>
            )
          })}
        </TabsList>

        {CATEGORIES.map((cat) => (
          <TabsContent key={cat.id} value={cat.id}>
            <PTCTableContent
              ptcs={filteredPTCs}
              contractorId={selectedContractorId}
              editingCell={editingCell}
              setEditingCell={setEditingCell}
              updatePTC={updatePTC}
              toggleStatus={toggleStatus}
              onAddRow={addNewPTC}
            />
          </TabsContent>
        ))}
      </Tabs>
    </div>
  )
}

interface PTCTableContentProps {
  ptcs: PTCItem[]
  contractorId: string
  editingCell: { ptcId: string; field: keyof PTCItem } | null
  setEditingCell: (cell: { ptcId: string; field: keyof PTCItem } | null) => void
  updatePTC: (
    contractorId: string,
    ptcId: string,
    field: keyof PTCItem,
    value: string
  ) => void
  toggleStatus: (contractorId: string, ptcId: string) => void
  onAddRow: () => void
}

function PTCTableContent({
  ptcs,
  contractorId,
  editingCell,
  setEditingCell,
  updatePTC,
  toggleStatus,
  onAddRow,
}: PTCTableContentProps) {
  return (
    <div className="border rounded-lg">
      <div className="overflow-x-auto">
      <Table className="min-w-[900px]">
        <TableHeader>
          <TableRow>
            <TableHead className="min-w-[160px] w-[160px]">Reference Section</TableHead>
            <TableHead className="min-w-[320px]">Query Description</TableHead>
            <TableHead className="min-w-[320px]">Vendor Response</TableHead>
            <TableHead className="min-w-[100px] w-[100px]">Status</TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          {ptcs.map((ptc) => (
            <TableRow key={ptc.id}>
              <TableCell>
                <EditableCell
                  value={ptc.referenceSection}
                  isEditing={
                    editingCell?.ptcId === ptc.id &&
                    editingCell?.field === "referenceSection"
                  }
                  onEdit={() =>
                    setEditingCell({ ptcId: ptc.id, field: "referenceSection" })
                  }
                  onChange={(value) =>
                    updatePTC(contractorId, ptc.id, "referenceSection", value)
                  }
                  onBlur={() => setEditingCell(null)}
                />
              </TableCell>
              <TableCell>
                <EditableCell
                  value={ptc.queryDescription}
                  isEditing={
                    editingCell?.ptcId === ptc.id &&
                    editingCell?.field === "queryDescription"
                  }
                  onEdit={() =>
                    setEditingCell({ ptcId: ptc.id, field: "queryDescription" })
                  }
                  onChange={(value) =>
                    updatePTC(contractorId, ptc.id, "queryDescription", value)
                  }
                  onBlur={() => setEditingCell(null)}
                />
              </TableCell>
              <TableCell>
                <EditableCell
                  value={ptc.vendorResponse}
                  isEditing={
                    editingCell?.ptcId === ptc.id &&
                    editingCell?.field === "vendorResponse"
                  }
                  onEdit={() =>
                    setEditingCell({ ptcId: ptc.id, field: "vendorResponse" })
                  }
                  onChange={(value) =>
                    updatePTC(contractorId, ptc.id, "vendorResponse", value)
                  }
                  onBlur={() => setEditingCell(null)}
                  placeholder="Enter response..."
                />
              </TableCell>
              <TableCell>
                <StatusBadge
                  status={ptc.status}
                  onClick={() => toggleStatus(contractorId, ptc.id)}
                />
              </TableCell>
            </TableRow>
          ))}
          {ptcs.length === 0 && (
            <TableRow>
              <TableCell
                colSpan={4}
                className="h-24 text-center text-muted-foreground"
              >
                No PTCs in this category
              </TableCell>
            </TableRow>
          )}
        </TableBody>
      </Table>
      </div>
      <div className="border-t p-2">
        <Button
          variant="ghost"
          size="sm"
          onClick={onAddRow}
          className="text-muted-foreground"
        >
          <Plus className="size-4" data-icon="inline-start" />
          Add row
        </Button>
      </div>
    </div>
  )
}

interface EditableCellProps {
  value: string
  isEditing: boolean
  onEdit: () => void
  onChange: (value: string) => void
  onBlur: () => void
  placeholder?: string
}

function EditableCell({
  value,
  isEditing,
  onEdit,
  onChange,
  onBlur,
  placeholder,
}: EditableCellProps) {
  if (isEditing) {
    return (
      <Input
        autoFocus
        value={value}
        onChange={(e) => onChange(e.target.value)}
        onBlur={onBlur}
        onKeyDown={(e) => {
          if (e.key === "Enter" || e.key === "Escape") {
            onBlur()
          }
        }}
        className="h-8 text-sm"
        placeholder={placeholder}
      />
    )
  }

  return (
    <div
      onClick={onEdit}
      className={cn(
        "cursor-pointer min-h-[32px] flex items-center px-1 -mx-1 rounded hover:bg-muted/50",
        !value && "text-muted-foreground/50"
      )}
    >
      {value || placeholder || "Click to edit"}
    </div>
  )
}

interface StatusBadgeProps {
  status: PTCStatus
  onClick: () => void
}

function StatusBadge({ status, onClick }: StatusBadgeProps) {
  return (
    <button onClick={onClick} className="focus:outline-none">
      {status === "pending" ? (
        <Badge className="bg-orange-100 text-orange-800 hover:bg-orange-200 cursor-pointer border-orange-200">
          Pending
        </Badge>
      ) : (
        <Badge className="bg-gray-100 text-gray-800 hover:bg-gray-200 cursor-pointer border-gray-200">
          Closed
        </Badge>
      )}
    </button>
  )
}
