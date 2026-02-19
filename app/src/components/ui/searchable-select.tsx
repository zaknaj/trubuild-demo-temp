import { useState, useMemo, useRef, useEffect, useCallback } from "react"
import { CheckIcon } from "lucide-react"
import { cn } from "@/lib/utils"
import { Popover, PopoverContent, PopoverAnchor } from "@/components/ui/popover"

export interface SearchableSelectItem {
  code: string
  name: string
  emoji?: string
}

export interface SearchableSelectProps<T extends SearchableSelectItem> {
  items: readonly T[]
  value: string
  onValueChange: (value: string) => void
  disabled?: boolean
  className?: string
  placeholder?: string
  emptyMessage?: string
  defaultEmoji?: string
  /** Display format for the selected item in the dropdown list */
  renderItem?: (item: T) => React.ReactNode
  /** Display value when closed (defaults to item name) */
  getDisplayValue?: (item: T) => string
  /** Value to show in search input when focused (defaults to display value) */
  getSearchValue?: (item: T) => string
  /** Custom filter function. Return priority number (lower = higher priority) or -1 to exclude */
  filterFn?: (item: T, query: string) => number
  variant?: "default" | "dark"
}

/**
 * A generic searchable select component with keyboard navigation.
 * Can be customized for different data types (countries, currencies, etc.)
 */
export function SearchableSelect<T extends SearchableSelectItem>({
  items,
  value,
  onValueChange,
  disabled,
  className,
  placeholder = "Select...",
  emptyMessage = "No results found.",
  defaultEmoji = "🔍",
  renderItem,
  getDisplayValue,
  getSearchValue,
  filterFn,
  variant = "default",
}: SearchableSelectProps<T>) {
  const [open, setOpen] = useState(false)
  const [search, setSearch] = useState("")
  const inputRef = useRef<HTMLInputElement>(null)
  const listRef = useRef<HTMLDivElement>(null)
  const blurTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null)
  const [highlightedIndex, setHighlightedIndex] = useState(0)

  const selectedItem = items.find((item) => item.code === value)

  // Cleanup timeout on unmount
  useEffect(() => {
    return () => {
      if (blurTimeoutRef.current) {
        clearTimeout(blurTimeoutRef.current)
      }
    }
  }, [])

  const displayValue = getDisplayValue
    ? selectedItem
      ? getDisplayValue(selectedItem)
      : ""
    : (selectedItem?.name ?? "")

  const searchValue = getSearchValue
    ? selectedItem
      ? getSearchValue(selectedItem)
      : ""
    : displayValue

  // When open, show search value; when closed, show selected display value
  const inputValue = open ? search : displayValue

  // Default filter: prioritize by code match, name starts with, name contains, code contains
  const defaultFilter = useCallback((item: T, query: string): number => {
    const q = query.toLowerCase()
    const codeLower = item.code.toLowerCase()
    const nameLower = item.name.toLowerCase()

    if (codeLower === q) return 0
    if (codeLower.startsWith(q)) return 1
    if (nameLower.startsWith(q)) return 2
    if (nameLower.includes(q)) return 3
    if (codeLower.includes(q)) return 4
    return -1
  }, [])

  const filteredItems = useMemo(() => {
    if (!search.trim()) return [...items]

    const query = search.toLowerCase().trim()
    const filter = filterFn ?? defaultFilter

    const results: Array<{ item: T; priority: number }> = []
    for (const item of items) {
      const priority = filter(item, query)
      if (priority >= 0) {
        results.push({ item, priority })
      }
    }

    return results.sort((a, b) => a.priority - b.priority).map((r) => r.item)
  }, [search, items, filterFn, defaultFilter])

  useEffect(() => {
    setHighlightedIndex(0)
  }, [filteredItems])

  const handleSelect = (item: T) => {
    onValueChange(item.code)
    setOpen(false)
    setSearch("")
    inputRef.current?.blur()
  }

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (!open) {
      if (e.key === "ArrowDown" || e.key === "ArrowUp" || e.key === "Enter") {
        setOpen(true)
        e.preventDefault()
      }
      return
    }

    switch (e.key) {
      case "ArrowDown":
        e.preventDefault()
        setHighlightedIndex((i) => Math.min(i + 1, filteredItems.length - 1))
        break
      case "ArrowUp":
        e.preventDefault()
        setHighlightedIndex((i) => Math.max(i - 1, 0))
        break
      case "Enter":
        e.preventDefault()
        if (filteredItems[highlightedIndex]) {
          handleSelect(filteredItems[highlightedIndex])
        }
        break
      case "Escape":
        setOpen(false)
        setSearch("")
        inputRef.current?.blur()
        break
    }
  }

  // Scroll highlighted item into view
  useEffect(() => {
    if (open && listRef.current) {
      const highlighted = listRef.current.querySelector(
        "[data-highlighted=true]"
      )
      highlighted?.scrollIntoView({ block: "nearest" })
    }
  }, [highlightedIndex, open])

  const defaultRenderItem = (item: T) => (
    <>
      {item.emoji && <span>{item.emoji}</span>}
      <span>{item.name}</span>
    </>
  )

  return (
    <Popover open={open}>
      <PopoverAnchor asChild>
        <div
          className={cn(
            "flex h-9 w-full items-center gap-2 rounded-md border px-3 text-sm",
            "focus-within:outline-none focus-within:ring-2 focus-within:ring-offset-2",
            disabled && "cursor-not-allowed opacity-50",
            variant === "dark"
              ? "text-white border-white/30 bg-white/20 focus-within:border-white/50 focus-within:ring-white/20"
              : "border-input bg-background focus-within:ring-ring",
            className
          )}
        >
          <span className="text-base leading-none shrink-0">
            {selectedItem?.emoji ?? defaultEmoji}
          </span>
          <input
            ref={inputRef}
            type="text"
            disabled={disabled}
            value={inputValue}
            onChange={(e) => {
              setSearch(e.target.value)
              if (!open) setOpen(true)
            }}
            onFocus={() => {
              // Cancel any pending blur
              if (blurTimeoutRef.current) {
                clearTimeout(blurTimeoutRef.current)
                blurTimeoutRef.current = null
              }
              setSearch(searchValue)
              setOpen(true)
            }}
            onBlur={() => {
              // Delay to allow click on item
              blurTimeoutRef.current = setTimeout(() => {
                setOpen(false)
                setSearch("")
              }, 50)
            }}
            onKeyDown={handleKeyDown}
            placeholder={placeholder}
            className={cn(
              "flex-1 bg-transparent outline-none min-w-0",
              variant === "dark"
                ? "placeholder:text-white/40"
                : "placeholder:text-muted-foreground"
            )}
          />
        </div>
      </PopoverAnchor>
      <PopoverContent
        className="p-1 max-h-60 overflow-auto"
        style={{ width: "var(--radix-popover-trigger-width)" }}
        align="start"
        onOpenAutoFocus={(e) => e.preventDefault()}
      >
        <div ref={listRef}>
          {filteredItems.length === 0 ? (
            <div className="py-6 text-center text-sm text-muted-foreground">
              {emptyMessage}
            </div>
          ) : (
            filteredItems.map((item, index) => (
              <button
                key={item.code}
                type="button"
                data-highlighted={index === highlightedIndex}
                className={cn(
                  "flex w-full items-center gap-2 rounded-sm px-2 py-1.5 text-sm outline-none cursor-default",
                  index === highlightedIndex && "bg-muted",
                  value === item.code && "font-medium"
                )}
                onMouseEnter={() => setHighlightedIndex(index)}
                onMouseDown={(e) => {
                  e.preventDefault()
                  handleSelect(item)
                }}
              >
                {renderItem ? renderItem(item) : defaultRenderItem(item)}
                {value === item.code && (
                  <CheckIcon className="ml-auto h-4 w-4" />
                )}
              </button>
            ))
          )}
        </div>
      </PopoverContent>
    </Popover>
  )
}
