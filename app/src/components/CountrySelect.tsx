import { COUNTRIES } from "@/lib/countries"
import { SearchableSelect } from "@/components/ui/searchable-select"

export const DEFAULT_COUNTRY_CODE = "SA" as const

export function CountrySelect({
  value,
  onValueChange,
  disabled,
  className,
  variant = "default",
}: {
  value: string
  onValueChange: (value: string) => void
  disabled?: boolean
  className?: string
  variant?: "default" | "dark"
}) {
  return (
    <SearchableSelect
      items={COUNTRIES}
      value={value}
      onValueChange={onValueChange}
      disabled={disabled}
      className={className}
      variant={variant}
      placeholder="Select country..."
      emptyMessage="No country found."
      defaultEmoji="🌐"
    />
  )
}
