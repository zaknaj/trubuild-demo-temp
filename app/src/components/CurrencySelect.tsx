import { CURRENCIES } from "@/lib/countries"
import { SearchableSelect } from "@/components/ui/searchable-select"

type Currency = (typeof CURRENCIES)[number]

export function CurrencySelect({
  value,
  onValueChange,
  disabled,
  className,
}: {
  value: string
  onValueChange: (value: string) => void
  disabled?: boolean
  className?: string
}) {
  return (
    <SearchableSelect
      items={CURRENCIES}
      value={value}
      onValueChange={onValueChange}
      disabled={disabled}
      className={className}
      placeholder="Select currency..."
      emptyMessage="No currency found."
      defaultEmoji="💰"
      getDisplayValue={(item) => item.code}
      getSearchValue={(item) => item.code}
      renderItem={(item: Currency) => (
        <>
          <span>{item.emoji}</span>
          <span>{item.code}</span>
          <span className="text-muted-foreground">({item.name})</span>
        </>
      )}
    />
  )
}
