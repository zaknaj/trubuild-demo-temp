import { clsx, type ClassValue } from "clsx"
import { twMerge } from "tailwind-merge"
import { DEFAULT_COUNTRY_CODE } from "@/components/CountrySelect"

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs))
}

/** Parse country from org metadata, falling back to DEFAULT_COUNTRY_CODE */
export function getOrgCountry(metadata: string | object | null | undefined): string {
  if (!metadata) return DEFAULT_COUNTRY_CODE
  const parsed = typeof metadata === "string" ? JSON.parse(metadata) : metadata
  return (parsed?.country as string) ?? DEFAULT_COUNTRY_CODE
}

/** Format a number as currency (USD by default) */
export function formatCurrency(
  amount: number,
  currency = "USD",
  locale = "en-US"
): string {
  return new Intl.NumberFormat(locale, {
    style: "currency",
    currency,
    minimumFractionDigits: 0,
    maximumFractionDigits: 0,
  }).format(amount)
}
