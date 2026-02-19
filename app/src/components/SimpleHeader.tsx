interface SimpleHeaderProps {
  title: string
}

export function SimpleHeader({ title }: SimpleHeaderProps) {
  return (
    <div className="flex items-center justify-between border-b-[0.5px] border-black/15 h-12 pl-7 pr-2">
      <h1 className="text-16 font-semibold w-[280px] truncate">{title}</h1>
    </div>
  )
}
