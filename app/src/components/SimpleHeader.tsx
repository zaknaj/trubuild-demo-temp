interface SimpleHeaderProps {
  title: string
}

export function SimpleHeader({ title }: SimpleHeaderProps) {
  return (
    <div className="flex items-center h-12 px-6">
      <h1 className="text-[14px] font-medium text-black truncate">{title}</h1>
    </div>
  )
}
