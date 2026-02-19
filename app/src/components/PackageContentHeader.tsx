import { Link } from "@tanstack/react-router"
import { ChevronDown, Plus } from "lucide-react"
import { Button } from "@/components/ui/button"
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu"
import { AIChatButton } from "@/components/AIChatButton"
import { cn } from "@/lib/utils"

type Round = {
  id: string
  roundName: string
}

interface TechnicalHeaderProps {
  variant: "technical"
  packageId: string
  rounds: Round[]
  currentRound?: Round
  onRoundSelect: (roundId: string) => void
  onNewRound: () => void
}

interface CommercialSummaryHeaderProps {
  variant: "commercial-summary"
}

interface AssetHeaderProps {
  variant: "asset"
  packageId: string
  assetId: string
  assetName: string
  rounds: Round[]
  currentRound?: Round
  onRoundSelect: (roundId: string) => void
  onNewRound: () => void
}

type PackageContentHeaderProps =
  | TechnicalHeaderProps
  | CommercialSummaryHeaderProps
  | AssetHeaderProps

function HeaderNavLink({
  to,
  params,
  exact,
  children,
}: {
  to: string
  params: Record<string, string>
  exact?: boolean
  children: React.ReactNode
}) {
  return (
    <Link
      to={to}
      params={params}
      activeOptions={exact ? { exact: true } : undefined}
      className="h-full flex items-center mx-3.5 text-13 font-medium text-foreground/50 hover:text-foreground transition-colors border-b-2 border-transparent data-[status=active]:text-primary data-[status=active]:border-primary"
    >
      {children}
    </Link>
  )
}

function RoundsDropdown({
  rounds,
  currentRound,
  onRoundSelect,
  onNewRound,
}: {
  rounds: Round[]
  currentRound?: Round
  onRoundSelect: (roundId: string) => void
  onNewRound: () => void
}) {
  return (
    <DropdownMenu>
      <DropdownMenuTrigger asChild>
        <Button
          variant="ghost"
          size="sm"
          className="h-7 gap-1 text-13 font-normal text-muted-foreground"
        >
          {currentRound?.roundName ?? "Select Round"}
          <ChevronDown className="size-3" />
        </Button>
      </DropdownMenuTrigger>
      <DropdownMenuContent align="start" className="w-48">
        {rounds.map((round) => (
          <DropdownMenuItem
            key={round.id}
            onClick={() => onRoundSelect(round.id)}
            className={cn(currentRound?.id === round.id && "bg-muted")}
          >
            {round.roundName}
          </DropdownMenuItem>
        ))}
        <DropdownMenuSeparator />
        <DropdownMenuItem onClick={onNewRound}>
          <Plus className="size-4 mr-2" />
          New Round
        </DropdownMenuItem>
      </DropdownMenuContent>
    </DropdownMenu>
  )
}

export function PackageContentHeader(props: PackageContentHeaderProps) {
  if (props.variant === "commercial-summary") {
    return (
      <div className="flex items-center justify-between border-b-[0.5px] border-black/15 h-12 px-6">
        <div className="flex items-center">
          <span className="text-primary font-semibold text-16">
            Commercial summary
          </span>
        </div>
        <AIChatButton />
      </div>
    )
  }

  if (props.variant === "technical") {
    const { packageId, rounds, currentRound, onRoundSelect, onNewRound } = props

    return (
      <div className="flex items-center justify-between border-b-[0.5px] border-black/15 h-12 px-6">
        <div className="flex items-center h-full">
          <span className="text-primary font-semibold text-16 mr-2">
            Technical Evaluation
          </span>
          {rounds.length > 0 && (
            <RoundsDropdown
              rounds={rounds}
              currentRound={currentRound}
              onRoundSelect={onRoundSelect}
              onNewRound={onNewRound}
            />
          )}
          <nav className="flex items-center h-full ml-4">
            <HeaderNavLink
              to="/package/$id/tech"
              params={{ id: packageId }}
              exact
            >
              Summary
            </HeaderNavLink>
            <HeaderNavLink
              to="/package/$id/tech/ptc"
              params={{ id: packageId }}
            >
              PTC Insights
            </HeaderNavLink>
            <HeaderNavLink
              to="/package/$id/tech/docs"
              params={{ id: packageId }}
            >
              Documents
            </HeaderNavLink>
          </nav>
        </div>
        <AIChatButton />
      </div>
    )
  }

  // Asset variant
  const {
    packageId,
    assetId,
    assetName,
    rounds,
    currentRound,
    onRoundSelect,
    onNewRound,
  } = props

  return (
    <div className="flex items-center justify-between border-b-[0.5px] border-black/15 h-12 px-6">
      <div className="flex items-center h-full">
        <span className="text-primary font-semibold text-16 mr-2">
          {assetName}
        </span>
        {rounds.length > 0 && (
          <RoundsDropdown
            rounds={rounds}
            currentRound={currentRound}
            onRoundSelect={onRoundSelect}
            onNewRound={onNewRound}
          />
        )}
        <nav className="flex items-center h-full ml-4">
          <HeaderNavLink
            to="/package/$id/comm/$assetId"
            params={{ id: packageId, assetId }}
            exact
          >
            Summary
          </HeaderNavLink>
          <HeaderNavLink
            to="/package/$id/comm/$assetId/ptc"
            params={{ id: packageId, assetId }}
          >
            PTC Insights
          </HeaderNavLink>
          <HeaderNavLink
            to="/package/$id/comm/$assetId/docs"
            params={{ id: packageId, assetId }}
          >
            Vendor documents
          </HeaderNavLink>
        </nav>
      </div>
      <AIChatButton />
    </div>
  )
}
