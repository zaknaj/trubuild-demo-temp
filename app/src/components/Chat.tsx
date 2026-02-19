import useStore from "@/lib/store"
import { cn } from "@/lib/utils"
import { Button } from "./ui/button"
import { ScrollArea } from "./ui/scroll-area"
import { Send, Sparkles, X } from "lucide-react"

function SkeletonMessage({ align }: { align: "left" | "right" }) {
  return (
    <div className={cn("flex flex-col gap-1.5", align === "right" ? "items-end" : "items-start")}>
      <div className="h-2 w-12 bg-muted rounded" />
      <div
        className={cn(
          "rounded-lg p-3 space-y-2",
          align === "right" ? "bg-primary/10" : "bg-muted"
        )}
      >
        <div className="h-2 w-48 bg-black/5 rounded" />
        <div className="h-2 w-36 bg-black/5 rounded" />
        {align === "left" && <div className="h-2 w-24 bg-black/5 rounded" />}
      </div>
    </div>
  )
}

export const Chat = () => {
  const { chatOpen, setChatOpen } = useStore()
  return (
    <div
      className={cn(
        "transition-all overflow-hidden border-l bg-background shrink-0",
        chatOpen ? "w-[360px]" : "w-0"
      )}
    >
      <div className="w-[360px] h-full flex flex-col">
        {/* Header */}
        <div className="flex items-center justify-between px-4 py-3 border-b shrink-0">
          <div className="flex items-center gap-2">
            <Sparkles className="size-4 text-purple-500" />
            <span className="font-medium text-sm bg-gradient-to-r from-blue-500 via-purple-500 via-pink-500 to-orange-500 bg-clip-text text-transparent">
              AI Chat
            </span>
          </div>
          <Button
            variant="ghost"
            size="icon"
            className="size-7"
            onClick={() => setChatOpen(false)}
          >
            <X className="size-4" />
          </Button>
        </div>

        {/* Messages */}
        <ScrollArea className="flex-1 p-4">
          <div className="flex flex-col gap-4">
            <SkeletonMessage align="left" />
            <SkeletonMessage align="right" />
            <SkeletonMessage align="left" />
          </div>
        </ScrollArea>

        {/* Input */}
        <div className="p-3 border-t shrink-0">
          <div className="flex items-center gap-2 bg-muted rounded-lg px-3 py-2">
            <input
              type="text"
              placeholder="Ask anything..."
              className="flex-1 bg-transparent text-sm outline-none placeholder:text-muted-foreground"
            />
            <Button size="icon" className="size-7 shrink-0">
              <Send className="size-3.5" />
            </Button>
          </div>
          <p className="text-[10px] text-muted-foreground text-center mt-2">
            AI responses may not always be accurate
          </p>
        </div>
      </div>
    </div>
  )
}
