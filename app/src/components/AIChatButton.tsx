import { Button } from "@/components/ui/button"
import useStore from "@/lib/store"

export function AIChatButton() {
  const { chatOpen, setChatOpen } = useStore()

  if (chatOpen) return null

  return (
    <Button variant="ghost" size="sm" onClick={() => setChatOpen(true)}>
      <span className="text-gradient">AI Chat</span>
    </Button>
  )
}
