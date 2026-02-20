import { create } from "zustand"

const useStore = create<{
  navbarOpen: boolean
  setNavbarOpen: (navbarOpen: boolean) => void
  // Technical evaluation rounds (keyed by packageId)
  selectedTechRound: Record<string, string>
  setTechRound: (packageId: string, roundId: string) => void
  // Commercial evaluation rounds (keyed by assetId)
  selectedCommRound: Record<string, string>
  setCommRound: (assetId: string, roundId: string) => void
}>((set) => ({
  navbarOpen: true,
  setNavbarOpen: (navbarOpen) => set({ navbarOpen }),
  // Technical rounds
  selectedTechRound: {},
  setTechRound: (packageId, roundId) =>
    set((state) => ({
      selectedTechRound: { ...state.selectedTechRound, [packageId]: roundId },
    })),
  // Commercial rounds
  selectedCommRound: {},
  setCommRound: (assetId, roundId) =>
    set((state) => ({
      selectedCommRound: { ...state.selectedCommRound, [assetId]: roundId },
    })),
}))

export default useStore
