import { create } from "zustand"

// Types for uploaded files (matching upload-zone component)
interface UploadedFile {
  id: string
  name: string
  key: string // S3 object key
}

interface AssetFiles {
  boqFile: UploadedFile[]
  pteFile: UploadedFile[]
  vendorFiles: Record<string, UploadedFile[]>
}

const useStore = create<{
  navbarOpen: boolean
  setNavbarOpen: (navbarOpen: boolean) => void
  // Technical evaluation rounds (keyed by packageId)
  selectedTechRound: Record<string, string>
  setTechRound: (packageId: string, roundId: string) => void
  // Commercial evaluation rounds (keyed by assetId)
  selectedCommRound: Record<string, string>
  setCommRound: (assetId: string, roundId: string) => void
  // Create asset sheet state
  createAssetSheetOpen: boolean
  setCreateAssetSheetOpen: (open: boolean) => void
  // Asset files (keyed by assetId) - persists files from asset creation
  assetFiles: Record<string, AssetFiles>
  setAssetFiles: (assetId: string, files: AssetFiles) => void
  getAssetFiles: (assetId: string) => AssetFiles | undefined
}>((set, get) => ({
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
  // Create asset sheet
  createAssetSheetOpen: false,
  setCreateAssetSheetOpen: (open) => set({ createAssetSheetOpen: open }),
  // Asset files
  assetFiles: {},
  setAssetFiles: (assetId, files) =>
    set((state) => ({
      assetFiles: { ...state.assetFiles, [assetId]: files },
    })),
  getAssetFiles: (assetId) => get().assetFiles[assetId],
}))

export default useStore
