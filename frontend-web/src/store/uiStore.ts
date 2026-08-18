import { create } from "zustand";

export type Theme = "light" | "dark" | "system";

interface UiState {
  activeConversationId: string | null;
  sidebarOpen: boolean;
  theme: Theme;
  apiKey: string;
  setActiveConversationId: (id: string | null) => void;
  setSidebarOpen: (open: boolean) => void;
  toggleSidebar: () => void;
  setTheme: (theme: Theme) => void;
  setApiKey: (key: string) => void;
}

const THEME_STORAGE_KEY = "rag-chat-theme";
const API_KEY_SESSION_KEY = "rag-chat-api-key";

function readInitialTheme(): Theme {
  const stored = window.localStorage.getItem(THEME_STORAGE_KEY);
  return stored === "light" || stored === "dark" || stored === "system" ? stored : "system";
}

function readInitialApiKey(): string {
  return window.sessionStorage.getItem(API_KEY_SESSION_KEY) || "";
}

function readInitialSidebarOpen(): boolean {
  return window.matchMedia("(min-width: 768px)").matches;
}

export const useUiStore = create<UiState>((set) => ({
  activeConversationId: null,
  sidebarOpen: readInitialSidebarOpen(),
  theme: readInitialTheme(),
  apiKey: readInitialApiKey(),
  setActiveConversationId: (id) => set({ activeConversationId: id }),
  setSidebarOpen: (open) => set({ sidebarOpen: open }),
  toggleSidebar: () => set((state) => ({ sidebarOpen: !state.sidebarOpen })),
  setTheme: (theme) => {
    window.localStorage.setItem(THEME_STORAGE_KEY, theme);
    set({ theme });
  },
  setApiKey: (key) => {
    if (key) {
      window.sessionStorage.setItem(API_KEY_SESSION_KEY, key);
    } else {
      window.sessionStorage.removeItem(API_KEY_SESSION_KEY);
    }
    set({ apiKey: key });
  },
}));
