import type { ReactNode } from "react";
import { Menu } from "lucide-react";
import { Sidebar } from "./Sidebar";
import { TopTabs, type View } from "./TopTabs";
import { Button } from "@/components/ui/button";
import { useUiStore } from "@/store/uiStore";

interface AppShellProps {
  sidebar: ReactNode;
  activeView: View;
  onViewChange: (view: View) => void;
  children: ReactNode;
}

export function AppShell({ sidebar, activeView, onViewChange, children }: AppShellProps) {
  const toggleSidebar = useUiStore((state) => state.toggleSidebar);

  return (
    <div className="flex h-screen w-screen overflow-hidden bg-background text-foreground">
      <Sidebar>{sidebar}</Sidebar>
      <div className="flex min-w-0 flex-1 flex-col">
        <header className="flex items-center gap-3 border-b border-border p-4">
          <Button variant="ghost" size="icon" className="md:hidden" onClick={toggleSidebar}>
            <Menu className="h-5 w-5" />
          </Button>
          <h1 className="text-lg font-semibold">RAG Document Chat</h1>
          <div className="ml-auto">
            <TopTabs active={activeView} onChange={onViewChange} />
          </div>
        </header>
        <main className="min-h-0 flex-1 overflow-hidden">{children}</main>
      </div>
    </div>
  );
}
