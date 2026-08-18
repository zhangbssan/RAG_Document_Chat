import type { ReactNode } from "react";
import { useUiStore } from "@/store/uiStore";
import { cn } from "@/lib/utils";

interface SidebarProps {
  children: ReactNode;
}

export function Sidebar({ children }: SidebarProps) {
  const sidebarOpen = useUiStore((state) => state.sidebarOpen);
  const setSidebarOpen = useUiStore((state) => state.setSidebarOpen);

  return (
    <>
      {sidebarOpen && (
        <div
          className="fixed inset-0 z-30 bg-black/40 md:hidden"
          onClick={() => setSidebarOpen(false)}
        />
      )}
      <aside
        className={cn(
          "fixed inset-y-0 left-0 z-40 flex h-full w-80 shrink-0 flex-col gap-6 overflow-y-auto border-r border-border bg-card p-4 transition-transform md:static md:z-auto md:translate-x-0",
          sidebarOpen ? "translate-x-0" : "-translate-x-full"
        )}
      >
        {children}
      </aside>
    </>
  );
}
