import { useState } from "react";
import { AppShell } from "@/components/layout/AppShell";
import type { View } from "@/components/layout/TopTabs";

export default function App() {
  const [view, setView] = useState<View>("chat");

  return (
    <AppShell
      sidebar={<div className="text-sm text-muted-foreground">Sidebar content lands in later tasks.</div>}
      activeView={view}
      onViewChange={setView}
    >
      {view === "chat" ? (
        <div className="p-6 text-sm text-muted-foreground">Chat view lands in Task 6.</div>
      ) : (
        <div className="p-6 text-sm text-muted-foreground">Evaluation view lands in Task 9.</div>
      )}
    </AppShell>
  );
}
