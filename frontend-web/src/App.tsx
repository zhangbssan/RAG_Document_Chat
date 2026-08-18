import { useEffect, useState } from "react";
import { AppShell } from "@/components/layout/AppShell";
import type { View } from "@/components/layout/TopTabs";
import { ConversationList } from "@/components/conversations/ConversationList";
import { useConversations } from "@/hooks/useConversations";
import { useUiStore } from "@/store/uiStore";

export default function App() {
  const [view, setView] = useState<View>("chat");
  const { data: conversations } = useConversations();
  const activeConversationId = useUiStore((state) => state.activeConversationId);
  const setActiveConversationId = useUiStore((state) => state.setActiveConversationId);

  useEffect(() => {
    if (activeConversationId || !conversations) return;
    setActiveConversationId(conversations[0]?.id ?? crypto.randomUUID());
  }, [conversations, activeConversationId, setActiveConversationId]);

  return (
    <AppShell
      sidebar={
        <ConversationList onNewChat={() => setActiveConversationId(crypto.randomUUID())} />
      }
      activeView={view}
      onViewChange={setView}
    >
      {view === "chat" ? (
        <div className="p-6 text-sm text-muted-foreground">
          Selected conversation: {activeConversationId ?? "none"}. Message list lands in Task 6.
        </div>
      ) : (
        <div className="p-6 text-sm text-muted-foreground">Evaluation view lands in Task 9.</div>
      )}
    </AppShell>
  );
}
