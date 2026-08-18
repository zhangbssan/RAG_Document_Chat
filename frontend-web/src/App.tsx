import { useEffect, useState } from "react";
import { AppShell } from "@/components/layout/AppShell";
import type { View } from "@/components/layout/TopTabs";
import { ConversationList } from "@/components/conversations/ConversationList";
import { DocumentPanel } from "@/components/documents/DocumentPanel";
import { ChatPage } from "@/pages/ChatPage";
import { EvaluationPage } from "@/pages/EvaluationPage";
import { useConversations } from "@/hooks/useConversations";
import { useUiStore } from "@/store/uiStore";
import { Toaster } from "@/components/ui/toast";
import { Separator } from "@/components/ui/separator";

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
    <>
      <AppShell
        sidebar={
          <div className="flex flex-col gap-4">
            <ConversationList onNewChat={() => setActiveConversationId(crypto.randomUUID())} />
            <Separator />
            <DocumentPanel />
          </div>
        }
        activeView={view}
        onViewChange={setView}
      >
        {view === "chat" ? <ChatPage /> : <EvaluationPage />}
      </AppShell>
      <Toaster />
    </>
  );
}
