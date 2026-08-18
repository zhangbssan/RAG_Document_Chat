import { Tabs, TabsList, TabsTrigger } from "@/components/ui/tabs";

export type View = "chat" | "evaluation";

interface TopTabsProps {
  active: View;
  onChange: (view: View) => void;
}

export function TopTabs({ active, onChange }: TopTabsProps) {
  return (
    <Tabs value={active} onValueChange={(value) => onChange(value as View)}>
      <TabsList>
        <TabsTrigger value="chat">Chat</TabsTrigger>
        <TabsTrigger value="evaluation">Evaluation</TabsTrigger>
      </TabsList>
    </Tabs>
  );
}
