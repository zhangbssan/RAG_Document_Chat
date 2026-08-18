import { Input } from "@/components/ui/input";
import { ThemeToggle } from "./ThemeToggle";
import { useUiStore } from "@/store/uiStore";
import { getApiBaseUrl } from "@/api/client";

export function SettingsPanel() {
  const apiKey = useUiStore((state) => state.apiKey);
  const setApiKey = useUiStore((state) => state.setApiKey);

  return (
    <div className="flex flex-col gap-3">
      <h2 className="text-sm font-semibold">⚙️ Settings</h2>
      <p className="text-xs text-muted-foreground">
        Backend: <code>{getApiBaseUrl()}</code>
      </p>
      <div className="flex flex-col gap-1">
        <label className="text-xs font-medium">OpenAI API Key (optional)</label>
        <Input
          type="password"
          value={apiKey}
          onChange={(event) => setApiKey(event.target.value.trim())}
          placeholder="sk-…"
        />
        <p className="text-xs text-muted-foreground">
          Used only for this session. If empty, the app falls back to the backend's configured key or
          extractive answers.
        </p>
      </div>
      <div className="flex flex-col gap-1">
        <label className="text-xs font-medium">Theme</label>
        <ThemeToggle />
      </div>
    </div>
  );
}
