import { Moon, Sun, Monitor } from "lucide-react";
import { Button } from "@/components/ui/button";
import { useTheme } from "@/hooks/useTheme";
import { cn } from "@/lib/utils";
import type { Theme } from "@/store/uiStore";

const OPTIONS: { value: Theme; icon: typeof Sun; label: string }[] = [
  { value: "light", icon: Sun, label: "Light" },
  { value: "dark", icon: Moon, label: "Dark" },
  { value: "system", icon: Monitor, label: "System" },
];

export function ThemeToggle() {
  const { theme, setTheme } = useTheme();

  return (
    <div className="flex gap-1 rounded-md border border-border p-1">
      {OPTIONS.map(({ value, icon: Icon, label }) => (
        <Button
          key={value}
          type="button"
          variant="ghost"
          size="sm"
          className={cn("h-7 flex-1 gap-1 px-2", theme === value && "bg-muted")}
          onClick={() => setTheme(value)}
          title={label}
        >
          <Icon className="h-3.5 w-3.5" />
        </Button>
      ))}
    </div>
  );
}
