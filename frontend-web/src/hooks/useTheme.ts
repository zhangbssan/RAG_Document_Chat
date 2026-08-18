import { useEffect, useState } from "react";
import { useUiStore, type Theme } from "@/store/uiStore";

function systemPrefersDark(): boolean {
  return window.matchMedia("(prefers-color-scheme: dark)").matches;
}

export function useTheme() {
  const theme = useUiStore((state) => state.theme);
  const setTheme = useUiStore((state) => state.setTheme);
  const [resolvedTheme, setResolvedTheme] = useState<"light" | "dark">(
    theme === "system" ? (systemPrefersDark() ? "dark" : "light") : theme
  );

  useEffect(() => {
    const resolve = () => (theme === "system" ? (systemPrefersDark() ? "dark" : "light") : theme);
    setResolvedTheme(resolve());

    if (theme !== "system") return;
    const media = window.matchMedia("(prefers-color-scheme: dark)");
    const listener = () => setResolvedTheme(resolve());
    media.addEventListener("change", listener);
    return () => media.removeEventListener("change", listener);
  }, [theme]);

  useEffect(() => {
    document.documentElement.classList.toggle("dark", resolvedTheme === "dark");
  }, [resolvedTheme]);

  return { theme, setTheme, resolvedTheme } as {
    theme: Theme;
    setTheme: (t: Theme) => void;
    resolvedTheme: "light" | "dark";
  };
}
