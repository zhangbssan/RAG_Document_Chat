import { useState, type ReactElement } from "react";
import {
  Dialog,
  DialogContent,
  DialogFooter,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";

interface RenameDialogProps {
  currentTitle: string;
  onRename: (title: string) => void;
  trigger: ReactElement;
}

export function RenameDialog({ currentTitle, onRename, trigger }: RenameDialogProps) {
  const [title, setTitle] = useState(currentTitle);
  const [open, setOpen] = useState(false);

  return (
    <Dialog open={open} onOpenChange={setOpen}>
      <DialogTrigger render={trigger} />
      <DialogContent>
        <DialogHeader>
          <DialogTitle>Rename conversation</DialogTitle>
        </DialogHeader>
        <Input value={title} onChange={(event) => setTitle(event.target.value)} maxLength={200} />
        <DialogFooter>
          <Button
            onClick={() => {
              if (title.trim()) {
                onRename(title.trim());
                setOpen(false);
              }
            }}
          >
            Save title
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
