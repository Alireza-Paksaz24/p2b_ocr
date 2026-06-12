import * as DialogPrimitive from '@radix-ui/react-dialog';
import React from 'react';
import { X } from 'lucide-react'; // optional; we'll use plain X text

export function Dialog({
  open,
  onOpenChange,
  title,
  description,
  children,
  trigger,
}: {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  title: string;
  description?: string;
  children: React.ReactNode;
  trigger?: React.ReactNode;
}) {
  return (
    <DialogPrimitive.Root open={open} onOpenChange={onOpenChange}>
      {trigger && <DialogPrimitive.Trigger asChild>{trigger}</DialogPrimitive.Trigger>}
      <DialogPrimitive.Portal>
        <DialogPrimitive.Overlay className="fixed inset-0 bg-black/50 data-[state=open]:animate-overlayShow" />
        <DialogPrimitive.Content className="fixed left-1/2 top-1/2 max-h-[85vh] w-[90vw] max-w-md -translate-x-1/2 -translate-y-1/2 rounded-lg bg-white p-6 shadow-lg focus:outline-none">
          <DialogPrimitive.Title className="text-lg font-medium">{title}</DialogPrimitive.Title>
          {description && (
            <DialogPrimitive.Description className="mt-2 text-sm text-gray-500">
              {description}
            </DialogPrimitive.Description>
          )}
          <div className="mt-4">{children}</div>
          <DialogPrimitive.Close className="absolute top-4 right-4 text-gray-400 hover:text-gray-600">
            ✕
          </DialogPrimitive.Close>
        </DialogPrimitive.Content>
      </DialogPrimitive.Portal>
    </DialogPrimitive.Root>
  );
}