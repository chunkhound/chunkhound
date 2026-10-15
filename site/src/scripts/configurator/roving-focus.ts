// Roving arrow-key focus shared by platform pills and stage tabs.
export const ROVING_KEYS: readonly string[] = ["ArrowLeft", "ArrowRight", "Home", "End"];

export function bindRovingKeys<T extends HTMLElement>(
  element: T,
  items: () => readonly T[],
  activate: (item: T) => void,
): void {
  element.addEventListener("keydown", (event: KeyboardEvent) => {
    if (!ROVING_KEYS.includes(event.key)) return;
    const buttons = items();
    const current = buttons.indexOf(element);
    if (current < 0 || buttons.length === 0) return;
    event.preventDefault();
    activate(buttons[rovingNextIndex(buttons, current, event.key)]!);
  });
}

export function rovingNextIndex(
  buttons: readonly HTMLElement[],
  current: number,
  key: string,
): number {
  if (key === "ArrowLeft") return (current - 1 + buttons.length) % buttons.length;
  if (key === "ArrowRight") return (current + 1) % buttons.length;
  if (key === "Home") return 0;
  if (key === "End") return buttons.length - 1;
  return current;
}
