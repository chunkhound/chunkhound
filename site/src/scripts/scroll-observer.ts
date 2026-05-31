export function setupScrollAnimation(selector: string, threshold = 0.25): void {
  const section = document.querySelector(selector);
  if (!section) return;
  section.classList.add("js-animate");
  const observer = new IntersectionObserver(
    ([entry]) => {
      if (entry.isIntersecting) {
        void (section as HTMLElement).offsetHeight;
        section.classList.add("is-visible");
      } else {
        section.classList.remove("is-visible");
      }
    },
    { threshold }
  );
  observer.observe(section);
}
