function setCopyStatus(button, message) {
    button.closest(".platform-code-block, .code-block-md, .install-command")
        ?.querySelector("[data-copy-status]")
        ?.replaceChildren(message);
}

function copyWithSelection(text) {
    const textarea = document.createElement("textarea");
    textarea.value = text;
    textarea.style.cssText = "position:fixed;opacity:0";
    document.body.appendChild(textarea);
    textarea.select();
    // Throw → false so the caller renders "Copy failed"; finally never
    // leaves the scratch textarea behind.
    try {
        return document.execCommand("copy");
    } catch {
        return false;
    } finally {
        textarea.remove();
    }
}

async function copyText(text) {
    if (navigator.clipboard) {
        try {
            await navigator.clipboard.writeText(text);
            return true;
        } catch {
            return copyWithSelection(text);
        }
    }
    return copyWithSelection(text);
}

const copyResetTimers = new Map();

document.addEventListener("click", async (event) => {
    const button = event.target.closest(".copy-btn");
    const text = button?.getAttribute("data-copy");
    if (!button || !text) return;

    const copied = await copyText(text);
    if (copied) button.classList.add("copied");
    setCopyStatus(button, copied ? "Copied" : "Copy failed");
    // Reset visual + status together so the two never disagree.
    clearTimeout(copyResetTimers.get(button));
    const resetTimer = setTimeout(() => {
        button.classList.remove("copied");
        setCopyStatus(button, "");
        copyResetTimers.delete(button);
    }, 1500);
    copyResetTimers.set(button, resetTimer);
});
