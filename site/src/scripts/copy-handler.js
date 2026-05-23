document.addEventListener("click", function (e) {
    var btn = e.target.closest(".copy-btn");
    if (!btn) return;
    var text = btn.getAttribute("data-copy");
    if (!text) return;

    function flash() {
        btn.classList.add("copied");
        setTimeout(function () { btn.classList.remove("copied"); }, 1500);
    }

    if (navigator.clipboard) {
        navigator.clipboard.writeText(text).then(flash, flash);
    } else {
        var ta = document.createElement("textarea");
        ta.value = text;
        ta.style.cssText = "position:fixed;opacity:0";
        document.body.appendChild(ta);
        ta.select();
        try { document.execCommand("copy"); } catch (_) {}
        document.body.removeChild(ta);
        flash();
    }
});
