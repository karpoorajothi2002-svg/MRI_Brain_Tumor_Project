document.addEventListener("DOMContentLoaded", function () {
    const languageToggle = document.getElementById("languageToggle");

    if (!languageToggle) return;

    let currentLanguage = localStorage.getItem("language") || "en";

    function changeLanguage(language) {
        document.querySelectorAll("[data-en][data-ta]").forEach(function (element) {
            element.textContent = element.getAttribute("data-" + language);
        });

        languageToggle.textContent = language === "en" ? "தமிழ்" : "English";
        document.documentElement.lang = language === "en" ? "en" : "ta";

        localStorage.setItem("language", language);
        currentLanguage = language;
    }

    changeLanguage(currentLanguage);

    languageToggle.addEventListener("click", function () {
        changeLanguage(currentLanguage === "en" ? "ta" : "en");
    });
});