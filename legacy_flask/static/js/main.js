/**
 * CinePulse - Main interactive logic for sentiment analysis
 */

document.addEventListener("DOMContentLoaded", () => {
    const reviewInput = document.getElementById("reviewInput");
    const charCounter = document.getElementById("charCounter");
    const wordCounter = document.getElementById("wordCounter");
    const clearBtn = document.getElementById("clearBtn");
    const form = document.getElementById("sentimentForm");
    const submitBtn = document.getElementById("submitBtn");
    const btnText = submitBtn.querySelector(".btn-text");
    const btnSpinner = submitBtn.querySelector(".btn-spinner");

    // Result elements
    const emptyState = document.getElementById("emptyState");
    const analysisContent = document.getElementById("analysisContent");
    const verdictBanner = document.getElementById("verdictBanner");
    const verdictEmoji = document.getElementById("verdictEmoji");
    const verdictTitle = document.getElementById("verdictTitle");
    const verdictPill = document.getElementById("verdictPill");
    const confidenceValue = document.getElementById("confidenceValue");
    const posPercent = document.getElementById("posPercent");
    const negPercent = document.getElementById("negPercent");
    const positiveBar = document.getElementById("positiveBar");
    const negativeBar = document.getElementById("negativeBar");
    const wordChips = document.getElementById("wordChips");
    const statWords = document.getElementById("statWords");
    const statReadTime = document.getElementById("statReadTime");
    const statAccuracy = document.getElementById("statAccuracy");

    const presetButtons = document.querySelectorAll(".preset-btn");

    // Helper: Update character and word counters
    function updateCounters() {
        const text = reviewInput.value.trim();
        const chars = reviewInput.value.length;
        const words = text ? text.split(/\s+/).length : 0;

        charCounter.textContent = `${chars.toLocaleString()} character${chars === 1 ? '' : 's'}`;
        wordCounter.textContent = `${words.toLocaleString()} word${words === 1 ? '' : 's'}`;
    }

    // Input listeners
    reviewInput.addEventListener("input", updateCounters);
    updateCounters();

    // Clear input
    clearBtn.addEventListener("click", () => {
        reviewInput.value = "";
        updateCounters();
        reviewInput.focus();
    });

    // Preset button handlers
    presetButtons.forEach(btn => {
        btn.addEventListener("click", () => {
            const sampleText = btn.getAttribute("data-text");
            if (sampleText) {
                reviewInput.value = sampleText;
                updateCounters();
                analyzeSentiment(sampleText);
            }
        });
    });

    // Form submit handler
    form.addEventListener("submit", (e) => {
        e.preventDefault();
        const text = reviewInput.value.trim();
        if (!text) {
            reviewInput.focus();
            return;
        }
        analyzeSentiment(text);
    });

    // Sentiment Analysis API caller
    async function analyzeSentiment(text) {
        setLoading(true);

        try {
            const response = await fetch("/predict", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json"
                },
                body: JSON.stringify({ review: text })
            });

            const data = await response.json();

            if (response.ok && data.status === "success") {
                renderResults(data.data);
            } else {
                alert(data.message || "Failed to analyze sentiment. Please try again.");
            }
        } catch (err) {
            console.error("Analysis Error:", err);
            alert("Could not connect to the sentiment analysis server. Ensure Flask is running.");
        } finally {
            setLoading(false);
        }
    }

    // Loading state switcher
    function setLoading(isLoading) {
        if (isLoading) {
            submitBtn.disabled = true;
            btnText.style.display = "none";
            btnSpinner.style.display = "inline-block";
        } else {
            submitBtn.disabled = false;
            btnText.style.display = "inline-flex";
            btnSpinner.style.display = "none";
        }
    }

    // Render results into the UI
    function renderResults(res) {
        // Toggle view
        emptyState.style.display = "none";
        analysisContent.style.display = "flex";

        const isPositive = res.sentiment === "Positive";
        const isNegative = res.sentiment === "Negative";

        // Reset classes
        verdictBanner.className = "verdict-banner";
        verdictPill.className = "badge";

        if (isPositive) {
            verdictBanner.classList.add("positive");
            verdictPill.classList.add("badge-positive");
            verdictPill.textContent = "Positive Review";
            verdictEmoji.textContent = "🍿";
            verdictTitle.textContent = "Positive Sentiment";
        } else if (isNegative) {
            verdictBanner.classList.add("negative");
            verdictPill.classList.add("badge-negative");
            verdictPill.textContent = "Negative Review";
            verdictEmoji.textContent = "💔";
            verdictTitle.textContent = "Negative Sentiment";
        } else {
            verdictPill.classList.add("badge-neutral");
            verdictPill.textContent = "Neutral / Mixed";
            verdictEmoji.textContent = "⚖️";
            verdictTitle.textContent = res.sentiment || "Neutral";
        }

        // Confidence and probabilities
        confidenceValue.textContent = `${res.confidence}%`;
        posPercent.textContent = `${res.positive_score}%`;
        negPercent.textContent = `${res.negative_score}%`;

        positiveBar.style.width = `${res.positive_score}%`;
        negativeBar.style.width = `${res.negative_score}%`;

        // Explainability - Word Chips
        wordChips.innerHTML = "";
        if (res.top_words && res.top_words.length > 0) {
            res.top_words.forEach(item => {
                const chip = document.createElement("span");
                chip.className = `chip chip-${item.sentiment}`;
                const sign = item.sentiment === "positive" ? "+" : "-";
                chip.innerHTML = `${escapeHtml(item.term)} <small>${sign}${item.weight}</small>`;
                wordChips.appendChild(chip);
            });
        } else {
            wordChips.innerHTML = '<span class="text-muted">No strongly polar keywords found in model vocabulary.</span>';
        }

        // Stats
        statWords.textContent = res.word_count || 0;
        statReadTime.textContent = `${res.reading_time_sec || 1}s`;
        statAccuracy.textContent = `${res.accuracy || 100}%`;
    }

    function escapeHtml(string) {
        const div = document.createElement('div');
        div.innerText = string;
        return div.innerHTML;
    }
});
