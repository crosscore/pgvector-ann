/* pgvector-ann/frontend/static/script.js */

document.addEventListener("DOMContentLoaded", () => {
	const searchInput = document.getElementById("search-input");
	const topNInput = document.getElementById("top-n-input");
	const searchButton = document.getElementById("search-button");
	const searchMetrics = document.getElementById("search-metrics");
	const searchTime = document.getElementById("search-time");
	const searchResults = document.getElementById("search-results");

	let socket = new WebSocket("ws://" + window.location.host + "/ws");

	socket.onopen = function (e) {
		console.log("[open] Connection established");
	};

	socket.onmessage = function (event) {
		const data = JSON.parse(event.data);
		if (data.error) {
			searchResults.innerHTML = `<p>Error: ${data.error}</p>`;
		} else {
			displayResults(data);
		}
	};

	socket.onerror = function (error) {
		console.log(`[error] ${error.message}`);
	};

	searchButton.addEventListener("click", () => {
		const query = searchInput.value;
		const topN = parseInt(topNInput.value);
		if (query && topN) {
			socket.send(JSON.stringify({ question: query, top_n: topN }));
			searchResults.innerHTML = "<p>Searching...</p>";
		}
	});

	function displayResults(data) {
		searchMetrics.style.display = "block";
		searchTime.textContent = data.search_time.toFixed(4);

		let resultsHTML = "<h2>Search Results</h2>";
		data.results.forEach((result, index) => {
			resultsHTML += `
                <div class="result">
                    <h3>${index + 1}. <a href="${
				result.link
			}" target="_blank">${result.link_text}</a></h3>
                    <p>${result.chunk_text}</p>
                    <p>Distance: ${result.distance.toFixed(4)}</p>
                </div>
            `;
		});

		searchResults.innerHTML = resultsHTML;
	}
});
