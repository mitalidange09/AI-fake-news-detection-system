async function checkNews() {

    const newsText = document.getElementById("newsText").value.trim();

    if (!newsText) {
        alert("Please enter news text.");
        return;
    }

    const loading = document.getElementById("loading");
    const result = document.getElementById("result");

    loading.style.display = "block";
    result.style.display = "none";

    try {

        const response = await fetch("/predict", {

            method: "POST",

            headers: {
                "Content-Type": "application/json"
            },

            body: JSON.stringify({
                news: newsText
            })

        });

        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.error);
        }

        document.getElementById("prediction").innerText =
            "Prediction: " + data.prediction;

        document.getElementById("confidence").innerText =
            data.confidence + "%";

        document.getElementById("fakeProbability").innerText =
            data.fake_probability + "%";

        document.getElementById("realProbability").innerText =
            data.real_probability + "%";

        document.getElementById("fakeBar").style.width =
            data.fake_probability + "%";

        document.getElementById("realBar").style.width =
            data.real_probability + "%";

        result.style.display = "block";

    } catch (error) {

        alert("Error: " + error.message);

    } finally {

        loading.style.display = "none";
    }
}


function clearNews() {

    document.getElementById("newsText").value = "";

    document.getElementById("result").style.display = "none";

    document.getElementById("fakeBar").style.width = "0%";

    document.getElementById("realBar").style.width = "0%";
}