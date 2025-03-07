// Synchronize Slider and Input
function syncInputAndSlider(changedElementId, syncedElementId) {
    const changedElement = document.getElementById(changedElementId);
    const syncedElement = document.getElementById(syncedElementId);
    if (syncedElement) {
        syncedElement.value = changedElement.value;
    }
}

function postData(url = "", data = {}, postID = "") {
    return fetch(url, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(data)
    })
    .then(response => response.json())
    .then(jsonData => {
        console.log(`${postID} Post Success:`, jsonData);
        return jsonData;  // Return the JSON data
    })
    .catch(error => {
        console.error(`${postID} Post Error:`, error);
        throw error;  // Re-throw the error so it can be handled by the caller
    });
}

function getData(url = "", getID = "") {
    return fetch(url)
    .then(response => response.json())
    .then(jsonData => {
        console.log(`${getID} Get Success:`, jsonData);
        return jsonData;  // Return the JSON data
    })
    .catch(error => {
        console.error(`${getID} Get Error:`, error);
        throw error;  // Re-throw the error so it can be handled by the caller
    });
}

// get init slider and render
getData(url = '/sliders', getID = "get init slider")
    .then(sliders => {
        const container = document.getElementById('slider-container');
        container.innerHTML = ''; // Clear existing content

        sliders.forEach(sliderData => {
            const sliderId = `${sliderData.id}slider`;
            const inputId = `${sliderData.id}Input`;
            const step = sliderData.type === "i" ? "1" : "0.01";

            const sliderHTML = `
                <div class="slider-item">
                    <div class="slider-controls">
                        <label class="slider-label" for="${sliderId}">${sliderData.id}:</label>
                        <input type="number" id="${inputId}" class="value-input" min="${sliderData.min}" max="${sliderData.max}" value="${sliderData.value}" step="${step}" oninput="syncInputAndSlider('${inputId}', '${sliderId}')">
                    </div>
                    <input type="range" id="${sliderId}" class="slider" min="${sliderData.min}" max="${sliderData.max}" value="${sliderData.value}" step="${step}" oninput="syncInputAndSlider('${sliderId}', '${inputId}')">
                </div>
            `;

            container.insertAdjacentHTML('beforeend', sliderHTML);
        });
    });
    
// get init recom and render
getData(url = "recom", getID = "get init recom")
    .then(recom => {
        const container = document.getElementById('recom-container');
        container.innerHTML = ''; // Clear existing content

        recom.forEach(recomDate => {
            const recomHTML = `
                <div class="recom-item">
                    <label for="${recomDate.id}Recom">${recomDate.id}</label><br>
                    <button type="button" onclick="handleRecomClick('${recomDate.i}', 1)">max: ${recomDate.max_val}</button>
                    <button type="button" onclick="handleRecomClick('${recomDate.i}', 0)">min: ${recomDate.min_val}</button>
                </div>
            `;

            container.insertAdjacentHTML('beforeend', recomHTML);
        });
    });
    
// post slider val
function sendSliderValues() {
    document.querySelector('.visual-section').scrollIntoView({ behavior: 'smooth' });

    // Get all input elements of type range (sliders) within the container
    const sliders = document.getElementById('slider-container').querySelectorAll('input[type="range"]');

    // Use reduce to create a dictionary of slider IDs and their values
    const sliderData = Array.from(sliders).reduce((acc, slider) => {
        acc[slider.id] = slider.value;
        return acc;
    }, {});

    postData(url = "/handle_slider", data = sliderData, postID = "send slider val")
        .then(data => {render_graph_tree(data.graph_tree_data, 0);});
};

// func of recom button clicked
function handleRecomClick(id, value) {
    document.querySelector('.visual-section').scrollIntoView({ behavior: 'smooth' });
    postData('/handle_recom', { "buttonId": id, "val": value }, postID = "send recom click")
        .then(data => {render_graph_tree(data.graph_tree_data, 0);});
};

// Fetch and render graph_tree data
getData(url = '/graph_tree_data', getID = "get_graph_tree")
    .then(data => {render_graph_tree(data.graph_tree_data, 0);});