// Synchronize Slider and Input
function syncInputAndSlider(changedElementId, syncedElementId) {
    const changedElement = document.getElementById(changedElementId);
    const syncedElement = document.getElementById(syncedElementId);
    if (syncedElement) {
        syncedElement.value = changedElement.value;
    }
}

// Fetch sliders from the server
fetch('/sliders')
    .then(response => response.json())
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

        document.getElementById('search-button').addEventListener('click', () => {
            const sliderValues = sliders.reduce((acc, sliderData) => {
                acc[sliderData.id] = parseFloat(document.getElementById(`${sliderData.id}slider`).value);
                return acc;
            }, {});

            fetch('/submit', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(sliderValues),
            })
            .then(response => response.json())
            .then(data => console.log('Server response:', data))
            .catch(error => console.error('Error sending slider values:', error));
        });
    })
    .catch(error => console.error('Error fetching slider data:', error));

// Fetch sliders from the server
fetch('/recom')
    .then(response => response.json())
    .then(recom => {
        const container = document.getElementById('recom-container');
        container.innerHTML = ''; // Clear existing content

        recom.forEach(recomDate => {
            const recomHTML = `
                <div class="recom-item">
                    <label for="${recomDate.id}Recom">${recomDate.id}</label><br>
                    <button type="button" onclick="handleRecomClick('${recomDate.i}', 'max')">max: ${recomDate.max_val}</button>
                    <button type="button" onclick="handleRecomClick('${recomDate.i}', 'min')">min: ${recomDate.min_val}</button>
                </div>
            `;

            container.insertAdjacentHTML('beforeend', recomHTML);
        });
    })
    .catch(error => console.error('Error fetching recom data:', error));

// Define the function to handle button clicks
function handleRecomClick(id, value) {
    fetch('/handle_recom', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ buttonId: id, val: value }),
    })
    .then(response => response.json())
    .then(data => {('Success:', data);})
    .catch((error) => {console.error('Error:', error);});
}

// Fetch and render graph data
const svg = d3.select("#graph-area");

fetch('/graph-data')
    .then(response => response.json())
    .then(nodes => {
        if (nodes.length > 1) {
            svg.selectAll("line")
                .data(nodes.slice(1))
                .enter()
                .append("line")
                .attr("class", "link")
                .attr("x1", d => nodes[0].x)
                .attr("y1", d => nodes[0].y)
                .attr("x2", d => d.x)
                .attr("y2", d => d.y);
        }

        const nodeGroups = svg.selectAll("g.node-group")
            .data(nodes)
            .enter()
            .append("g")
            .attr("class", "node-group")
            .attr("transform", d => `translate(${d.x},${d.y})`);

        nodeGroups.append("circle")
            .attr("class", "node")
            .attr("r", 20)
            .attr("fill", d => d.color);

        nodeGroups.append("text")
            .attr("class", "node-text")
            .attr("dx", 0)
            .attr("dy", ".35em")
            .text(d => d.id);
    })
    .catch(error => console.error('Error fetching graph data:', error));

// Fetch and render tree data
fetch('/tree-data')
    .then(response => response.json())
    .then(treeData => {
        const treeSvg = d3.select("#tree-svg"),
            width = treeSvg.node().clientWidth,
            height = treeSvg.node().clientHeight;

        const treeLayout = d3.tree().size([height - 160, width - 160]);

        const root = d3.hierarchy(treeData);
        treeLayout(root);

        const g = treeSvg.append("g")
            .attr("transform", `translate(${width/4} ,10)`);

        g.selectAll(".link")
            .data(root.links())
            .enter()
            .append("line")
            .attr("class", "link")
            .attr("x1", d => d.source.x)
            .attr("y1", d => d.source.y)
            .attr("x2", d => d.target.x)
            .attr("y2", d => d.target.y)
            .attr("stroke", "#ccc")
            .attr("stroke-width", 2);

        const node = g.selectAll(".node")
            .data(root.descendants())
            .enter()
            .append("g")
            .attr("class", "node")
            .attr("transform", d => `translate(${d.x},${d.y})`);

        node.append("text")
            .attr("dx", 0)
            .attr("dy", ".35em")
            .attr("class", "node-text")
            .attr("fill", "black")
            .text(d => d.data.name);
    })
    .catch(error => console.error('Error fetching tree data:', error));
