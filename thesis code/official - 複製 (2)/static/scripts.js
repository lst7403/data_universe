const socket = io();

    // Synchronize Slider and Input
    function syncInputAndSlider(changedElementId, syncedElementId) {
        const changedElement = document.getElementById(changedElementId);
        const syncedElement = document.getElementById(syncedElementId);
        if (syncedElement) {
            syncedElement.value = changedElement.value;
        }
    }

    socket.on('sliders', sliders => {
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

            socket.emit('submit', sliderValues);
        });
    });

    // Handle server response after submitting slider values
    socket.on('server_response', data => {
        console.log('Server response:', data);
    });

    // Fetch and render graph data
    socket.on('graph_data', nodes => {
        const svg = d3.select("#graph-area");
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
    });

    // Fetch and render tree data
    socket.on('tree_data', treeData => {
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
    });

    // Request data from the server when the page loads
    socket.emit('request_data');