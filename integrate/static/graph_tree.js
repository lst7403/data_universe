let graph_tree_base_node_r = 30;

// Sample data
let got_data = [
    {'id': 74, 'cos_sim': -0.3249267290917288, 'euclide_dist': 10, 'r_ratio': 1.5},
    {'id': 111, 'cos_sim': 0.5, 'euclide_dist': 6, 'r_ratio': 1.5},
    {'id': 28, 'cos_sim': 0.7107, 'euclide_dist': 14, 'r_ratio': 2},
    {'id': 18, 'cos_sim': -1, 'euclide_dist': 2, 'r_ratio': 2},
    {'id': 37, 'cos_sim': 1, 'euclide_dist': 0, 'r_ratio': 1.0},
];

function rotateCircle(selection, d_angle, d_dist) {
    // Calculate initial distance from center
    let initialX = +selection.attr("cx"); // Get initial x position
    let initialY = +selection.attr("cy"); // Get initial y position
    let deltaX = initialX - svg_center_x; // Difference in X
    let deltaY = initialY - svg_center_y; // Difference in Y

    let init_dist = Math.sqrt((deltaX) ** 2 + (deltaY) ** 2); // Calculate initial distance from center
    let init_angle = Math.atan2(deltaY, deltaX) * (180 / Math.PI); // Convert to degrees

    selection.transition()
        .duration(2000) // Custom duration for each circle
        .attrTween("transform", function() {
            return function(t) {
                // Calculate the current angle based on time t
                let cur_angle = init_angle + t * d_angle;
                let cur_dist = init_dist + t * d_dist;

                // Convert degrees to radians
                let radians = (cur_angle * Math.PI) / 180;

                // Calculate new position based on the angle and distance
                let x = cur_dist * Math.cos(radians);
                let y = cur_dist * Math.sin(radians);

                // Return the translation with the SVG center offset
                return `translate(${x - deltaX}, ${y - deltaY})`;
            };
        });
}

// Function to calculate x, y, and angle
function cal_x_y_angle(cos_sim, euclide_dist, scaler) {
    let normal_radian = Math.acos(cos_sim);
    let angle = normal_radian * (180 / Math.PI);
    let special_radian = normal_radian + (cos_sim >= 0 ? -Math.PI / 4 : Math.PI / 4);
    let scaled_dist = scaler(euclide_dist);

    return [scaled_dist * Math.cos(special_radian), scaled_dist * Math.sin(special_radian), angle];
}

function server_data_to_graph_data(server_data) {

    let euclide_scaler = d3.scaleLinear()
    .domain(d3.extent(server_data.map(d => d.euclide_dist)))
    .range([graph_tree_axis_radius + graph_tree_base_node_r * 2 + short_tick_len * 2, Math.min(graph_tree_width / 2, graph_tree_height / 2 / Math.cos(Math.PI / 4)) - graph_tree_base_node_r * 2]);

    let translated_graph_tree_data = [];
    // push center circle
    translated_graph_tree_data.push({
        id: server_data[server_data.length-1].id,
        x: graph_tree_width / 2,
        y: graph_tree_height / 2,
        r: graph_tree_base_node_r * server_data[server_data.length-1].r_ratio,
        angle: 0,
        dist: 0,
    });

    // push other circles 
    for (let i = 0; i < server_data.length-1; i++) {
        let tmp_cal_x_y_angle_res = cal_x_y_angle(server_data[i].cos_sim, server_data[i].euclide_dist, euclide_scaler);

        translated_graph_tree_data.push({
            id: server_data[i].id,
            x: tmp_cal_x_y_angle_res[0] + graph_tree_width / 2,
            y: tmp_cal_x_y_angle_res[1] + graph_tree_height / 2,
            r: graph_tree_base_node_r * server_data[i].r_ratio,
            angle: tmp_cal_x_y_angle_res[2],
            dist: euclide_scaler(server_data[i].euclide_dist), 
        });
    }

    return translated_graph_tree_data
}
    
let new_data = server_data_to_graph_data(got_data);

let cur_links = [];

let lines = graph_tree_svg.selectAll("line.new-line")
    .data(new_data)
    .enter()
    .append("line")
        .attr("class", "new-line")
        .attr("x1", new_data[0].x)
        .attr("y1", new_data[0].y)
        .attr("x2", d => d.x)
        .attr("y2", d => d.y)
        .attr("stroke", "grey")

// append circles and text in group
let circle_groups = graph_tree_svg.selectAll("g.cluster")
    .data(new_data)
    .enter()
    .append("g")
        .attr("class", "cluster")
        .attr("id", d => d.id) // Initial position
        .attr("transform", d => `translate(${d.x}, ${d.y})`) // Initial position
        .call(g => {
            g.append("circle")
                .attr("r", d => d.r)
                .attr("fill", d => `hsla(${d.angle}, 100%, 50%, 0.6)`);

            g.append("text")
                .text(d => d.id)
                .attr("text-anchor", "middle")
                .attr("dominant-baseline", "central")
        });

        
// function updateCircles(new_data) {
//     let circles = graph_tree_svg.selectAll("circle")
//         .data(new_data, d => d.id);

//     let texts = graph_tree_svg.selectAll("text")
//         .data(new_data, d => d.id);

//     // Remove circles that are not in the new data with a fade-out transition
//     circles.exit()
//         .transition()
//         .duration(1000)
//         .style("opacity", 0)
//         .remove();

//     texts.exit()
//         .transition()
//         .duration(1000)
//         .style("opacity", 0)
//         .remove();

//     // Update existing circles (matching ids) with transitions
//     circles
//         .transition()
//         .duration(1000)
//         .attr("cx", d => d.x)
//         .attr("cy", d => d.y)
//         .attr("r", d => d.r);

//     texts
//         .transition()
//         .duration(1000)
//         .attr("x", d => d.x)
//         .attr("y", d => d.y);

//     // Append new circles with a fade-in effect
//     let newCircles = circles.enter().append("circle")
//         .attr("cx", d => d.x)
//         .attr("cy", d => d.y)
//         .attr("r", d => d.r)
//         .style("opacity", 0)  // Start invisible
//         .attr("fill", "blue")
//         .transition()
//         .duration(1000)
//         .style("opacity", 1);  // Fade in

//     // Append new text elements with fade-in effect
//     let newTexts = texts.enter().append("text")
//         .attr("x", d => d.x)
//         .attr("y", d => d.y)
//         .text(d => d.id)
//         .style("opacity", 0)  // Start invisible
//         .transition()
//         .duration(1000)
//         .style("opacity", 1);  // Fade in
// }

// updateCircles(new_data)