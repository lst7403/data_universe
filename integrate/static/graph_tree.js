const graph_tree_base_node_r = 25;

// Sample data
let got_data = [
    {'id': 74, 'cos_sim': -0.3249267290917288, 'euclide_dist': 0.5, 'r_ratio': 1.5384615384615385},
    {'id': 111, 'cos_sim': -0.11059286214225257, 'euclide_dist': 0.42832980162039197, 'r_ratio': 1.9},
    {'id': 28, 'cos_sim': 0.5, 'euclide_dist': 0.8, 'r_ratio': 1.5384615384615385},
    {'id': 37, 'cos_sim': 0.7107, 'euclide_dist': 0.42832980162039197, 'r_ratio': 1.0},
    {'id': 1, 'cos_sim': 1, 'euclide_dist': 0.7, 'r_ratio': 1.5384615384615385},
    {'id': 2, 'cos_sim': -1, 'euclide_dist': 0.7, 'r_ratio': 1.5384615384615385},
    {'id': 68, 'cos_sim': -0.9, 'euclide_dist': 0.6, 'r_ratio': 2},
    {'id': 79, 'cos_sim': 0, 'euclide_dist': 0.6, 'r_ratio': 0.6},
    {'id': 110, 'cos_sim': 1.0, 'euclide_dist': 0.0, 'r_ratio': 1}
];

// Create a scale for the euclidean distance
let euclide_scaler = d3.scaleLinear()
    .domain(d3.extent(got_data.map(d => d.euclide_dist)))
    .range([graph_tree_axis_radius - graph_tree_base_node_r * 2, Math.min(graph_tree_width / 2, graph_tree_height / 2 / Math.cos(Math.PI / 4)) - graph_tree_base_node_r * 2]);
    // .range([graph_tree_axis_radius - graph_tree_base_node_r * 2, 60]);

// Function to calculate x, y, and angle
function cal_x_y_angle(cos_sim, euclide_dist) {
    let normal_radian = Math.acos(cos_sim);
    let angle = normal_radian * (180 / Math.PI);
    let special_radian = normal_radian + (cos_sim >= 0 ? -Math.PI / 4 : Math.PI / 4);
    let scaled_dist = euclide_scaler(euclide_dist);

    return [scaled_dist * Math.cos(special_radian), scaled_dist * Math.sin(special_radian), angle];
}

// Create new data array
let new_data = [];
for (let i = 0; i < got_data.length; i++) {
    let cos_sim = got_data[i].cos_sim;
    let euclide_dist = got_data[i].euclide_dist;
    let tmp_cal_x_y_angle_res = cal_x_y_angle(cos_sim, euclide_dist);

    new_data.push({
        id: got_data[i].id,
        x: tmp_cal_x_y_angle_res[0] + graph_tree_width / 2,
        y: tmp_cal_x_y_angle_res[1] + graph_tree_height / 2,
        r: 25 * got_data[i].r_ratio,
        angle: tmp_cal_x_y_angle_res[2]
    });
}

console.log(new_data); // Log to verify data

// Append text labels
let lines = graph_tree_svg.selectAll("line.new-line")
    .data(new_data)
    .enter()
    .append("line")
        .attr("class", "new-line")
        .attr("x1", new_data[new_data.length - 1].x)
        .attr("y1", new_data[new_data.length - 1].y)
        .attr("x2", d => d.x)
        .attr("y2", d => d.y)
        .attr("stroke", "grey")

// Append circles
let circles = graph_tree_svg.selectAll("circle.new-circle")
    .data(new_data)
    .enter()
    .append("circle")
        .attr("class", "new-circle")
        .attr("cx", d => d.x)
        .attr("cy", d => d.y)
        .attr("r", d => d.r)
        .attr("fill", d => `hsla(${d.angle}, 100%, 50%, 0.6)`);

let texts1 = graph_tree_svg.selectAll("text.new-text")
    .data(new_data)
    .enter()
    .append("text") // Corrected from "circle" to "text"
        .attr("class", "new-text") // Set a class for the text
        .attr("x", d => d.x) // Set the x position for the text
        .attr("y", d => d.y) // Set the y position for the text
        .text(d => d.id) // Set the text content to the id from new_data
        .attr("text-anchor", "middle") // Center the text
        .attr("dominant-baseline", "middle") // Vertically center the text
        .style("font-size", "12px"); // Optional: set font size

// function updateCircles(new_data) {
//     const circles = graph_tree_svg.selectAll("circle")
//         .data(new_data, d => d.id);

//     const texts = graph_tree_svg.selectAll("text")
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
//     const newCircles = circles.enter().append("circle")
//         .attr("cx", d => d.x)
//         .attr("cy", d => d.y)
//         .attr("r", d => d.r)
//         .style("opacity", 0)  // Start invisible
//         .attr("fill", "blue")
//         .transition()
//         .duration(1000)
//         .style("opacity", 1);  // Fade in

//     // Append new text elements with fade-in effect
//     const newTexts = texts.enter().append("text")
//         .attr("x", d => d.x)
//         .attr("y", d => d.y)
//         .text(d => d.id)
//         .style("opacity", 0)  // Start invisible
//         .transition()
//         .duration(1000)
//         .style("opacity", 1);  // Fade in
// }

// updateCircles(new_data)