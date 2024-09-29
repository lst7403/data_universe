const graph_tree_base_node_r = 25;

// Sample data
let got_data = [
    {'id': 74, 'cos_sim': -0.3249267290917288, 'euclide_dist': 10, 'r_ratio': 1.5384615384615385},
    {'id': 111, 'cos_sim': -0.11059286214225257, 'euclide_dist': 6, 'r_ratio': 2},
    {'id': 28, 'cos_sim': 0.7107, 'euclide_dist': 14, 'r_ratio': 0.5},
    {'id': 18, 'cos_sim': -0.9, 'euclide_dist': 0, 'r_ratio': 2},
    {'id': 37, 'cos_sim': 1, 'euclide_dist': 0, 'r_ratio': 1.0},
];

// Create a scale for the euclidean distance
let euclide_scaler = d3.scaleLinear()
    .domain(d3.extent(got_data.map(d => d.euclide_dist)))
    .range([graph_tree_axis_radius + graph_tree_base_node_r * 2 + short_tick_len * 2, Math.min(graph_tree_width / 2, graph_tree_height / 2 / Math.cos(Math.PI / 4)) - graph_tree_base_node_r * 2]);

// Function to calculate x, y, and angle
function cal_x_y_angle(cos_sim, euclide_dist) {
    let normal_radian = Math.acos(cos_sim);
    let angle = normal_radian * (180 / Math.PI);
    let special_radian = normal_radian + (cos_sim >= 0 ? -Math.PI / 4 : Math.PI / 4);
    let scaled_dist = euclide_scaler(euclide_dist);

    return [scaled_dist * Math.cos(special_radian), scaled_dist * Math.sin(special_radian), angle];
}


function rotateCircle(selection, d_angle, d_dist) {
    // Calculate initial distance from center
    const initialX = +selection.attr("cx"); // Get initial x position
    const initialY = +selection.attr("cy"); // Get initial y position
    const deltaX = initialX - svg_center_x; // Difference in X
    const deltaY = initialY - svg_center_y; // Difference in Y

    const init_dist = Math.sqrt((deltaX) ** 2 + (deltaY) ** 2); // Calculate initial distance from center
    const init_angle = Math.atan2(deltaY, deltaX) * (180 / Math.PI); // Convert to degrees

    selection.transition()
        .duration(2000) // Custom duration for each circle
        .attrTween("transform", function() {
            return function(t) {
                // Calculate the current angle based on time t
                const cur_angle = init_angle + t * d_angle;
                const cur_dist = init_dist + t * d_dist;

                // Convert degrees to radians
                const radians = (cur_angle * Math.PI) / 180;

                // Calculate new position based on the angle and distance
                const x = cur_dist * Math.cos(radians);
                const y = cur_dist * Math.sin(radians);

                // Return the translation with the SVG center offset
                return `translate(${x - deltaX}, ${y - deltaY})`;
            };
        });
}


// Create new data array
let new_data = [];
for (let i = 0; i < got_data.length; i++) {
    if (i == got_data.length-1){
        new_data.push({
            id: got_data[i].id,
            x: graph_tree_width / 2,
            y: graph_tree_height / 2,
            r: 25 * got_data[i].r_ratio,
            angle: 0
        });
        break
    }
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

let cur_links = [];

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

// append circles and text in group
let circle_groups = graph_tree_svg.selectAll("g.cluster")
    .data(new_data)
    .enter()
    .append("g")
        .classed("cluster", 1)
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