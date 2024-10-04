let graph_tree_base_node_r = 30;
let circle_transition_time = 2500;
let max_circle_ratio = 2;


// // Sample data
// let got_data = [
//     {'id': 74, 'cos_sim': -0.3249267290917288, 'euclide_dist': 10, 'r_ratio': 1.5},
//     {'id': 111, 'cos_sim': 0.5, 'euclide_dist': 6, 'r_ratio': 1.5},
//     {'id': 37, 'cos_sim': 1, 'euclide_dist': 0, 'r_ratio': 1.0},
// ];

// function rotateCircle(selection, d_angle, d_dist) {
//     // Calculate initial distance from center
//     let initialX = +selection.attr("cx"); // Get initial x position
//     let initialY = +selection.attr("cy"); // Get initial y position
//     let deltaX = initialX - svg_center_x; // Difference in X
//     let deltaY = initialY - svg_center_y; // Difference in Y

//     let init_dist = Math.sqrt((deltaX) ** 2 + (deltaY) ** 2); // Calculate initial distance from center
//     let init_angle = Math.atan2(deltaY, deltaX) * (180 / Math.PI); // Convert to degrees

//     selection.transition()
//         .duration(2000) // Custom duration for each circle
//         .attrTween("transform", function() {
//             return function(t) {
//                 // Calculate the current angle based on time t
//                 let cur_angle = init_angle + t * d_angle;
//                 let cur_dist = init_dist + t * d_dist;

//                 // Convert degrees to radians
//                 let radians = (cur_angle * Math.PI) / 180;

//                 // Calculate new position based on the angle and distance
//                 let x = cur_dist * Math.cos(radians);
//                 let y = cur_dist * Math.sin(radians);

//                 // Return the translation with the SVG center offset
//                 return `translate(${x - deltaX}, ${y - deltaY})`;
//             };
//         });
// }

function circle_size_scaler(x) {
    let L = max_circle_ratio*2-2; // max circle size follow 1 + 0.5L
    let k = Math.log((L + 1) / (L - 1));
    let exponent = -k * (x - 1);
    let denominator = 1 + Math.exp(exponent);
    return (L / denominator) - (L - 2) / 2;
}

// let circle_size_scaler = d3.scaleLinear()
//     .domain([0, 1])
//     .range([0.5, 1])

function hsla(angle, saturation, lightness, alpha) {
    return `hsla(${angle}, ${saturation}%, ${lightness}%, ${alpha})`
}

// Function to calculate x, y, and angle
function cal_x_y_angle(cos_sim, dist) {
    let normal_radian = Math.acos(cos_sim);
    let angle = normal_radian * (180 / Math.PI);
    let special_radian = normal_radian + (cos_sim >= 0 ? -Math.PI / 4 : Math.PI / 4);

    let x = dist * Math.cos(special_radian);
    let y = dist * Math.sin(special_radian);

    return [x, y, angle];
}

function cal_x1_x2_y1_y2(cos_sim, dist) {
    let [x1, y1, angle] = cal_x_y_angle(cos_sim, graph_tree_base_node_r)
    let [x2, y2, foo2] = cal_x_y_angle(cos_sim, dist)
    return [x1, y1, x2, y2, angle]
}

function server_data_to_graph_data(server_data) {

    let euclide_scaler = d3.scaleLinear()
    .domain(d3.extent(server_data.slice(0, -1), d => d.euclide_dist))
    .range([graph_tree_axis_radius + graph_tree_base_node_r * max_circle_ratio + short_tick_len * 2, Math.min(graph_tree_width / 2, graph_tree_height / 2 / Math.cos(Math.PI / 4)) - graph_tree_base_node_r * max_circle_ratio]);

    let translated_graph_tree_circles_data = [];
    let translated_graph_tree_links_data = [];

    // push center circle
    translated_graph_tree_circles_data.push({
        id: server_data[server_data.length-1].id,
        x: graph_tree_width / 2,
        y: graph_tree_height / 2,
        r: graph_tree_base_node_r * circle_size_scaler(server_data[server_data.length-1].r_ratio),
        angle: 0,
        dist: 0,
    });

    // push other circles
    for (let i = 0; i < server_data.length-1; i++) {
        let cur_r = graph_tree_base_node_r * circle_size_scaler(server_data[i].r_ratio);
        let cur_dist = euclide_scaler(server_data[i].euclide_dist);
        let [x, y, angle_circle] = cal_x_y_angle(server_data[i].cos_sim, cur_dist);
        
        translated_graph_tree_circles_data.push({
            id: server_data[i].id,
            x: x + graph_tree_width / 2,
            y: y + graph_tree_height / 2,
            r: cur_r,
            angle: angle_circle,
            dist: cur_dist,
        });

        let [x1, y1, x2, y2, angle_link] = cal_x1_x2_y1_y2(server_data[i].cos_sim, cur_dist - cur_r);

        translated_graph_tree_links_data.push({
            id: server_data[i].id,
            x1: x1 + graph_tree_width / 2,
            y1: y1 + graph_tree_height / 2,
            x2: x2 + graph_tree_width / 2,
            y2: y2 + graph_tree_height / 2,
            angle: angle_link,
        });
    }
    
    return [translated_graph_tree_circles_data, translated_graph_tree_links_data]
}

function update_circles_links(circle_data, link_data) {
    let circles = graph_tree_svg.selectAll(".cluster_circle")
        .data(circle_data, d => d.id);

    let texts = graph_tree_svg.selectAll(".cluster_lable")
        .data(circle_data, d => d.id);

    let links = graph_tree_svg.selectAll(".cluster_link")
    .data(link_data, d => d.id)

    // Remove circles that are not in the new data with a fade-out transition
    links.exit()
        .transition()
        .duration(circle_transition_time)
        .style("opacity", 0)
        .remove();

    circles.exit()
        .transition()
        .duration(circle_transition_time)
        .style("opacity", 0)
        .remove();

    texts.exit()
        .transition()
        .duration(circle_transition_time)
        .style("opacity", 0)
        .remove();


    // Update existing circles (matching ids) with transitions
    links
        .transition()
        .duration(circle_transition_time)
        .attr("x1", d => d.x1)
        .attr("y1", d => d.y1)
        .attr("x2", d => d.x2)
        .attr("y2", d => d.y2)
        .attr("stroke", d => hsla(d.angle, 100, 30, 0.4))

    circles
        .transition()
        .duration(circle_transition_time)
        .attr("cx", d => d.x)
        .attr("cy", d => d.y)
        .attr("r", d => d.r)
        .attr("fill", d => hsla((d.angle + cur_color_angle) % 360, 100, 50, 0.4))
        .attr("stroke", d => hsla((d.angle + cur_color_angle) % 360, 100, 30, 0.4));

    texts
        .transition()
        .duration(circle_transition_time)
        .attr("x", d => d.x)
        .attr("y", d => d.y);

    // Append new links with a fade-in effect
    links.enter().append("line")
        .attr("x1", d => d.x1)
        .attr("y1", d => d.y1)
        .attr("x2", d => d.x2)
        .attr("y2", d => d.y2)
        .attr("stroke", d => `hsla(${(d.angle + cur_color_angle) % 360}, 100%, 30%, 0.4)`)
        .attr("class", "cluster_link")
            .style("opacity", 0)  // Start invisible
            .transition()
            .duration(circle_transition_time)
            .style("opacity", 1);  // Fade in

    // Append new circles with a fade-in effect
    circles.enter().append("circle")
        .attr("cx", d => d.x)
        .attr("cy", d => d.y)
        .attr("r", d => d.r)
        .attr("fill", d => `hsla(${(d.angle + cur_color_angle) % 360}, 100%, 50%, 0.4)`)
        .attr("stroke", d => `hsla(${(d.angle + cur_color_angle) % 360}, 100%, 30%, 0.4)`)         // Add stroke color
        .attr("stroke-width", 1)
        .attr("class", "cluster_circle")
        .on("click", function(event, d) {
            graph_tree_circle_clicked(d.id)
        })
            .style("opacity", 0)  // Start invisible
            .transition()
            .duration(circle_transition_time)
            .style("opacity", 1);  // Fade in

    // Append new text elements with fade-in effect
    texts.enter().append("text")
        .attr("x", d => d.x)
        .attr("y", d => d.y)
        .attr("class", "cluster_lable")
        .style("text-anchor", "middle")
        .style("text-anchor", "middle")
        .style("dominant-baseline", "central")
        .text(d => d.id)
        .on("click", function(event, d) {
            graph_tree_circle_clicked(d.id)
        })
            .style("opacity", 0)  // Start invisible
            .transition()
            .duration(circle_transition_time)
            .style("opacity", 1);  // Fade in

    d3.select("line").lower();
}

function render_graph_tree(server_data) {
    console.log(server_data)
    let [new_circle_data, new_link_data] = server_data_to_graph_data(server_data);
    update_circles_links(new_circle_data, new_link_data)
}

function graph_tree_circle_clicked(id) {
    postData('/graph_tree_circle_click', { "id": id }, postID = "send circle click")
        .then(data => {render_graph_tree(data.graph_tree_data);});
}