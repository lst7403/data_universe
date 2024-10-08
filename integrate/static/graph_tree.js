let graph_tree_base_node_r = 30;
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

function translate_neighbour_data(data, scaler) {
    let cur_cos_sim = data[i].cos_sim
    let radian = Math.acos(cur_cos_sim)
    let cur_angle = rad_2_ang(radian)
    let special_radian = radian + (cur_cos_sim >= 0 ? -Math.PI / 4 : Math.PI / 4)

    let cur_r = graph_tree_base_node_r * circle_size_scaler(data[i].r_ratio);
    let cur_dist = scaler(data[i].euclide_dist);
    let [x, y] = cal_x_y(special_radian, cur_dist);


    let [x1, y1, x2, y2] = cal_x1_y1_x2_y2(special_radian, graph_tree_base_node_r, cur_dist - cur_r);

    return [{
        id: data[i].id,
        x: x + graph_tree_width / 2,
        y: y + graph_tree_height / 2,
        r: cur_r,
        angle: cur_angle,
        dist: cur_dist,
    }, {
        id: server_data[i].id,
        x1: x1 + graph_tree_width / 2,
        y1: y1 + graph_tree_height / 2,
        x2: x2 + graph_tree_width / 2,
        y2: y2 + graph_tree_height / 2,
        angle: cur_angle,
    }]
}

function translate_child_data(data, base_child_x_gap, base_child_y_gap) {
    let cur_r = graph_tree_base_node_r * circle_size_scaler(data.r_ratio);
    let x = graph_tree_width / 2 + base_child_x_gap * data.dx
    let y = graph_tree_height / 2 + graph_tree_base_node_r + base_child_y_gap * data.dy

    return [{
        id: data.id,
        x: x,
        y: y,
        r: cur_r,
        angle: cur_angle,
    }, {
        id: data.id,
        x1: graph_tree_width / 2,
        y1: graph_tree_height / 2 + graph_tree_base_node_r,
        x2: x,
        y2: y - cur_r,
        angle: cur_angle,
    }]
}

function translate_parent_data(data) {
    let cur_r = graph_tree_base_node_r * circle_size_scaler(data.r_ratio);
    let y = graph_tree_height / 2 - graph_tree_base_node_r
    let x = graph_tree_width / 2
    
    return [{
        id: data.id,
        x: x,
        y: y - cur_r,
        r: cur_r,
        angle: cur_angle,
    }, {
        id: data.id,
        x1: x,
        y1: graph_tree_height / 2 + graph_tree_base_node_r,
        x2: x,
        y2: y,
        angle: cur_angle,
    }]
}



function server_data_to_graph_data(server_data) {
    
    let euclide_scaler = d3.scaleLinear()
    .domain(d3.extent(server_data.filter(d => d.cluster_type == "neighbour"), d => d.euclide_dist))
    .range([graph_tree_axis_radius + graph_tree_base_node_r * max_circle_ratio + short_tick_len * 2, Math.min(graph_tree_width / 2, graph_tree_height / 2 / Math.cos(Math.PI / 4)) - graph_tree_base_node_r * max_circle_ratio]);

    let translated_graph_tree_circles_data = [];
    let translated_graph_tree_links_data = [];

    // push center circle
    translated_graph_tree_circles_data.push({
        id: server_data[server_data.length-1].id,
        x: graph_tree_width / 2,
        y: graph_tree_height / 2,
        r: graph_tree_base_node_r,
        angle: cur_color_angle,
        dist: 0,
    });

    // push other circles
    for (let i = 0; i < server_data.length; i++) {
        switch (server_data[i].cluster_type){
            case "neighbour":
                let [neighbour_circle, neighbour_link]  = translate_neighbour_data(server_data[i], scaler)
                translated_graph_tree_circles_data.push(neighbour_circle);
                translated_graph_tree_links_data.push(neighbour_link);
                break
            case "child":
                let [child_circle, child_link]  = translate_child_data(server_data[i], graph_tree_base_node_r, graph_tree_base_node_r)
                translated_graph_tree_circles_data.push(child_circle);
                translated_graph_tree_links_data.push(child_link);
                break
            case "parent":
                let [parent_circle, parent_link] = translate_parent_data(server_data[i])
                translated_graph_tree_circles_data.push(parent_circle);
                translated_graph_tree_links_data.push(parent_link);
            case "center":
                translated_graph_tree_circles_data.push({
                    id: server_data[server_data.length-1].id,
                    x: graph_tree_width / 2,
                    y: graph_tree_height / 2,
                    r: graph_tree_base_node_r,
                    angle: cur_color_angle,
                    dist: 0,
                });
        }
        
        
    }
    
    return [translated_graph_tree_circles_data, translated_graph_tree_links_data]
}

function custom_exit(updated_data, duration) {
    return updated_data.exit()
        .transition()
        .duration(duration)
        .style("opacity", 0)
        .remove()
}

function custom_link_update(links_update, delay, duration) {
    links_update
        .transition()
        .delay(delay)
        .duration(duration)
        .attr("x1", d => d.x1)
        .attr("y1", d => d.y1)
        .attr("x2", d => d.x2)
        .attr("y2", d => d.y2)
        .attr("stroke", d => hsla((d.angle + cur_color_angle) % 360, 100, 30, 0.4))
}

function custom_circle_update(circles_update, delay, duration) {
    circles_update
        .transition()
        .delay(delay)
        .duration(duration)
        .attr("cx", d => d.x)
        .attr("cy", d => d.y)
        .attr("r", d => d.r)
        .attr("fill", d => hsla((d.angle + cur_color_angle) % 360, 100, 50, 0.4))
        .attr("stroke", d => hsla((d.angle + cur_color_angle) % 360, 100, 30, 0.4));
}

function custom_text_update(texts_update, delay, duration) {
    texts_update
        .transition()
        .delay(delay)
        .duration(duration)
        .attr("x", d => d.x)
        .attr("y", d => d.y)
        .style("font-size", d => Math.max(14, d.r*0.3));
}

function custom_link_enter(links_update, delay, duration) {
    links_update.enter()
        .append("line")
        .attr("x1", d => d.x1)
        .attr("y1", d => d.y1)
        .attr("x2", d => d.x2)
        .attr("y2", d => d.y2)
        .attr("stroke", d => hsla((d.angle + cur_color_angle) % 360, 100, 30, 0.4))
        .attr("class", "cluster_link")
            .style("opacity", 0)  // Start invisible
            .transition()
            .delay(delay)
            .duration(duration)
            .style("opacity", 1);  // Fade in
}

function custom_circle_enter(circles_update, delay, duration) {
    circles_update.enter()
        .append("circle")
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
            .delay(delay)
            .duration(duration)
            .style("opacity", 1);  // Fade in
}

function custom_text_enter(texts_update, delay, duration) {
    texts_update.enter()
        .append("text")
        .attr("x", d => d.x)
        .attr("y", d => d.y)
        .attr("class", "cluster_label")
        .style("font-size", d => Math.max(14, d.r*0.3))
        .text(d => d.id)
        .on("click", function(event, d) {
            graph_tree_circle_clicked(d.id)
        })
            .style("opacity", 0)  // Start invisible
            .transition()
            .delay(delay)
            .duration(duration)
            .style("opacity", 1);  // Fade in
}

function update_circles_links(circle_data, link_data) {

    let circles_update = graph_tree_svg.selectAll(".cluster_circle")
        .data(circle_data, d => d.id);

    let texts_update = graph_tree_svg.selectAll(".cluster_label")
        .data(circle_data, d => d.id);

    let links_update = graph_tree_svg.selectAll(".cluster_link")
        .data(link_data, d => d.id)

    let links_exited = custom_exit(links_update, circle_transition_time)
    let circles_exited = custom_exit(circles_update, circle_transition_time)
    let texts_exited = custom_exit(texts_update, circle_transition_time)

    let update_delay = (circles_exited.size() > 0 ? circle_transition_time : 0)
    let enter_delay = (circles_update.size() > 0 ? update_delay + circle_transition_time : update_delay)

    custom_link_update(links_update, update_delay, circle_transition_time)
    custom_circle_update(circles_update, update_delay, circle_transition_time)
    custom_text_update(texts_update, update_delay, circle_transition_time)

    custom_link_enter(links_update, enter_delay, circle_transition_time)
    custom_circle_enter(circles_update, enter_delay, circle_transition_time)
    custom_text_enter(texts_update, enter_delay, circle_transition_time)
    
}

function render_graph_tree(server_data) {
    let [new_circle_data, new_link_data] = server_data_to_graph_data(server_data);
    console.log("new_graph_tree_data", new_circle_data, new_link_data)
    update_circles_links(new_circle_data, new_link_data)
}

function graph_tree_circle_clicked(id) {
    postData('/graph_tree_circle_click', { "id": id }, postID = "send circle click")
        .then(data => {render_graph_tree(data.graph_tree_data);});
}