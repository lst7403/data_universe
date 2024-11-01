let graph_tree_base_node_r = 30;
let max_circle_ratio = 2;
let vertical_y_gap = graph_tree_base_node_r * 1.5
let vertical_x_gap = graph_tree_base_node_r * 1.5
let vertical_center_gap = graph_tree_base_node_r * 4

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
//                 let cur_color_angle = init_angle + t * d_angle;
//                 let cur_dist = init_dist + t * d_dist;

//                 // Convert degrees to radians
//                 let radians = (cur_color_angle * Math.PI) / 180;

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
    let cur_cos_sim = data.cos_sim
    let radian = Math.acos(cur_cos_sim)
    let cur_color_angle = rad_2_ang(radian)
    let special_radian = radian + (cur_cos_sim >= 0 ? -Math.PI / 4 : Math.PI / 4)

    let cur_r = graph_tree_base_node_r * circle_size_scaler(data.r_ratio);
    let cur_dist = scaler(data.euclide_dist);
    let [x, y] = cal_x_y(special_radian, cur_dist);


    let [x1, y1, x2, y2] = cal_x1_y1_x2_y2(special_radian, graph_tree_base_node_r, cur_dist - cur_r);

    return [{
        id: data.id,
        x: x + graph_tree_width / 2,
        y: y + graph_tree_height / 2,
        r: cur_r,
        angle: cur_color_angle,
        dist: cur_dist,
        cluster_type: "neighbour",
    }, {
        id: data.id,
        x1: x1 + graph_tree_width / 2,
        y1: y1 + graph_tree_height / 2,
        x2: x2 + graph_tree_width / 2,
        y2: y2 + graph_tree_height / 2,
        angle: cur_color_angle,
    }]
}

function translate_child_data(data) {
    let cur_r = graph_tree_base_node_r * circle_size_scaler(data.r_ratio);
    let x = graph_tree_width / 2 + vertical_x_gap * data.dx
    let y = graph_tree_height / 2 + vertical_center_gap + vertical_y_gap * data.dy
    let lightness = 80

    return [{
        id: data.id,
        x: x,
        y: y,
        r: cur_r,
        lightness: lightness,
    }, {
        id: data.id,
        source: {x: graph_tree_width / 2, y: graph_tree_height / 2 + graph_tree_base_node_r},
        target: {x: x, y: y - cur_r},
        lightness: lightness - 20,
    }]
}

function translate_parent_data(data) {
    let cur_r = graph_tree_base_node_r * circle_size_scaler(data.r_ratio);
    let y = graph_tree_height / 2 - vertical_center_gap
    let x = graph_tree_width / 2
    let lightness = 40

    return [{
        id: data.id,
        x: x,
        y: y,
        r: cur_r,
        lightness: lightness,
    }, {
        id: data.id,
        source: {x: x, y: graph_tree_height / 2 - graph_tree_base_node_r},
        target: {x: x, y: y + cur_r},
        lightness: lightness - 20,
    }]
}

function server_data_to_graph_data(server_graph_tree_horizontal_data, server_graph_tree_vertical_data) {

    let euclide_scaler = d3.scaleLinear()
    .domain(d3.extent(server_graph_tree_horizontal_data.filter(d => d.cluster_type == "neighbour"), d => d.euclide_dist))
    .range([graph_tree_axis_radius + graph_tree_base_node_r * max_circle_ratio + short_tick_len * 2, Math.min(graph_tree_width / 2, graph_tree_height / 2 / Math.cos(Math.PI / 4)) - graph_tree_base_node_r * max_circle_ratio]);

    let translated_graph_tree_horizontal_circle_data = [];
    let translated_graph_tree_horizontal_links_data = [];

    for (let i = 0; i < server_graph_tree_horizontal_data.length; i++) {
        if (server_graph_tree_horizontal_data[i].cluster_type == "neighbour") {
            let [neighbour_circle, neighbour_link]  = translate_neighbour_data(server_graph_tree_horizontal_data[i], euclide_scaler)
            translated_graph_tree_horizontal_circle_data.push(neighbour_circle);
            translated_graph_tree_horizontal_links_data.push(neighbour_link);
        } else {
            translated_graph_tree_horizontal_circle_data.push({
                id: server_graph_tree_horizontal_data[i].id,
                x: graph_tree_width / 2,
                y: graph_tree_height / 2,
                r: graph_tree_base_node_r,
                angle: 0,
                dist: 0,
                cluster_type: "center",
            })
        }
    }

    let translated_graph_tree_vertical_circle_data = []
    let translated_graph_tree_vertical_link_data = []

    for (let i = 0; i < server_graph_tree_vertical_data.length; i++) {
        if (server_graph_tree_vertical_data[i].cluster_type == "child") {
            let [child_circle, child_link]  = translate_child_data(server_graph_tree_vertical_data[i])
            translated_graph_tree_vertical_circle_data.push(child_circle);
            translated_graph_tree_vertical_link_data.push(child_link);
        } else {
            let [parent_circle, parent_link] = translate_parent_data(server_graph_tree_vertical_data[i])
            translated_graph_tree_vertical_circle_data.push(parent_circle);
            translated_graph_tree_vertical_link_data.push(parent_link);
        }
    }

    console.log("hi", translated_graph_tree_horizontal_circle_data, translated_graph_tree_horizontal_links_data)

    return [translated_graph_tree_horizontal_circle_data,
        translated_graph_tree_horizontal_links_data,
        translated_graph_tree_vertical_circle_data,
        translated_graph_tree_vertical_link_data]
}

/////////////////////////////////////////////////////////////////////////////////////////////////////

function move_to_center(id, duration) {
    return new Promise((resolve) => {
        path_select = graph_tree_svg.select(`#path${id}`)
        line_select = graph_tree_svg.select(`#link${id}`)

        graph_tree_svg.selectAll(`#path${id}, #link${id}, circle.center, text.center`)
            .transition()
            .duration(duration)
            .style("opacity", 0)
            .remove()
        

        // // Remove the .center elements separately
        // graph_tree_svg.select("circle.center")
        //     .transition()
        //     .duration(duration)
        //     .style("opacity", 0)
        //     .remove()

        // graph_tree_svg.select("text.center")
        //     .transition()
        //     .duration(duration)
        //     .style("opacity", 0)
        //     .remove()

        graph_tree_svg.select(`#circle${id}`)
            .transition()
            .duration(duration)
            .attr("cx", graph_tree_width/2)
            .attr("cy", graph_tree_height/2)
            .attr("r", graph_tree_base_node_r)
            .classed("center", 1)

        graph_tree_svg.select(`#text${id}`)
            .transition()
            .duration(duration)
            .attr("x", graph_tree_width/2)
            .attr("y", graph_tree_height/2)
            .classed("center", 1)
            .on("end", resolve)
    })
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
        .attr("id", d => "link"+d.id)
        .attr("x1", d => d.x1)
        .attr("y1", d => d.y1)
        .attr("x2", d => d.x2)
        .attr("y2", d => d.y2)
        .attr("stroke", d => hsla((d.angle + cur_color_angle) % 360, 100, 30, 0.4))
        .attr("class", "horizontal_link")
            .style("opacity", 0)  // Start invisible
            .transition()
            .delay(delay)
            .duration(duration)
            .style("opacity", 1);  // Fade in
}

function custom_circle_enter(circles_update, delay, duration) {
    circles_update.enter()
        .append("circle")
        .attr("id", d => "circle"+d.id)
        .attr("cx", d => d.x)
        .attr("cy", d => d.y)
        .attr("r", d => d.r)
        .attr("fill", d => hsla((d.angle + cur_color_angle) % 360, 100, 50, 0.4))
        .attr("stroke", d => hsla((d.angle + cur_color_angle) % 360, 100, 30, 0.4))
        .attr("stroke-width", 1)
        .attr("class", "horizontal_circle")
        .classed("center", d => d.cluster_type == "center" ? 1 : 0)
        .on("click", function(event, d) {
            move_to_center(d.id, circle_transition_time)
            .then(graph_tree_circle_clicked(d.id))
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
        .attr("id", d => "text"+d.id)
        .attr("x", d => d.x)
        .attr("y", d => d.y)
        .attr("class", "horizontal_label cluster_label")
        .classed("center", d => d.cluster_type == "center" ? 1 : 0)
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

function update_horizontal_circles_links(circle_data, link_data, delay) {

    let circles_update = graph_tree_svg.selectAll(".horizontal_circle")
        .data(circle_data, d => d.id)

    let texts_update = graph_tree_svg.selectAll(".horizontal_label")
        .data(circle_data, d => d.id)

    let links_update = graph_tree_svg.selectAll(".horizontal_link")
        .data(link_data, d => d.id)

    let links_exited = custom_exit(links_update, circle_transition_time)
    let circles_exited = custom_exit(circles_update, circle_transition_time)
    let texts_exited = custom_exit(texts_update, circle_transition_time)

    let enter_delay = (circles_exited.size() > 0 ? delay + circle_transition_time : delay)
    let update_delay = (circles_update.size() > 0 ? enter_delay + circle_transition_time : enter_delay)

    custom_link_enter(links_update, enter_delay, circle_transition_time)
    custom_circle_enter(circles_update, enter_delay, circle_transition_time)
    custom_text_enter(texts_update, enter_delay, circle_transition_time)

    custom_link_update(links_update, update_delay, circle_transition_time)
    custom_circle_update(circles_update, update_delay, circle_transition_time)
    custom_text_update(texts_update, update_delay, circle_transition_time)
}


let linkGenerator = d3.linkVertical()
      .x(d => d.x)
      .y(d => d.y);

function custom_vertical_data_link_enter(links_update, delay, duration) {
    links_update.enter()
        .append("path")
        .attr("id", d => "path"+d.id)
        .attr("class", "vertical_link")
        .attr("d", d => linkGenerator({ source: d.source, target: d.source }))
        .attr("fill", "none")
        .style("opacity", 0)
        .attr("stroke", d => hsla(cur_color_angle, 100, d.lightness, 0.3))
            .transition()
            .delay(delay)
            .duration(duration)  // 1 second transition
            .attr("d", d => linkGenerator({ source: d.source, target: d.target })) 
            .style("opacity", 1) // Fade to fully visible
}

function custom_vertical_data_circle_enter(circles_update, delay, duration) {
    circles_update.enter()
        .append("circle")
        .attr("id", d => "circle"+d.id)
        .attr("cx", graph_tree_width / 2)
        .attr("cy", graph_tree_height / 2)
        .attr("r", 0)
        .style("opacity", 0)
        .attr("fill", d => hsla(cur_color_angle, 100, d.lightness, 0.4))
        .attr("stroke", d => hsla(cur_color_angle, 100, d.lightness - 20, 0.4))
        .attr("stroke-width", 1)
        .attr("class", "vertical_circle")
        // .on("click", function(event, d) {
        //     graph_tree_circle_clicked(d.id)
        // })
            .transition()
            .delay(delay)
            .duration(duration)
            .style("opacity", 1)
            .attr("r", d => d.r)
            .attr("cx", d => d.x)
            .attr("cy", d => d.y)
}

function custom_vertical_data_text_enter(texts_update, delay, duration) {
    texts_update.enter()
        .append("text")
        .attr("id", d => "text"+d.id)
        .attr("x", graph_tree_width / 2)
        .attr("y", graph_tree_height / 2)
        .text(d => d.id)
        .style("opacity", 0)
        .attr("class", "vertical_label cluster_label")
        // .on("click", function(event, d) {
        //     graph_tree_circle_clicked(d.id)
        // })
        .transition()
        .delay(delay) // Delay before starting the transition
        .duration(duration) // Duration of the transition
        .style("opacity", 1) // Fade in to fully visible
        .style("font-size", d => Math.max(14, d.r * 0.3)) // Adjust font size based on radius
        .attr("x", d => d.x) // Transition to the new x position
        .attr("y", d => d.y); // Transition to the new y position
}

function custom_vertical_data_link_update(links_update, delay, duration) {
    links_update
        .transition()
        .delay(delay)
        .duration(duration)  // 1 second transition
        .attr("d", d => linkGenerator({ source: d.source, target: d.target })) 
}

function custom_vertical_data_circle_update(circles_update, delay, duration) {
    circles_update
        .transition()
        .delay(delay)
        .duration(duration)
        .attr("r", d => d.r)
        .attr("fill", d => hsla(cur_color_angle, 100, d.lightness, 0.4))
        .attr("stroke", d => hsla(cur_color_angle, 100, d.lightness - 20, 0.4));
}


function update_vertical_circles_links(circle_data, link_data, delay) {

    let circles_update = graph_tree_svg.selectAll(".vertical_circle")
        .data(circle_data, d => d.id);

    let texts_update = graph_tree_svg.selectAll(".vertical_label")
        .data(circle_data, d => d.id);

    let links_update = graph_tree_svg.selectAll(".vertical_link")
        .data(link_data, d => d.id)

    let links_exited = custom_exit(links_update, circle_transition_time)
    let circles_exited = custom_exit(circles_update, circle_transition_time)
    let texts_exited = custom_exit(texts_update, circle_transition_time)

    let update_delay = (circles_exited.size() > 0 ? delay + circle_transition_time : delay)
    let enter_delay = (circles_update.size() > 0 ? update_delay + circle_transition_time : update_delay)

    custom_vertical_data_link_update(links_update, update_delay, circle_transition_time)
    custom_vertical_data_circle_update(circles_update, update_delay, circle_transition_time)
    // custom_text_update(texts_update, update_delay, circle_transition_time)

    custom_vertical_data_link_enter(links_update, enter_delay, circle_transition_time)
    custom_vertical_data_circle_enter(circles_update, enter_delay, circle_transition_time)
    custom_vertical_data_text_enter(texts_update, enter_delay, circle_transition_time)
    
}

function render_graph_tree(server_graph_tree_data, delay) {
    let [translated_graph_tree_horizontal_circle_data,
        translated_graph_tree_horizontal_links_data,
        translated_graph_tree_vertical_circle_data,
        translated_graph_tree_vertical_link_data] = server_data_to_graph_data(server_graph_tree_data.graph_tree_horizontal_data, server_graph_tree_data.graph_tree_vertical_data);
    
        // console.log("new_graph_tree_data", translated_graph_tree_horizontal_circle_data,
        // translated_graph_tree_horizontal_links_data,
        // translated_graph_tree_vertical_circle_data,
        // translated_graph_tree_vertical_link_data)

    update_horizontal_circles_links(translated_graph_tree_horizontal_circle_data, translated_graph_tree_horizontal_links_data, delay)
    update_vertical_circles_links(translated_graph_tree_vertical_circle_data, translated_graph_tree_vertical_link_data, delay)
}

function graph_tree_circle_clicked(id) {
    postData('/graph_tree_circle_click', { "id": id }, postID = "send circle click")
        .then(data => {render_graph_tree(data.graph_tree_data, circle_transition_time)});
}