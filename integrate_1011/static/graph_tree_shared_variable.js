let cur_color_angle = 180;

let circle_transition_time = 1500;

let graph_tree_container = d3.select("#graph-tree-area");

// Get the dimensions of the container
let graph_tree_width = graph_tree_container.node().clientWidth;  // Get the width in pixels
let graph_tree_height = graph_tree_container.node().clientHeight; // Get the height in pixels

let graph_tree_axis_radius = 60;
let text_to_axis_margin = 2;

let graph_tree_svg = graph_tree_container.append('svg')
    .attr('width', graph_tree_width)
    .attr('height', graph_tree_height);

let group = graph_tree_svg.append('g')
    .classed("axis", 1)
    .attr('transform', `translate(${graph_tree_width / 2}, ${graph_tree_height / 2})`);

function hsla(angle, saturation, lightness, alpha) {
    return `hsla(${angle}, ${saturation}%, ${lightness}%, ${alpha})`
}

function cal_x_y(radian, r) {
    let x = r * Math.cos(radian);
    let y = r * Math.sin(radian);
    return [x, y];
}

function cal_x1_y1_x2_y2(radian, start_r, end_r) {
    let [x1, y1] = cal_x_y(radian, start_r)
    let [x2, y2] = cal_x_y(radian, end_r)
    return [x1, y1, x2, y2]
}

function rad_2_ang(radian) {
    return radian * (180 / Math.PI);
}

function ang_2_rad(angle) {
    return angle * (Math.PI / 180)
}