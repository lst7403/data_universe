const graph_tree_container = d3.select("#graph-tree-area");

// Get the dimensions of the container
const graph_tree_width = graph_tree_container.node().clientWidth;  // Get the width in pixels
const graph_tree_height = graph_tree_container.node().clientHeight; // Get the height in pixels

const graph_tree_axis_radius = 60;
const text_to_axis_margin = 2;
const short_tick_len = 6;
const stroke_width = 0.6;
const font_size = 14;

let graph_tree_svg = graph_tree_container.append('svg')
    .attr('width', graph_tree_width)
    .attr('height', graph_tree_height);

let group = graph_tree_svg.append('g')
    .classed("axis", 1)
    .attr('transform', `translate(${graph_tree_width / 2}, ${graph_tree_height / 2})`);

// 生成刻度
let ticks = [];
for (let i = 0; i <= 90; i += 5) {
    let special_radian = i * Math.PI / 180 - Math.PI / 4;
    let innerRadius = graph_tree_axis_radius + (i % 30 === 0 ? short_tick_len*2 : short_tick_len);
    let outerRadius = graph_tree_axis_radius;

    let x1 = innerRadius * Math.cos(special_radian);
    let y1 = innerRadius * Math.sin(special_radian);
    let x2 = outerRadius * Math.cos(special_radian);
    let y2 = outerRadius * Math.sin(special_radian);

    ticks.push({
        x1: x1, y1: y1, x2: x2, y2: y2, angle: i
    });
}

for (let i = 90; i <= 180; i += 5) {
    let special_radian = i * Math.PI / 180 + Math.PI / 4;
    let innerRadius = graph_tree_axis_radius + (i % 30 === 0 ? short_tick_len*2 : short_tick_len);
    let outerRadius = graph_tree_axis_radius;

    let x1 = innerRadius * Math.cos(special_radian);
    let y1 = innerRadius * Math.sin(special_radian);
    let x2 = outerRadius * Math.cos(special_radian);
    let y2 = outerRadius * Math.sin(special_radian);

    ticks.push({
        x1: x1, y1: y1, x2: x2, y2: y2, angle: i
    });
}

// 绘制刻度线
let axis = group.selectAll('line')
    .data(ticks)
    .enter()
        .append('line')
        .attr('x1', d => d.x1)
        .attr('y1', d => d.y1)
        .attr('x2', d => d.x2)
        .attr('y2', d => d.y2)
        .attr('stroke-width', d => (d.angle % 30 === 0 ? stroke_width/1.5 : stroke_width)) //  
        .attr('stroke', d => `hsla(${d.angle}, 100%, 40%, 1)`);  // 初始化颜色

let texts = [];
// Create text for the right side (0-90 degrees)
let innerRadius = graph_tree_axis_radius - text_to_axis_margin;
for (let i = 0; i <= 90; i += 30) {
    let special_radian = i * Math.PI / 180 - Math.PI / 4;

    let x = innerRadius * Math.cos(special_radian);
    let y = innerRadius * Math.sin(special_radian);

    texts.push({
        x: x, y: y, angle: i, anchor: "end"
    });
}

for (let i = 90; i <= 180; i += 30) {
    let special_radian = i * Math.PI / 180 + Math.PI / 4;

    let x = innerRadius * Math.cos(special_radian);
    let y = innerRadius * Math.sin(special_radian);

    texts.push({
        x: x, y: y, angle: i, anchor: "start"
    });
}

let axis_text = group.selectAll('text')
    .data(texts)
    .enter()
        .append('text')
        .text(d => d.angle+"")
        .attr('x', d => d.x)
        .attr('y', d => d.y)
        .attr('fill', d => `hsla(${d.angle}, 100%, 40%, 1)`)
        .attr("font-size", font_size)
        .attr("text-anchor", d => d.anchor)
        .attr("dominant-baseline", "middle")
        .attr("transform", d => `rotate(${d.anchor === "end" ? d.angle - 45 : d.angle -135 }, ${d.x}, ${d.y})`);

// a = group.append('circle')
//     .attr('cx', 0)
//     .attr('cy', 0)
//     .attr('r', radius - text_to_axis_margin - 1.8*font_size - 2)
//     .attr("stroke", "blue")
//     .attr("fill", "none");

// a = group.append('circle')
//     .attr('cx', 0)
//     .attr('cy', 0)
//     .attr('r', 25)
//     .attr("stroke", "blue")
//     .attr("fill", "none");

function change_color() {
    let duration = 1000;

    // Update color for lines
    axis.transition()
        .duration(duration)
        .attr('stroke', d => `hsla(${(d.angle + cur_color_ang) % 360}, 100%, 45%, 1)`);

    // Update color for text
    axis_text.transition()
        .duration(duration)
        .attr('fill', d => `hsla(${(d.angle + cur_color_ang) % 360}, 100%, 45%, 1)`);
}
