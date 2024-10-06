let short_tick_len = 6;
let axis_stroke_width = 0.6;
let axis_font_size = 14;

// 生成刻度
let ticks_and_lable = [];
for (let i = 0; i <= 90; i += 5) {
    let special_radian = i * Math.PI / 180 - Math.PI / 4;
    let outerRadius = graph_tree_axis_radius + (i % 30 === 0 ? short_tick_len*2 : short_tick_len);

    let [x1, y1, x2, y2] = cal_x1_y1_x2_y2(special_radian, graph_tree_axis_radius, outerRadius)
    let [text_x, text_y] = cal_x_y(special_radian, graph_tree_axis_radius - text_to_axis_margin)

    ticks_and_lable.push({
        x1: x1, y1: y1, x2: x2, y2: y2, angle: i, text_x, text_y, text_anchor: "end"
    });
}

for (let i = 90; i <= 180; i += 5) {
    let special_radian = i * Math.PI / 180 + Math.PI / 4;
    let outerRadius = graph_tree_axis_radius + (i % 30 === 0 ? short_tick_len*2 : short_tick_len);

    let [x1, y1, x2, y2] = cal_x1_y1_x2_y2(special_radian, graph_tree_axis_radius, outerRadius)
    let [text_x, text_y] = cal_x_y(special_radian, graph_tree_axis_radius - text_to_axis_margin)

    ticks_and_lable.push({
        x1: x1, y1: y1, x2: x2, y2: y2, angle: i, text_x, text_y, text_anchor: "start"
    });
}

// 绘制刻度线
let axis = group.selectAll('.axis_line')
    .data(ticks_and_lable)
    .enter()
        .append('line')
        .attr('x1', d => d.x1)
        .attr('y1', d => d.y1)
        .attr('x2', d => d.x2)
        .attr('y2', d => d.y2)
        .attr("class", "axis_line")
        .attr('stroke-width', d => (d.angle % 30 === 0 ? axis_stroke_width/1.5 : axis_stroke_width)) //  
        .attr('stroke', d => hsla((d.angle + cur_color_angle) % 360, 100, 40, 1));  // 初始化颜色

let axis_text = group.selectAll('axis_text')
    .data(ticks_and_lable)
    .enter()
        .append('text')
        .text(d => d.angle % 30 == 0? d.angle : "")
        .attr('x', d => d.text_x)
        .attr('y', d => d.text_y)
        .attr('fill', d => hsla((d.angle + cur_color_angle) % 360, 100, 40, 1))
        .attr("font-size", axis_font_size)
        .attr("text-anchor", d => d.text_anchor)
        .attr("dominant-baseline", "central")
        .attr("transform", d => `rotate(${d.text_anchor === "end" ? d.angle - 45 : d.angle -135 }, ${d.text_x}, ${d.text_y})`);

function change_color() {
    // Update color for lines
    axis.transition()
        .duration(circle_transition_time)
        .attr('stroke', d => hsla((d.angle + cur_color_angle) % 360, 100, 45, 1));

    // Update color for text
    axis_text.transition()
        .duration(circle_transition_time)
        .attr('fill', d => hsla((d.angle + cur_color_angle) % 360, 100, 45, 1));
}
