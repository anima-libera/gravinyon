struct VertexInput {
	@location(0) position: vec3<f32>,
	@location(1) color: vec3<f32>,
	@location(2) normal: vec3<f32>,
};

struct VertexOutput {
	@builtin(position) screen_position: vec4<f32>,
	@location(0) color: vec4<f32>,
};

@group(0) @binding(0) var<uniform> uniform_light_direction: vec3<f32>;
@group(0) @binding(1) var<uniform> uniform_aspect_ratio: f32;
@group(0) @binding(2) var<uniform> uniform_position: vec2<f32>;
@group(0) @binding(3) var<uniform> uniform_angle: f32;
@group(0) @binding(4) var<uniform> uniform_scale: f32;

// Negatives are mapped to 0.
// Positive x gets closer and closer to 1.
fn f(x: f32) -> f32 {
	if x <= 0.0 {
		return 0.0;
	} else {
		return 1.0 / (-(x + 1.0)) + 1.0;
	}
}

@vertex
fn vertex_shader_main(vertex_input: VertexInput) -> VertexOutput {
	var vertex_output: VertexOutput;
	vertex_output.screen_position = vec4<f32>(vertex_input.position, 1.0);
	
	var base_angle = atan2(vertex_output.screen_position.y, vertex_output.screen_position.x);
	var new_angle = base_angle + uniform_angle;
	var len = length(vertex_output.screen_position.xy) * uniform_scale;
	vertex_output.screen_position.x = uniform_position.x + cos(new_angle) * len;
	vertex_output.screen_position.y = uniform_position.y + sin(new_angle) * len;

	var base_normal_angle = atan2(vertex_input.normal.y, vertex_input.normal.x);
	var new_normal_angle = base_normal_angle + uniform_angle;
	var normal_2d_len = length(vertex_input.normal.xy);
	var normal = vec3(
		cos(new_normal_angle) * normal_2d_len,
		sin(new_normal_angle) * normal_2d_len,
		vertex_input.normal.z);

	vertex_output.screen_position.y *= uniform_aspect_ratio;
	vertex_output.screen_position.z = 1.0 - vertex_output.screen_position.z;

	var shade = dot(normal, uniform_light_direction);
	shade = f(shade * 4.0);

	vertex_output.color = vec4<f32>(vertex_input.color * (1.0 + shade * 3.0), 1.0);
	return vertex_output;
}

@fragment
fn fragment_shader_main(the: VertexOutput) -> @location(0) vec4<f32> {
	return the.color;
}
