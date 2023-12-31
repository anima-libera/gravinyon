struct VertexInput {
	@location(0) position: vec3<f32>,
	@location(1) color: vec3<f32>,
	@location(2) normal: vec3<f32>,
};
struct InstanceInput {
	@location(3) position: vec2<f32>,
	@location(4) angle: f32,
	@location(5) scale: f32,
	@location(6) shade_sensitivity: f32,
};


struct VertexOutput {
	@builtin(position) screen_position: vec4<f32>,
	@location(0) color: vec4<f32>,
};

@group(0) @binding(0) var<uniform> uniform_light_direction: vec3<f32>;
@group(0) @binding(1) var<uniform> uniform_aspect_ratio: f32;

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
fn vertex_shader_main(vertex_input: VertexInput, instance_input: InstanceInput) -> VertexOutput {
	var vertex_output: VertexOutput;
	vertex_output.screen_position = vec4<f32>(vertex_input.position, 1.0);
	
	var base_angle = atan2(vertex_output.screen_position.y, vertex_output.screen_position.x);
	var new_angle = base_angle + instance_input.angle;
	var len = length(vertex_output.screen_position.xy) * instance_input.scale;
	vertex_output.screen_position.x = instance_input.position.x + cos(new_angle) * len;
	vertex_output.screen_position.y = instance_input.position.y + sin(new_angle) * len;

	var base_normal_angle = atan2(vertex_input.normal.y, vertex_input.normal.x);
	var new_normal_angle = base_normal_angle + instance_input.angle;
	var normal_2d_len = length(vertex_input.normal.xy);
	var normal = vec3(
		cos(new_normal_angle) * normal_2d_len,
		sin(new_normal_angle) * normal_2d_len,
		vertex_input.normal.z);

	vertex_output.screen_position.y *= uniform_aspect_ratio;
	vertex_output.screen_position.z = 1.0 - vertex_output.screen_position.z;

	var shade = dot(normal, uniform_light_direction);
	shade = f(shade * 4.0);

	var sensitivity = instance_input.shade_sensitivity;
	vertex_output.color = vec4<f32>(vertex_input.color * (1.0 + shade * sensitivity), 1.0);
	return vertex_output;
}

@fragment
fn fragment_shader_main(the: VertexOutput) -> @location(0) vec4<f32> {
	return the.color;
}
