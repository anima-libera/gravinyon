use std::{collections::HashMap, f32::consts::TAU};

use bytemuck::Zeroable;
use cgmath::{InnerSpace, MetricSpace};
use rand::Rng;
use wgpu::util::DeviceExt;
use winit::{
	event_loop::{ControlFlow, EventLoop},
	window::WindowBuilder,
};

/// Vertex type used for mesh in object shader.
#[derive(Copy, Clone, Debug)]
/// Certified Plain Old Data (so it can be sent to the GPU as a uniform).
#[repr(C)]
#[derive(bytemuck::Pod, bytemuck::Zeroable)]
struct ObjectVertexPod {
	position: [f32; 3],
	color: [f32; 3],
	normal: [f32; 3],
}

/// Vertex type used for mesh in shape shader.
#[derive(Copy, Clone, Debug)]
/// Certified Plain Old Data (so it can be sent to the GPU as a uniform).
#[repr(C)]
#[derive(bytemuck::Pod, bytemuck::Zeroable)]
struct ShapeVertexPod {
	position: [f32; 3],
	color: [f32; 3],
}

/// Instance type used with object shader.
#[derive(Copy, Clone, Debug)]
/// Certified Plain Old Data (so it can be sent to the GPU as a uniform).
#[repr(C)]
#[derive(bytemuck::Pod, bytemuck::Zeroable)]
struct ObjectInstancePod {
	position: [f32; 2],
	angle: f32,
	scale: f32,
	shade_sensitivity: f32,
}

/// Instance type used with shape shader.
#[derive(Copy, Clone, Debug)]
/// Certified Plain Old Data (so it can be sent to the GPU as a uniform).
#[repr(C)]
#[derive(bytemuck::Pod, bytemuck::Zeroable)]
struct ShapeInstancePod {
	position: [f32; 2],
	angle: f32,
	scale: f32,
}

/// Vector in 3D.
#[derive(Copy, Clone, Debug)]
/// Certified Plain Old Data (so it can be sent to the GPU as a uniform).
#[repr(C)]
#[derive(bytemuck::Pod, bytemuck::Zeroable)]
pub struct Vector3Pod {
	values: [f32; 3],
}

/// Vector in 2D.
#[derive(Copy, Clone, Debug)]
/// Certified Plain Old Data (so it can be sent to the GPU as a uniform).
#[repr(C)]
#[derive(bytemuck::Pod, bytemuck::Zeroable)]
pub struct Vector2Pod {
	values: [f32; 2],
}

pub fn run() {
	// Wgpu uses the `log`/`env_logger` crates to log errors and stuff,
	// and we do want to see the errors very much.
	env_logger::init();

	let event_loop = EventLoop::new();
	let window = WindowBuilder::new()
		.with_title("Gravinyon")
		.with_maximized(true)
		.with_resizable(true)
		.build(&event_loop)
		.unwrap();
	let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
		backends: wgpu::Backends::all(),
		dx12_shader_compiler: Default::default(),
	});
	let window_surface = unsafe { instance.create_surface(&window) }.unwrap();

	// Try to get a cool adapter first.
	let adapter = instance
		.enumerate_adapters(wgpu::Backends::all())
		.find(|adapter| {
			let info = adapter.get_info();
			info.device_type == wgpu::DeviceType::DiscreteGpu
				&& adapter.is_surface_supported(&window_surface)
		});
	// In case we didn't find any cool adapter, at least we can try to get a bad adapter.
	let adapter = adapter.or_else(|| {
		futures::executor::block_on(async {
			instance
				.request_adapter(&wgpu::RequestAdapterOptions {
					power_preference: wgpu::PowerPreference::HighPerformance,
					compatible_surface: Some(&window_surface),
					force_fallback_adapter: false,
				})
				.await
		})
	});
	let adapter = adapter.unwrap();

	println!("SELECTED ADAPTER:");
	dbg!(adapter.get_info());
	// At some point it could be nice to allow the user to choose their preferred adapter.
	// No one should have to struggle to make some game use the big GPU instead of the tiny one.
	println!("AVAILABLE ADAPTERS:");
	for adapter in instance.enumerate_adapters(wgpu::Backends::all()) {
		dbg!(adapter.get_info());
	}

	let (device, queue) = futures::executor::block_on(async {
		adapter
			.request_device(
				&wgpu::DeviceDescriptor {
					features: wgpu::Features::empty(),
					limits: wgpu::Limits::default(),
					label: None,
				},
				None,
			)
			.await
	})
	.unwrap();

	let surface_caps = window_surface.get_capabilities(&adapter);
	let surface_format = surface_caps
		.formats
		.iter()
		.copied()
		.find(|f| f.is_srgb())
		.unwrap_or(surface_caps.formats[0]);
	assert!(surface_caps
		.present_modes
		.contains(&wgpu::PresentMode::Fifo));
	let size = window.inner_size();
	let mut config = wgpu::SurfaceConfiguration {
		usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
		format: surface_format,
		width: size.width,
		height: size.height,
		present_mode: wgpu::PresentMode::Fifo,
		alpha_mode: surface_caps.alpha_modes[0],
		view_formats: vec![],
	};
	window_surface.configure(&device, &config);

	fn make_z_buffer_texture_view(
		device: &wgpu::Device,
		format: wgpu::TextureFormat,
		width: u32,
		height: u32,
	) -> wgpu::TextureView {
		let z_buffer_texture_description = wgpu::TextureDescriptor {
			label: Some("Z Buffer"),
			size: wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
			mip_level_count: 1,
			sample_count: 1,
			dimension: wgpu::TextureDimension::D2,
			format,
			view_formats: &[],
			usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
		};
		let z_buffer_texture = device.create_texture(&z_buffer_texture_description);
		z_buffer_texture.create_view(&wgpu::TextureViewDescriptor::default())
	}
	let z_buffer_format = wgpu::TextureFormat::Depth32Float;
	let mut z_buffer_view =
		make_z_buffer_texture_view(&device, z_buffer_format, config.width, config.height);

	let mut aspect_ratio = config.width as f32 / config.height as f32;

	struct UniformStuff {
		binding: u32,
		buffer: wgpu::Buffer,
		bind_group_layout_entry: wgpu::BindGroupLayoutEntry,
	}
	impl UniformStuff {
		fn new(
			device: &wgpu::Device,
			name: &str,
			binding: u32,
			usage: wgpu::BufferUsages,
			visibility: wgpu::ShaderStages,
			contents: &[u8],
		) -> UniformStuff {
			let buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
				label: Some(&format!("{name} Buffer")),
				contents,
				usage,
			});
			let bind_group_layout_entry = wgpu::BindGroupLayoutEntry {
				binding,
				visibility,
				ty: wgpu::BindingType::Buffer {
					ty: wgpu::BufferBindingType::Uniform,
					has_dynamic_offset: false,
					min_binding_size: None,
				},
				count: None,
			};
			UniformStuff { binding, buffer, bind_group_layout_entry }
		}

		fn bind_group_entry(&self) -> wgpu::BindGroupEntry {
			wgpu::BindGroupEntry { binding: self.binding, resource: self.buffer.as_entire_binding() }
		}
	}

	let object_shader_uniform_light_direction = UniformStuff::new(
		&device,
		"Light Direction",
		0,
		wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
		wgpu::ShaderStages::VERTEX,
		bytemuck::cast_slice(&[Vector3Pod { values: [-1.0, 0.0, 0.0] }]),
	);
	let object_shader_uniform_aspect_ratio = UniformStuff::new(
		&device,
		"Aspect Ratio",
		1,
		wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
		wgpu::ShaderStages::VERTEX,
		bytemuck::cast_slice(&[aspect_ratio]),
	);

	let object_shader_bind_group_layout =
		device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
			entries: &[
				object_shader_uniform_light_direction.bind_group_layout_entry,
				object_shader_uniform_aspect_ratio.bind_group_layout_entry,
			],
			label: Some("Object Bind Group Layout"),
		});
	let object_shader_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
		layout: &object_shader_bind_group_layout,
		entries: &[
			object_shader_uniform_light_direction.bind_group_entry(),
			object_shader_uniform_aspect_ratio.bind_group_entry(),
		],
		label: Some("Object Bind Group"),
	});

	let object_render_pipeline = {
		let object_vertex_buffer_layout = wgpu::VertexBufferLayout {
			array_stride: std::mem::size_of::<ObjectVertexPod>() as wgpu::BufferAddress,
			step_mode: wgpu::VertexStepMode::Vertex,
			attributes: &[
				wgpu::VertexAttribute {
					offset: 0,
					shader_location: 0,
					format: wgpu::VertexFormat::Float32x3,
				},
				wgpu::VertexAttribute {
					offset: std::mem::size_of::<[f32; 3]>() as wgpu::BufferAddress,
					shader_location: 1,
					format: wgpu::VertexFormat::Float32x3,
				},
				wgpu::VertexAttribute {
					offset: (std::mem::size_of::<[f32; 3]>() * 2) as wgpu::BufferAddress,
					shader_location: 2,
					format: wgpu::VertexFormat::Float32x3,
				},
			],
		};
		let object_instance_buffer_layout = wgpu::VertexBufferLayout {
			array_stride: std::mem::size_of::<ObjectInstancePod>() as wgpu::BufferAddress,
			step_mode: wgpu::VertexStepMode::Instance,
			attributes: &[
				wgpu::VertexAttribute {
					offset: 0,
					shader_location: 3,
					format: wgpu::VertexFormat::Float32x2,
				},
				wgpu::VertexAttribute {
					offset: std::mem::size_of::<[f32; 2]>() as wgpu::BufferAddress,
					shader_location: 4,
					format: wgpu::VertexFormat::Float32,
				},
				wgpu::VertexAttribute {
					offset: std::mem::size_of::<[f32; 3]>() as wgpu::BufferAddress,
					shader_location: 5,
					format: wgpu::VertexFormat::Float32,
				},
				wgpu::VertexAttribute {
					offset: std::mem::size_of::<[f32; 4]>() as wgpu::BufferAddress,
					shader_location: 6,
					format: wgpu::VertexFormat::Float32,
				},
			],
		};
		let object_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
			label: Some("Object Shader"),
			source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/object.wgsl").into()),
		});
		let object_render_pipeline_layout =
			device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
				label: Some("Object Render Pipeline Layout"),
				bind_group_layouts: &[&object_shader_bind_group_layout],
				push_constant_ranges: &[],
			});
		device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
			label: Some("Object Render Pipeline"),
			layout: Some(&object_render_pipeline_layout),
			vertex: wgpu::VertexState {
				module: &object_shader,
				entry_point: "vertex_shader_main",
				buffers: &[object_vertex_buffer_layout, object_instance_buffer_layout],
			},
			fragment: Some(wgpu::FragmentState {
				module: &object_shader,
				entry_point: "fragment_shader_main",
				targets: &[Some(wgpu::ColorTargetState {
					format: config.format,
					blend: Some(wgpu::BlendState::REPLACE),
					write_mask: wgpu::ColorWrites::ALL,
				})],
			}),
			primitive: wgpu::PrimitiveState {
				topology: wgpu::PrimitiveTopology::TriangleList,
				strip_index_format: None,
				front_face: wgpu::FrontFace::Ccw,
				cull_mode: Some(wgpu::Face::Back),
				polygon_mode: wgpu::PolygonMode::Fill,
				unclipped_depth: false,
				conservative: false,
			},
			depth_stencil: Some(wgpu::DepthStencilState {
				format: z_buffer_format,
				depth_write_enabled: true,
				depth_compare: wgpu::CompareFunction::Less,
				stencil: wgpu::StencilState::default(),
				bias: wgpu::DepthBiasState::default(),
			}),
			multisample: wgpu::MultisampleState {
				count: 1,
				mask: !0,
				alpha_to_coverage_enabled: false,
			},
			multiview: None,
		})
	};

	let shape_shader_uniform_aspect_ratio = UniformStuff::new(
		&device,
		"Aspect Ratio",
		0,
		wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
		wgpu::ShaderStages::VERTEX,
		bytemuck::cast_slice(&[aspect_ratio]),
	);

	let shape_shader_bind_group_layout =
		device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
			entries: &[shape_shader_uniform_aspect_ratio.bind_group_layout_entry],
			label: Some("Shape Bind Group Layout"),
		});
	let shape_shader_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
		layout: &shape_shader_bind_group_layout,
		entries: &[shape_shader_uniform_aspect_ratio.bind_group_entry()],
		label: Some("Shape Bind Group"),
	});

	let shape_render_pipeline = {
		let shape_vertex_buffer_layout = wgpu::VertexBufferLayout {
			array_stride: std::mem::size_of::<ShapeVertexPod>() as wgpu::BufferAddress,
			step_mode: wgpu::VertexStepMode::Vertex,
			attributes: &[
				wgpu::VertexAttribute {
					offset: 0,
					shader_location: 0,
					format: wgpu::VertexFormat::Float32x3,
				},
				wgpu::VertexAttribute {
					offset: std::mem::size_of::<[f32; 3]>() as wgpu::BufferAddress,
					shader_location: 1,
					format: wgpu::VertexFormat::Float32x3,
				},
			],
		};
		let shape_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
			label: Some("Shape Shader"),
			source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/shape.wgsl").into()),
		});
		let shape_render_pipeline_layout =
			device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
				label: Some("Shape Render Pipeline Layout"),
				bind_group_layouts: &[&shape_shader_bind_group_layout],
				push_constant_ranges: &[],
			});
		device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
			label: Some("Shape Render Pipeline"),
			layout: Some(&shape_render_pipeline_layout),
			vertex: wgpu::VertexState {
				module: &shape_shader,
				entry_point: "vertex_shader_main",
				buffers: &[shape_vertex_buffer_layout],
			},
			fragment: Some(wgpu::FragmentState {
				module: &shape_shader,
				entry_point: "fragment_shader_main",
				targets: &[Some(wgpu::ColorTargetState {
					format: config.format,
					blend: Some(wgpu::BlendState::REPLACE),
					write_mask: wgpu::ColorWrites::ALL,
				})],
			}),
			primitive: wgpu::PrimitiveState {
				topology: wgpu::PrimitiveTopology::TriangleList,
				strip_index_format: None,
				front_face: wgpu::FrontFace::Ccw,
				cull_mode: None, //Some(wgpu::Face::Back),
				polygon_mode: wgpu::PolygonMode::Fill,
				unclipped_depth: false,
				conservative: false,
			},
			depth_stencil: Some(wgpu::DepthStencilState {
				format: z_buffer_format,
				depth_write_enabled: true,
				depth_compare: wgpu::CompareFunction::Less,
				stencil: wgpu::StencilState::default(),
				bias: wgpu::DepthBiasState::default(),
			}),
			multisample: wgpu::MultisampleState {
				count: 1,
				mask: !0,
				alpha_to_coverage_enabled: false,
			},
			multiview: None,
		})
	};

	enum Object {
		Ship {
			position: cgmath::Point2<f32>,
			motion: cgmath::Vector2<f32>,
			instance_id: InstanceID,
		},
		Shot {
			position: cgmath::Point2<f32>,
			angle: f32,
			instance_id: InstanceID,
		},
		Obstacle {
			position: cgmath::Point2<f32>,
			motion: cgmath::Vector2<f32>,
			angle: f32,
			angle_rotation: f32,
			scale: f32,
			life: u32,
			instance_id: InstanceID,
		},
	}

	impl Object {
		fn is_ship(&self) -> bool {
			matches!(self, Object::Ship { .. })
		}
		fn is_shot(&self) -> bool {
			matches!(self, Object::Shot { .. })
		}
		fn is_obstacle(&self) -> bool {
			matches!(self, Object::Obstacle { .. })
		}

		fn position(&self) -> cgmath::Point2<f32> {
			match self {
				Object::Ship { position, .. } => *position,
				Object::Shot { position, .. } => *position,
				Object::Obstacle { position, .. } => *position,
			}
		}

		const SHIP_SCALE: f32 = 0.02;

		fn scale(&self) -> f32 {
			match self {
				Object::Ship { .. } => Object::SHIP_SCALE,
				Object::Shot { .. } => 0.01,
				Object::Obstacle { scale, .. } => *scale,
			}
		}

		fn instance_ids(&self) -> impl Iterator<Item = InstanceID> {
			match self {
				Object::Ship { instance_id, .. } => std::iter::once(*instance_id),
				Object::Shot { instance_id, .. } => std::iter::once(*instance_id),
				Object::Obstacle { instance_id, .. } => std::iter::once(*instance_id),
			}
		}

		fn collide_with(&self, other: &Object) -> bool {
			// TODO: Make something serious that check for collision of triangles of the mesh projected
			// onto the screen plane.

			let distance = self.position().distance(other.position());
			distance < self.scale() + other.scale()
		}
	}

	#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
	enum WhichMesh {
		Obstacle,
		Shot,
		Ship,
	}

	let mut obstacle_mesh = Vec::new();
	let mut add_triangle = |positions: [[f32; 3]; 3]| {
		let a: cgmath::Vector3<f32> = positions[0].into();
		let b: cgmath::Vector3<f32> = positions[1].into();
		let c: cgmath::Vector3<f32> = positions[2].into();
		let normal = (a - b).cross(c - b).normalize();
		let normal: [f32; 3] = normal.into();
		let color = [0.3, 0.3, 0.3];
		obstacle_mesh.push(ObjectVertexPod { position: positions[0], color, normal });
		obstacle_mesh.push(ObjectVertexPod { position: positions[1], color, normal });
		obstacle_mesh.push(ObjectVertexPod { position: positions[2], color, normal });
	};
	let center = [0.0, 0.0, 0.1];
	let n = 5;
	for i in 0..n {
		let angle_i = i as f32 / n as f32 * TAU;
		let angle_i_plus_1 = (i + 1) as f32 / n as f32 * TAU;
		add_triangle([
			center,
			[f32::cos(angle_i), f32::sin(angle_i), 0.0],
			[f32::cos(angle_i_plus_1), f32::sin(angle_i_plus_1), 0.0],
		]);
	}
	let obstacle_vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
		label: Some("Obstacle Vertex Buffer"),
		contents: bytemuck::cast_slice(&obstacle_mesh),
		usage: wgpu::BufferUsages::VERTEX,
	});

	let mut ship_mesh = Vec::new();
	let mut add_triangle = |positions: [[f32; 3]; 3]| {
		let a: cgmath::Vector3<f32> = positions[0].into();
		let b: cgmath::Vector3<f32> = positions[1].into();
		let c: cgmath::Vector3<f32> = positions[2].into();
		let normal = (a - b).cross(c - b).normalize();
		let normal: [f32; 3] = normal.into();
		let color = [0.5, 0.2, 0.5];
		ship_mesh.push(ObjectVertexPod { position: positions[0], color, normal });
		ship_mesh.push(ObjectVertexPod { position: positions[1], color, normal });
		ship_mesh.push(ObjectVertexPod { position: positions[2], color, normal });
	};
	let center = [0.0, 0.0, 0.1];
	add_triangle([center, [1.0, -1.0, 0.0], [0.0, 1.0, 0.0]]);
	add_triangle([center, [0.0, -0.5, 0.0], [1.0, -1.0, 0.0]]);
	add_triangle([center, [-1.0, -1.0, 0.0], [0.0, -0.5, 0.0]]);
	add_triangle([center, [0.0, 1.0, 0.0], [-1.0, -1.0, 0.0]]);
	let ship_vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
		label: Some("Ship Vertex Buffer"),
		contents: bytemuck::cast_slice(&ship_mesh),
		usage: wgpu::BufferUsages::VERTEX,
	});

	let mut shot_mesh = Vec::new();
	let mut add_triangle = |positions: [[f32; 3]; 3]| {
		let a: cgmath::Vector3<f32> = positions[0].into();
		let b: cgmath::Vector3<f32> = positions[1].into();
		let c: cgmath::Vector3<f32> = positions[2].into();
		let normal = (a - b).cross(c - b).normalize();
		let normal: [f32; 3] = normal.into();
		let color = [1.0, 0.2, 0.0]; // The center has a different color.
		shot_mesh.push(ObjectVertexPod { position: positions[0], color, normal });
		let color = [1.0, 0.0, 0.0];
		shot_mesh.push(ObjectVertexPod { position: positions[1], color, normal });
		shot_mesh.push(ObjectVertexPod { position: positions[2], color, normal });
	};
	let center = [0.0, 0.0, 0.1];
	add_triangle([center, [0.6, 0.0, 0.0], [0.0, 1.5, 0.0]]);
	add_triangle([center, [0.0, -7.0, 0.0], [0.6, 0.0, 0.0]]);
	add_triangle([center, [-0.6, 0.0, 0.0], [0.0, -7.0, 0.0]]);
	add_triangle([center, [0.0, 1.5, 0.0], [-0.6, 0.0, 0.0]]);
	let shot_vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
		label: Some("Shot Vertex Buffer"),
		contents: bytemuck::cast_slice(&shot_mesh),
		usage: wgpu::BufferUsages::VERTEX,
	});

	let color = [0.0, 0.0, 0.0];
	let top_black_rectangle_mesh = vec![
		ShapeVertexPod { position: [-1.0, 1.0, 0.0], color },
		ShapeVertexPod { position: [1.0, 1.0, 0.0], color },
		ShapeVertexPod { position: [-1.0, 0.5, 0.0], color },
		ShapeVertexPod { position: [-1.0, 0.5, 0.0], color },
		ShapeVertexPod { position: [1.0, 1.0, 0.0], color },
		ShapeVertexPod { position: [1.0, 0.5, 0.0], color },
	];
	let top_black_rectangle_vertex_buffer =
		device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
			label: Some("Top Black Rectangle Vertex Buffer"),
			contents: bytemuck::cast_slice(&top_black_rectangle_mesh),
			usage: wgpu::BufferUsages::VERTEX,
		});

	let color = [0.0, 0.0, 0.0];
	let bottom_black_rectangle_mesh = vec![
		ShapeVertexPod { position: [-1.0, -1.0, 0.0], color },
		ShapeVertexPod { position: [1.0, -1.0, 0.0], color },
		ShapeVertexPod { position: [-1.0, -0.5, 0.0], color },
		ShapeVertexPod { position: [-1.0, -0.5, 0.0], color },
		ShapeVertexPod { position: [1.0, -1.0, 0.0], color },
		ShapeVertexPod { position: [1.0, -0.5, 0.0], color },
	];
	let bottom_black_rectangle_vertex_buffer =
		device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
			label: Some("Bottom Black Rectangle Vertex Buffer"),
			contents: bytemuck::cast_slice(&bottom_black_rectangle_mesh),
			usage: wgpu::BufferUsages::VERTEX,
		});

	enum MeshInstance {
		Object(ObjectInstancePod),
		Shape(ShapeInstancePod),
	}
	enum MeshInstanceVec {
		Object(Vec<ObjectInstancePod>),
		Shape(Vec<ShapeInstancePod>),
	}
	impl MeshInstanceVec {
		fn len(&self) -> usize {
			match self {
				MeshInstanceVec::Object(vec) => vec.len(),
				MeshInstanceVec::Shape(vec) => vec.len(),
			}
		}
		fn push(&mut self, instance: MeshInstance) {
			match (self, instance) {
				(MeshInstanceVec::Object(ref mut vec), MeshInstance::Object(instance)) => {
					vec.push(instance);
				},
				(MeshInstanceVec::Shape(ref mut vec), MeshInstance::Shape(instance)) => {
					vec.push(instance);
				},
				_ => panic!("instance variant does not match variant of the vec"),
			}
		}
		fn set(&mut self, index: usize, instance: MeshInstance) {
			match (self, instance) {
				(MeshInstanceVec::Object(ref mut vec), MeshInstance::Object(instance)) => {
					vec[index] = instance;
				},
				(MeshInstanceVec::Shape(ref mut vec), MeshInstance::Shape(instance)) => {
					vec[index] = instance;
				},
				_ => panic!("instance variant does not match variant of the vec"),
			}
		}
	}
	struct InstanceArrayForOneMesh {
		instances: MeshInstanceVec,
		unused_instances: Vec<bool>, // `true` means unused
		wgpu_buffer: Option<wgpu::Buffer>,
	}
	struct InstanceTable {
		table: HashMap<WhichMesh, InstanceArrayForOneMesh>,
	}

	let mut instance_table = InstanceTable { table: HashMap::new() };
	for mesh in [WhichMesh::Obstacle, WhichMesh::Ship, WhichMesh::Shot] {
		instance_table.table.insert(
			mesh,
			InstanceArrayForOneMesh {
				instances: MeshInstanceVec::Object(Vec::new()),
				unused_instances: Vec::new(),
				wgpu_buffer: None,
			},
		);
	}

	impl InstanceTable {
		fn instance_array_len(&self, mesh: WhichMesh) -> Option<usize> {
			Some(self.table.get(&mesh)?.instances.len())
		}

		fn instance_array_buffer_slice(&self, mesh: WhichMesh) -> Option<wgpu::BufferSlice> {
			Some(self.table.get(&mesh)?.wgpu_buffer.as_ref()?.slice(..))
		}
	}

	#[derive(Clone, Copy)]
	struct InstanceID {
		mesh: WhichMesh,
		instance_index: usize,
	}

	impl InstanceTable {
		fn insert_new_instance(&mut self, mesh: WhichMesh, instance: MeshInstance) -> InstanceID {
			if let Some(array) = self.table.get_mut(&mesh) {
				for index in 0..array.instances.len() {
					if array.unused_instances[index] {
						array.unused_instances[index] = false;
						array.instances.set(index, instance);
						return InstanceID { mesh, instance_index: index };
					}
				}
				array.instances.push(instance);
				array.unused_instances.push(false);
				InstanceID { mesh, instance_index: array.instances.len() - 1 }
			} else {
				panic!("The table for mesh {mesh:?} is missing");
			}
		}

		fn remove_instance(&mut self, instance_id: InstanceID) {
			// Note that a zeroed instance that has a `scale` field will have a scale of zero
			// and thus all its geometry is invisible, which is the intended effect of removing it.
			match self.table.get_mut(&instance_id.mesh).unwrap().instances {
				MeshInstanceVec::Object(ref mut vec) => {
					vec[instance_id.instance_index] = ObjectInstancePod::zeroed();
				},
				MeshInstanceVec::Shape(ref mut vec) => {
					vec[instance_id.instance_index] = ShapeInstancePod::zeroed();
				},
			}
			self
				.table
				.get_mut(&instance_id.mesh)
				.unwrap()
				.unused_instances[instance_id.instance_index] = true;
		}
	}

	let spawn_obstacles =
		|objects: &mut Vec<Object>, instance_table: &mut InstanceTable, how_many: usize| {
			for _i in 0..how_many {
				objects.push(Object::Obstacle {
					position: cgmath::Point2 { x: 1.05, y: rand::thread_rng().gen_range(-0.4..0.4) },
					angle: rand::thread_rng().gen_range(0.0..TAU),
					scale: rand::thread_rng().gen_range(0.02..0.04),
					motion: cgmath::Vector2 {
						x: rand::thread_rng().gen_range(-0.003..0.0005),
						y: rand::thread_rng().gen_range(-0.001..0.001),
					},
					angle_rotation: rand::thread_rng().gen_range((-TAU * 0.002)..(TAU * 0.002)),
					life: 21,
					instance_id: instance_table.insert_new_instance(
						WhichMesh::Obstacle,
						MeshInstance::Object(ObjectInstancePod::zeroed()),
					),
				});
			}
		};

	let init_objects =
		|objects: &mut Vec<Object>,
		 instance_table: &mut InstanceTable,
		 spawn_obstacles: &dyn Fn(&mut Vec<Object>, &mut InstanceTable, usize)| {
			*objects = Vec::new();
			objects.push(Object::Ship {
				position: (0.0, 0.0).into(),
				motion: (0.0, 0.0).into(),
				instance_id: instance_table.insert_new_instance(
					WhichMesh::Ship,
					MeshInstance::Object(ObjectInstancePod::zeroed()),
				),
			});
			spawn_obstacles(objects, instance_table, 5);
		};
	let mut objects = Vec::new();
	init_objects(&mut objects, &mut instance_table, &spawn_obstacles);

	let mut cursor_position: cgmath::Point2<f32> = (0.0, 0.0).into();

	let mut shooting = false;
	let mut shooting_delay = 0;
	let shooting_delay_max = 13;

	let mut game_over = false;
	let mut score = 0;

	use winit::event::*;
	event_loop.run(move |event, _, control_flow| match event {
		Event::WindowEvent { ref event, window_id } if window_id == window.id() => match event {
			WindowEvent::CloseRequested
			| WindowEvent::KeyboardInput {
				input:
					KeyboardInput {
						state: ElementState::Pressed,
						virtual_keycode: Some(VirtualKeyCode::Escape),
						..
					},
				..
			} => {
				println!("Window closed  Score: {score}");
				*control_flow = ControlFlow::Exit
			},

			WindowEvent::Resized(new_size) => {
				let winit::dpi::PhysicalSize { width, height } = *new_size;
				config.width = width;
				config.height = height;
				window_surface.configure(&device, &config);
				z_buffer_view = make_z_buffer_texture_view(&device, z_buffer_format, width, height);
				aspect_ratio = config.width as f32 / config.height as f32;
				queue.write_buffer(
					&object_shader_uniform_aspect_ratio.buffer,
					0,
					bytemuck::cast_slice(&[aspect_ratio]),
				);
				queue.write_buffer(
					&shape_shader_uniform_aspect_ratio.buffer,
					0,
					bytemuck::cast_slice(&[aspect_ratio]),
				);
			},

			WindowEvent::CursorMoved { position, .. } => {
				cursor_position.x = position.x as f32 / config.width as f32 * 2.0 - 1.0;
				cursor_position.y =
					(-position.y as f32 / config.height as f32 * 2.0 + 1.0) / aspect_ratio;
			},

			WindowEvent::MouseInput {
				button: MouseButton::Right,
				state: ElementState::Pressed,
				..
			} if !game_over => {
				if let Object::Ship { position, motion, .. } = objects.get_mut(0).unwrap() {
					let ship_to_cursor = cursor_position - *position;
					let ship_to_cursor_angle = f32::atan2(ship_to_cursor.y, ship_to_cursor.x);
					let force = cgmath::Vector2::<f32> {
						x: f32::cos(ship_to_cursor_angle),
						y: f32::sin(ship_to_cursor_angle),
					} * 0.003;
					*motion += force;
				} else {
					panic!();
				}
			},

			WindowEvent::MouseInput { button: MouseButton::Left, state, .. } if !game_over => {
				shooting = state == &ElementState::Pressed;
			},

			WindowEvent::MouseInput {
				button: MouseButton::Left,
				state: ElementState::Pressed,
				..
			} if game_over => {
				for dead_object in objects.iter() {
					for instance_id in dead_object.instance_ids() {
						instance_table.remove_instance(instance_id);
					}
				}

				game_over = false;
				score = 0;
				init_objects(&mut objects, &mut instance_table, &spawn_obstacles);
			},

			_ => {},
		},

		Event::MainEventsCleared => {
			if !game_over {
				if let Object::Ship { position, motion, .. } = objects.get_mut(0).unwrap() {
					let ship_to_cursor = cursor_position - *position;

					let ship_to_cursor_distance = ship_to_cursor.magnitude();
					let mut force = ship_to_cursor.normalize() / ship_to_cursor_distance.powi(2);
					force *= 0.000025;
					if force.magnitude() > 0.0001 {
						force = force.normalize() * 0.0001;
					}
					*motion += force;
				} else {
					panic!();
				}

				if 0 <= shooting_delay {
					shooting_delay -= 1;
				}
				if shooting && shooting_delay <= 0 {
					let (ship_position, ship_direction, ship_direction_left) =
						if let Object::Ship { position, .. } = objects.get(0).unwrap() {
							let ship_to_cursor = cursor_position - *position;
							let ship_to_cursor_angle = f32::atan2(ship_to_cursor.y, ship_to_cursor.x);
							let ship_position = *position;
							let ship_direction = cgmath::Vector2::<f32> {
								x: f32::cos(ship_to_cursor_angle),
								y: f32::sin(ship_to_cursor_angle),
							};
							let ship_direction_left = cgmath::Vector2::<f32> {
								x: f32::cos(ship_to_cursor_angle + TAU / 4.0),
								y: f32::sin(ship_to_cursor_angle + TAU / 4.0),
							};
							(ship_position, ship_direction, ship_direction_left)
						} else {
							panic!();
						};
					for i in 0..2 {
						let position = ship_position
							+ ship_direction * 0.035
							+ ship_direction_left * 0.016 * ((i * 2 - 1) as f32);
						let position_to_cursor = (cursor_position - position).normalize();
						let position_to_cursor_angle =
							f32::atan2(position_to_cursor.y, position_to_cursor.x);
						let shot = Object::Shot {
							position,
							angle: position_to_cursor_angle,
							instance_id: instance_table.insert_new_instance(
								WhichMesh::Shot,
								MeshInstance::Object(ObjectInstancePod::zeroed()),
							),
						};
						objects.push(shot);
					}
					shooting_delay = shooting_delay_max;
				}

				// Spawning more obstacles when one is taken down.
				let mut spawn_event = false;

				let mut dead_object_indices = Vec::new();

				'object_loop: for object_index in 0..objects.len() {
					let object = objects.get(object_index).unwrap();

					let mut object_is_shot_and_dies = false;
					let mut object_is_obstacle_and_takes_damage = 0;
					'object_pairs_loop: for other_object_index in 0..objects.len() {
						if object_index == other_object_index {
							continue 'object_pairs_loop;
						}
						let other_object = objects.get(other_object_index).unwrap();
						if object.is_shot()
							&& other_object.is_obstacle()
							&& object.collide_with(other_object)
						{
							object_is_shot_and_dies = true;
						} else if object.is_obstacle()
							&& other_object.is_shot()
							&& object.collide_with(other_object)
						{
							object_is_obstacle_and_takes_damage += 1;
						} else if object.is_ship()
							&& other_object.is_obstacle()
							&& object.collide_with(other_object)
						{
							game_over = true;
							println!("Game over >w<  Score: {score}");
						}
					}

					let object = objects.get_mut(object_index).unwrap();

					if object_is_obstacle_and_takes_damage > 0 {
						if let Object::Obstacle { life, .. } = object {
							*life = life.saturating_sub(object_is_obstacle_and_takes_damage);
						} else {
							panic!();
						}
					}

					if object_is_shot_and_dies
						|| matches!(object, Object::Obstacle { life, .. } if *life == 0)
					{
						dead_object_indices.push(object_index);
						if object.is_obstacle() {
							score += 1;
							spawn_event = true;
						}
						continue 'object_loop;
					}

					match object {
						Object::Obstacle { position, motion, angle, angle_rotation, scale, .. } => {
							*position += *motion;
							*angle += *angle_rotation;

							if position.x <= -1.1 {
								position.x = 1.1;
							} else if position.x > 1.1 {
								position.x = -1.1;
							}
							if position.y < -0.5 + *scale {
								position.y = -0.5 + *scale;
								motion.y = f32::abs(motion.y);
							} else if position.y > 0.5 - *scale {
								position.y = 0.5 - *scale;
								motion.y = -f32::abs(motion.y);
							}
						},

						Object::Ship { position, motion, .. } => {
							*position += *motion;

							if position.x <= -1.1 {
								position.x = 1.1;
							} else if position.x > 1.1 {
								position.x = -1.1;
							}
							if position.y < -0.5 + Object::SHIP_SCALE {
								position.y = -0.5 + Object::SHIP_SCALE;
								motion.y = f32::abs(motion.y);
								*motion *= 0.95;
							} else if position.y > 0.5 - Object::SHIP_SCALE {
								position.y = 0.5 - Object::SHIP_SCALE;
								motion.y = -f32::abs(motion.y);
								*motion *= 0.95;
							}
						},

						Object::Shot { position, angle, .. } => {
							let motion =
								cgmath::Vector2::<f32> { x: f32::cos(*angle), y: f32::sin(*angle) } * 0.015;
							*position += motion;

							if position.x <= -1.1
								|| position.x > 1.1 || position.y <= -0.6
								|| position.y > 0.6
							{
								dead_object_indices.push(object_index);
								continue 'object_loop;
							}
						},
					}
				}

				dead_object_indices.sort();
				for dead_object_index in dead_object_indices.into_iter().rev() {
					let dead_object = objects.remove(dead_object_index);
					for instance_id in dead_object.instance_ids() {
						instance_table.remove_instance(instance_id);
					}
				}

				if spawn_event {
					spawn_obstacles(&mut objects, &mut instance_table, 2);
				}
			}

			let window_texture = window_surface.get_current_texture().unwrap();
			let window_texture_view = window_texture
				.texture
				.create_view(&wgpu::TextureViewDescriptor::default());

			{
				let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
					label: Some("Clear Render Encoder"),
				});
				let render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
					label: Some("Clear Render Pass"),
					color_attachments: &[Some(wgpu::RenderPassColorAttachment {
						view: &window_texture_view,
						resolve_target: None,
						ops: wgpu::Operations {
							load: wgpu::LoadOp::Clear(wgpu::Color { r: 0.02, g: 0.0, b: 0.05, a: 1.0 }),
							store: true,
						},
					})],
					depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
						view: &z_buffer_view,
						depth_ops: Some(wgpu::Operations { load: wgpu::LoadOp::Clear(1.0), store: true }),
						stencil_ops: None,
					}),
				});

				// Release `render_pass.parent` which is a ref mut to `encoder`.
				drop(render_pass);

				queue.submit(std::iter::once(encoder.finish()));
			}

			{
				for object in objects.iter() {
					if object.is_ship() && game_over {
						continue;
					}

					let position = match object {
						Object::Ship { position, .. } => position,
						Object::Shot { position, .. } => position,
						Object::Obstacle { position, .. } => position,
					};
					let mesh_angle = match object {
						Object::Ship { position, .. } => {
							let ship_to_cursor = cursor_position - *position;
							let angle = f32::atan2(ship_to_cursor.y, ship_to_cursor.x);
							angle - TAU / 4.0
						},
						Object::Shot { angle, .. } => angle - TAU / 4.0,
						Object::Obstacle { angle, .. } => *angle,
					};
					let scale = object.scale();
					let instance_id = match object {
						Object::Obstacle { instance_id, .. } => instance_id,
						Object::Shot { instance_id, .. } => instance_id,
						Object::Ship { instance_id, .. } => instance_id,
					};
					let shade_sensitivity = match object {
						Object::Obstacle { .. } | Object::Ship { .. } => 3.0,
						Object::Shot { .. } => 0.0,
					};

					let instances = &mut instance_table
						.table
						.get_mut(&instance_id.mesh)
						.unwrap()
						.instances;
					match instances {
						MeshInstanceVec::Object(ref mut vec) => {
							vec[instance_id.instance_index] = ObjectInstancePod {
								position: [position.x, position.y],
								angle: mesh_angle,
								scale,
								shade_sensitivity,
							};
						},
						MeshInstanceVec::Shape(ref mut vec) => {
							vec[instance_id.instance_index] = ShapeInstancePod {
								position: [position.x, position.y],
								angle: mesh_angle,
								scale,
							};
						},
					}
				}

				for mesh in &[WhichMesh::Obstacle, WhichMesh::Ship, WhichMesh::Shot] {
					let instances = if let MeshInstanceVec::Object(instances) =
						&instance_table.table[mesh].instances
					{
						instances
					} else {
						panic!();
					};
					let buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
						label: Some(&format!("{mesh:?} Instance Buffer")),
						contents: bytemuck::cast_slice(instances.as_slice()),
						usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
					});
					instance_table.table.get_mut(mesh).unwrap().wgpu_buffer = Some(buffer);
					if let MeshInstanceVec::Object(instances) = &instance_table.table[mesh].instances {
						queue.write_buffer(
							instance_table.table[mesh].wgpu_buffer.as_ref().unwrap(),
							0,
							bytemuck::cast_slice(instances.as_slice()),
						);
					} else {
						panic!();
					}
				}

				let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
					label: Some("Render Encoder"),
				});
				let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
					label: Some("Render Pass"),
					color_attachments: &[Some(wgpu::RenderPassColorAttachment {
						view: &window_texture_view,
						resolve_target: None,
						ops: wgpu::Operations { load: wgpu::LoadOp::Load, store: true },
					})],
					depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
						view: &z_buffer_view,
						depth_ops: Some(wgpu::Operations { load: wgpu::LoadOp::Load, store: true }),
						stencil_ops: None,
					}),
				});

				render_pass.set_pipeline(&object_render_pipeline);
				render_pass.set_bind_group(0, &object_shader_bind_group, &[]);

				for mesh in [WhichMesh::Obstacle, WhichMesh::Ship, WhichMesh::Shot] {
					let mesh_buffer = match mesh {
						WhichMesh::Obstacle => &obstacle_vertex_buffer,
						WhichMesh::Ship => &ship_vertex_buffer,
						WhichMesh::Shot => &shot_vertex_buffer,
					};
					let mesh_len = match mesh {
						WhichMesh::Obstacle => obstacle_mesh.len(),
						WhichMesh::Ship => ship_mesh.len(),
						WhichMesh::Shot => shot_mesh.len(),
					};
					render_pass.set_vertex_buffer(0, mesh_buffer.slice(..));
					render_pass
						.set_vertex_buffer(1, instance_table.instance_array_buffer_slice(mesh).unwrap());
					render_pass.draw(
						0..(mesh_len as u32),
						0..(instance_table.instance_array_len(mesh).unwrap() as u32),
					);
				}

				// Release `render_pass.parent` which is a ref mut to `encoder`.
				drop(render_pass);

				queue.submit(std::iter::once(encoder.finish()));
			}

			{
				let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
					label: Some("Render Encoder"),
				});
				let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
					label: Some("Render Pass"),
					color_attachments: &[Some(wgpu::RenderPassColorAttachment {
						view: &window_texture_view,
						resolve_target: None,
						ops: wgpu::Operations { load: wgpu::LoadOp::Load, store: true },
					})],
					depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
						view: &z_buffer_view,
						depth_ops: Some(wgpu::Operations { load: wgpu::LoadOp::Load, store: true }),
						stencil_ops: None,
					}),
				});

				render_pass.set_pipeline(&shape_render_pipeline);
				render_pass.set_bind_group(0, &shape_shader_bind_group, &[]);

				render_pass.set_vertex_buffer(0, top_black_rectangle_vertex_buffer.slice(..));
				render_pass.draw(0..(top_black_rectangle_mesh.len() as u32), 0..1);

				render_pass.set_vertex_buffer(0, bottom_black_rectangle_vertex_buffer.slice(..));
				render_pass.draw(0..(bottom_black_rectangle_mesh.len() as u32), 0..1);

				// Release `render_pass.parent` which is a ref mut to `encoder`.
				drop(render_pass);

				queue.submit(std::iter::once(encoder.finish()));
			}

			window_texture.present();
		},

		_ => {},
	});
}
