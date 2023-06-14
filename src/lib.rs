use bytemuck::Zeroable;
use wgpu::util::DeviceExt;
use winit::{
	event_loop::{ControlFlow, EventLoop},
	window::WindowBuilder,
};

/// Vertex type used in object meshes.
#[derive(Copy, Clone, Debug)]
/// Certified Plain Old Data (so it can be sent to the GPU as a uniform).
#[repr(C)]
#[derive(bytemuck::Pod, bytemuck::Zeroable)]
struct ObjectVertexPod {
	position: [f32; 3],
	color: [f32; 3],
	normal: [f32; 3],
}

/// Vector in 3D.
#[derive(Copy, Clone, Debug)]
/// Certified Plain Old Data (so it can be sent to the GPU as a uniform).
#[repr(C)]
#[derive(bytemuck::Pod, bytemuck::Zeroable)]
pub struct Vector3Pod {
	values: [f32; 3],
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

	let light_direction_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
		label: Some("Light Direction Buffer"),
		contents: bytemuck::cast_slice(&[Vector3Pod::zeroed()]),
		usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
	});
	let light_direction_bind_group_layout =
		device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
			entries: &[wgpu::BindGroupLayoutEntry {
				binding: 0,
				visibility: wgpu::ShaderStages::VERTEX,
				ty: wgpu::BindingType::Buffer {
					ty: wgpu::BufferBindingType::Uniform,
					has_dynamic_offset: false,
					min_binding_size: None,
				},
				count: None,
			}],
			label: Some("Light Direction Bind Group Layout"),
		});
	let light_direction_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
		layout: &light_direction_bind_group_layout,
		entries: &[wgpu::BindGroupEntry {
			binding: 0,
			resource: light_direction_buffer.as_entire_binding(),
		}],
		label: Some("Light Direction Bind Group"),
	});

	let aspect_ratio = config.width as f32 / config.height as f32;
	let aspect_ratio_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
		label: Some("Aspect Ratio Buffer"),
		contents: bytemuck::cast_slice(&[aspect_ratio]),
		usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
	});
	let aspect_ratio_bind_group_layout =
		device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
			entries: &[wgpu::BindGroupLayoutEntry {
				binding: 1,
				visibility: wgpu::ShaderStages::VERTEX,
				ty: wgpu::BindingType::Buffer {
					ty: wgpu::BufferBindingType::Uniform,
					has_dynamic_offset: false,
					min_binding_size: None,
				},
				count: None,
			}],
			label: Some("Aspect Ratio Bind Group Layout"),
		});
	let aspect_ratio_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
		layout: &aspect_ratio_bind_group_layout,
		entries: &[wgpu::BindGroupEntry {
			binding: 1,
			resource: aspect_ratio_buffer.as_entire_binding(),
		}],
		label: Some("Aspect Ratio Bind Group"),
	});

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
		let object_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
			label: Some("Object Shader"),
			source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/object.wgsl").into()),
		});
		let object_render_pipeline_layout =
			device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
				label: Some("Object Render Pipeline Layout"),
				bind_group_layouts: &[
					&light_direction_bind_group_layout,
					&aspect_ratio_bind_group_layout,
				],
				push_constant_ranges: &[],
			});
		device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
			label: Some("Object Render Pipeline"),
			layout: Some(&object_render_pipeline_layout),
			vertex: wgpu::VertexState {
				module: &object_shader,
				entry_point: "vertex_shader_main",
				buffers: &[object_vertex_buffer_layout],
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

	let test_object_vertices = vec![
		ObjectVertexPod {
			position: [0.0, 0.0, 0.0],
			color: [1.0, 0.0, 0.0],
			normal: [0.0, 0.0, 0.0],
		},
		ObjectVertexPod {
			position: [1.0, 0.0, 0.0],
			color: [0.0, 1.0, 0.0],
			normal: [0.0, 0.0, 0.0],
		},
		ObjectVertexPod {
			position: [0.0, 1.0, 0.0],
			color: [0.0, 0.0, 1.0],
			normal: [0.0, 0.0, 0.0],
		},
	];
	let test_object_vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
		label: Some("Object Vertex Buffer"),
		contents: bytemuck::cast_slice(&test_object_vertices),
		usage: wgpu::BufferUsages::VERTEX,
	});

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
			} => *control_flow = ControlFlow::Exit,

			WindowEvent::Resized(new_size) => {
				let winit::dpi::PhysicalSize { width, height } = *new_size;
				config.width = width;
				config.height = height;
				window_surface.configure(&device, &config);
				z_buffer_view = make_z_buffer_texture_view(&device, z_buffer_format, width, height);
				let aspect_ratio = config.width as f32 / config.height as f32;
				queue.write_buffer(
					&aspect_ratio_buffer,
					0,
					bytemuck::cast_slice(&[aspect_ratio]),
				);
			},

			_ => {},
		},

		Event::MainEventsCleared => {
			let window_texture = window_surface.get_current_texture().unwrap();
			let window_texture_view = window_texture
				.texture
				.create_view(&wgpu::TextureViewDescriptor::default());
			let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
				label: Some("Render Encoder"),
			});
			let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
				label: Some("Render Pass"),
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

			render_pass.set_pipeline(&object_render_pipeline);
			render_pass.set_bind_group(0, &light_direction_bind_group, &[]);
			render_pass.set_bind_group(1, &aspect_ratio_bind_group, &[]);

			render_pass.set_vertex_buffer(0, test_object_vertex_buffer.slice(..));
			render_pass.draw(0..(test_object_vertices.len() as u32), 0..1);

			// Release `render_pass.parent` which is a ref mut to `encoder`.
			drop(render_pass);

			queue.submit(std::iter::once(encoder.finish()));
			window_texture.present();
		},
		_ => {},
	});
}
