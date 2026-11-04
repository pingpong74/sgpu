pub(crate) mod compile;
mod execute;
mod resources;
mod timing;

use self::compile::alias::Heap;
pub(crate) use self::resources::{InnerBuffer, InnerImage, ResourcePool};
use self::timing::NodeTimingState;
pub use self::timing::{FrameTimings, NodeTiming};
use crate::node::{ExecutionContext, Node, NodeWriter, PresentNode};
use crate::resource::{PassBufferId, PassImageId};
use compile::{GraphError, NodeDeclaration};
use execute::{CompiledGraph, RuntimeInfo};
use sgpu::{AcquiredImage, Buffer, BufferDescription, Counter, Image, ImageDescription, QueueType};

/// A graph consisting of Node
/// Each Node points to some work on the GPU
/// The `TaskGraph` handles all syncronization
pub struct TaskGraph<'g, R, D> {
    pub(crate) pool: ResourcePool,
    entries: Vec<NodeEntry<'g, R, D>>,
    present_node: Option<usize>,
    pub(crate) compiled: Option<CompiledGraph<'g, R, D>>,
    pub(crate) frames_in_flight: usize,
    node_timing_enabled: bool,
    node_timing: Option<NodeTimingState>,
}

impl<'g, R: 'g, D: 'g> TaskGraph<'g, R, D> {
    /// Create a new `TaskGraph` with `frames_in_flight` slots.
    pub fn new(frames_in_flight: usize) -> TaskGraph<'g, R, D> {
        assert!(frames_in_flight >= 1, "frame slot count must be at least 1");
        TaskGraph {
            pool: ResourcePool::new(),
            entries: Vec::new(),
            present_node: None,
            compiled: None,
            frames_in_flight,
            node_timing_enabled: false,
            node_timing: None,
        }
    }

    /// Change the number of frames in flight. Invalidates a compiled graph.
    pub fn set_frames_in_flight(&mut self, f: usize) {
        assert!(f >= 1, "frame slot count must be at least 1");
        self.frames_in_flight = f;
        self.compiled = None;
    }

    /// Add an external Image
    pub fn add_image(&mut self, image: Image) -> PassImageId {
        self.compiled = None;
        self.pool.push_image(InnerImage::External(image))
    }

    /// Create persistent images owned by the graph.
    /// If `resize` is true, the image follows `resize()` and its contents are discarded.
    pub fn create_persistent_image(&mut self, desc: &ImageDescription, resize: bool) -> PassImageId {
        self.compiled = None;
        self.pool.push_image(InnerImage::Persistent {
            image_desc: *desc,
            images: Vec::new(),
            resize,
        })
    }

    /// Create a transient image owned by the graph
    /// Used for aliasing
    pub fn create_transient_image(&mut self, desc: &ImageDescription, resize: bool) -> PassImageId {
        self.compiled = None;
        self.pool.push_image(InnerImage::Transient {
            image_desc: *desc,
            images: Vec::new(),
            resize,
        })
    }

    /// Add an external Buffer
    pub fn add_buffer(&mut self, buffer: Buffer) -> PassBufferId {
        self.compiled = None;
        self.pool.push_buffer(InnerBuffer::External(buffer))
    }

    /// Create persistent buffers owned by the graph
    /// Not used for aliasing
    pub fn create_persistent_buffer(&mut self, desc: &BufferDescription) -> PassBufferId {
        self.compiled = None;
        self.pool.push_buffer(InnerBuffer::Persistent {
            buffer_desc: *desc,
            buffers: Vec::new(),
        })
    }

    /// Create a transient buffer owned by the graph
    /// Used for aliasing
    pub fn create_transient_buffer(&mut self, desc: &BufferDescription) -> PassBufferId {
        self.compiled = None;
        self.pool.push_buffer(InnerBuffer::Transient {
            buffer_desc: *desc,
            buffers: Vec::new(),
        })
    }

    /// Add a Node
    pub fn add_node<N: Node<R, D> + 'g>(&mut self, config: N::Config) {
        self.compiled = None;
        self.entries.push(NodeEntry {
            queue: N::QUEUE,
            label: std::any::type_name::<N>(),
            build: Box::new(move |res, w| Box::new(N::declare(res, &config, w))),
        });
    }

    /// This marks the external image the app presents with
    pub fn present_target(&mut self, id: PassImageId) {
        if self.pool.get_image(id).is_graph_owned() {
            panic!("only external images may be present targets ({id:?})");
        }

        assert!(self.present_node.is_none(), "the graph already has a present target");
        assert!(self.compiled.is_none(), "present target must be marked before compiling; rebuild the graph or mark it at declare time");

        self.present_node = Some(self.entries.len());
        self.add_node::<PresentNode>(id);
    }

    /// Rebind an external image
    pub fn set_image(&mut self, id: PassImageId, image: Image) { self.pool.set_image(id, image) }

    /// Rebind an external buffer
    pub fn set_buffer(&mut self, id: PassBufferId, buffer: Buffer) { self.pool.set_buffer(id, buffer) }

    /// Resizes all resize able resources
    pub fn resize(&mut self, width: u32, height: u32) {
        sgpu::wait_idle();
        self.pool.resize_images(width, height, self.frames_in_flight);

        let Some(compiled) = &mut self.compiled else {
            return;
        };

        compiled.runtime = RuntimeInfo::new(self.frames_in_flight);
        self.pool.materialize_transients(self.frames_in_flight, &compiled.heaps);
    }

    /// Builds the task graph.
    pub fn compile(&mut self, res: &R) -> Result<(), GraphError> {
        let declarations = self
            .entries
            .iter()
            .map(|entry| {
                let mut w = NodeWriter::new(&mut self.pool);
                let built = (entry.build)(res, &mut w);
                NodeDeclaration {
                    label: entry.label,
                    queue: entry.queue,
                    images: w.image_accesses,
                    buffers: w.buffer_accesses,
                    built,
                }
            })
            .collect::<Vec<_>>();

        let (plan, heaps) = compile::build(&self.pool, &declarations)?;

        sgpu::wait_idle();
        self.pool.materialize_persistent_images(self.frames_in_flight, false);
        self.pool.materialize_persistent_buffers(self.frames_in_flight);
        self.pool.materialize_transients(self.frames_in_flight, &heaps);

        self.compiled = Some(CompiledGraph {
            declarations,
            plan,
            heaps,
            runtime: RuntimeInfo::new(self.frames_in_flight),
        });

        Ok(())
    }

    /// Execute this taskgraph, requires self.compile first
    pub fn execute(&mut self, data: &D) -> Counter { self.compiled.as_mut().expect("TaskGraph is not compiled").run(&mut self.pool, self.node_timing_enabled, &mut self.node_timing, None, data) }

    /// Execute this taskgraph on the swapchain's frame timeline
    pub fn execute_with_swapchain(&mut self, data: &D, acquired: &AcquiredImage) -> Counter { self.compiled.as_mut().expect("TaskGraph is not compiled").run(&mut self.pool, self.node_timing_enabled, &mut self.node_timing, Some(acquired), data) }

    /// Enable or disable automatic per-node GPU timing.
    pub fn set_node_timing(&mut self, enabled: bool) {
        if self.node_timing_enabled == enabled {
            return;
        }

        self.node_timing_enabled = enabled;

        if let Some(timing) = &mut self.node_timing {
            timing.written.fill(false);
        }
    }

    /// Whether automatic per-node GPU timing is enabled.
    pub fn node_timing_enabled(&self) -> bool { self.node_timing_enabled }

    /// Read timings from the most recently executed timed frame.
    pub fn node_timings(&self) -> Option<FrameTimings> {
        if !self.node_timing_enabled {
            return None;
        }

        let compiled = self.compiled.as_ref()?;
        let timing = self.node_timing.as_ref()?;

        let query_count = timing.query_count as usize;
        if !compiled.runtime.executed_once || compiled.declarations.len().checked_mul(2) != Some(query_count) || self.entries.len().checked_mul(2) != Some(query_count) || timing.pools.len() != self.frames_in_flight {
            return None;
        }

        let slot = compiled.runtime.curr_frame;
        if !timing.written.get(slot).copied().unwrap_or(false) {
            return None;
        }

        let timestamps = timing.pools.get(slot)?.get_timestamps_ns(0, timing.query_count);
        let present_node = self.present_node;
        let nodes = self
            .entries
            .iter()
            .enumerate()
            .filter(|(node, _)| Some(*node) != present_node)
            .map(|(node, entry)| {
                let first = node * 2;
                let start_ns = timestamps[first];
                let end_ns = timestamps[first + 1];

                NodeTiming {
                    node,
                    label: entry.label,
                    queue: entry.queue,
                    start_ns,
                    end_ns,
                    duration_ns: end_ns - start_ns,
                }
            })
            .collect();

        Some(FrameTimings {
            slot,
            nodes,
        })
    }
}

pub(crate) trait ErasedNode<R, D> {
    fn execute(&self, data: &D, ctx: &mut ExecutionContext<'_>);
}

impl<R, D, N: Node<R, D>> ErasedNode<R, D> for N {
    fn execute(&self, data: &D, ctx: &mut ExecutionContext<'_>) { Node::execute(self, data, ctx) }
}

struct NodeEntry<'g, R, D> {
    queue: QueueType,
    label: &'static str,
    build: Box<dyn Fn(&R, &mut NodeWriter) -> Box<dyn ErasedNode<R, D> + 'g> + 'g>,
}
